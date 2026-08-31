"""
Oracle pretraining — DDP (torchrun).

Launch with torchrun:
  torchrun --nproc_per_node=4 experiments/training/oracle_train.py \
      --config experiments/training/configs/qwen3_4b_v0-1.yaml \
      --task activation

global_step counts optimizer steps (post grad-accumulation), rank-0 only.
Only the MLP adapter is trained; the base LM is frozen.
"""

import argparse
import contextlib
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from utils.adapter import MLPAdapter, build_adapter
from utils.train_config import OracleTrainConfig
from utils.train_utils import (
    ACT_TOKEN, RECON_QUESTION, TaggedDataset, compute_loss, latest_checkpoint,
    load_checkpoint, log_metric, make_collator, prune_old_step_checkpoints,
    prune_to_best_as_final, save_checkpoint, subsplit,
)


def _load_frozen_tensor_from_hf(model_name: str, key: str,
                               dst: torch.Tensor, device: torch.device) -> None:
    """Load a single frozen tensor (e.g. embed_tokens.weight) from the HF
    safetensors cache into ``dst`` in-place. Every rank reads independently;
    page cache makes ranks 2..N essentially free. Avoids the rank-0 → all
    broadcast that NCCL-times-out for very large single ops on flaky IB."""
    from safetensors import safe_open
    from transformers.utils import cached_file

    try:
        index_path = cached_file(model_name, "model.safetensors.index.json")
        with open(index_path) as f:
            idx = json.load(f)
        shard_file = idx["weight_map"][key]
    except (OSError, FileNotFoundError, KeyError):
        shard_file = "model.safetensors"

    shard_path = cached_file(model_name, shard_file)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        w = f.get_tensor(key)
    with torch.no_grad():
        dst.copy_(w.to(device=device, dtype=dst.dtype))


def main() -> None:
    # ── DDP init ──────────────────────────────────────────────────────────────
    # set_device BEFORE init_process_group so NCCL binds to the correct GPU
    # when there are multiple ranks per node (else all local ranks default to cuda:0 → hang).
    local_rank  = int(os.environ["LOCAL_RANK"])
    device      = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    world_size  = dist.get_world_size()
    global_rank = dist.get_rank()
    is_main     = (global_rank == 0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task",   required=True, choices=["activation"])
    args = parser.parse_args()

    cfg      = OracleTrainConfig.from_yaml(args.config)
    task_cfg = cfg.tasks[args.task]
    mc, ac, dc, tc, ttc, io = (
        cfg.model, task_cfg.adapter, task_cfg.dataset,
        cfg.training, task_cfg.task_training, task_cfg.io,
    )

    out_dir = Path(io.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(Path(args.config).read_text())
    dist.barrier()

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(mc.name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"additional_special_tokens": [ACT_TOKEN]})
    act_token_id: int = tokenizer.convert_tokens_to_ids(ACT_TOKEN)

    # ── dataset ───────────────────────────────────────────────────────────────
    # dc.source may be a str ("wikipedia") or list (["wikipedia", "scientific"]);
    # make_activation_dataset returns ConcatDataset for multi-source. Per-source
    # seeded split ensures train/test boundary is deterministic across resumes.
    from utils.oracle_dataset import make_activation_dataset

    train_ds = make_activation_dataset(dc.source, split="train", seed=dc.seed,
                                       test_ratio=dc.test_ratio, n_samples=dc.n_samples,
                                       tokenizer=tokenizer,
                                       subdir=dc.subdir, reps_subdir=dc.reps_subdir)
    eval_ds  = make_activation_dataset(dc.source, split="test",  seed=dc.seed,
                                       test_ratio=dc.test_ratio, n_samples=dc.n_samples,
                                       tokenizer=tokenizer,
                                       subdir=dc.subdir, reps_subdir=dc.reps_subdir)

    # stage 1: recon only — direct inverse mapping; qa_ds reserved for stage 2
    recon_ds, qa_ds = subsplit(train_ds, dc.recon_ratio, dc.seed + 1000)
    train_concat    = recon_ds
    if is_main:
        print(f"[config] task={args.task}  model={mc.name}")
        print(f"[config] lm_lr={tc.lm_lr}  adapter_lr={tc.adapter_lr}  n_epochs={tc.n_epochs}")
        print(f"[config] batch_size={ttc.batch_size}  grad_accum={ttc.grad_accum_steps}  world_size={world_size}")
        print(f"[config] effective_global_batch={ttc.batch_size * ttc.grad_accum_steps * world_size}")
        print(f"[config] output_dir={io.output_dir}")
        print(f"stage 1 data: {len(recon_ds)} recon samples | {len(qa_ds)} qa samples held out for stage 2")
    collate         = make_collator(tokenizer, act_token_id, ac.n_tokens, dc.max_label_len)

    train_sampler = DistributedSampler(
        train_concat, num_replicas=world_size, rank=global_rank,
        shuffle=True, seed=dc.seed, drop_last=True,
    )
    eval_sampler = DistributedSampler(
        TaggedDataset(eval_ds, task="recon"),
        num_replicas=world_size, rank=global_rank, shuffle=False,
    )

    def make_train_loader(epoch: int, skip_samples: int = 0) -> DataLoader:
        train_sampler.set_epoch(epoch)
        sampler = train_sampler
        if skip_samples > 0:
            sampler = list(train_sampler)[skip_samples:]
        return DataLoader(train_concat, batch_size=ttc.batch_size, sampler=sampler,
                          collate_fn=collate, num_workers=4, pin_memory=True)

    eval_loader = DataLoader(TaggedDataset(eval_ds, task="recon"),
                             batch_size=ttc.batch_size, sampler=eval_sampler,
                             collate_fn=collate, num_workers=4, pin_memory=True)

    # steps per rank per epoch; drop_last ensures consistent count across ranks
    steps_per_epoch    = len(train_sampler) // ttc.batch_size
    optimizer_steps_pe = steps_per_epoch // ttc.grad_accum_steps

    # ── model (frozen) ────────────────────────────────────────────────────────
    # Three modes:
    #   default                : `device_map={"": device}`, standard DDP for adapter.
    #   `model_parallel: true` : HF `device_map="auto"` naive-PP within one process.
    #                            Launch with `--nproc_per_node=1`. KNOWN ISSUE:
    #                            backward across PP boundaries crashes for 70B+.
    #   `use_fsdp: true`       : PyTorch FSDP FULL_SHARD across all ranks.
    #                            Launch with `--nproc_per_node=N` (per-GPU rank).
    assert not (mc.model_parallel and mc.use_fsdp), \
        "model_parallel and use_fsdp are mutually exclusive"
    if mc.model_parallel:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        assert local_world_size == 1, (
            f"model_parallel=True requires `torchrun --nproc_per_node=1` "
            f"(got LOCAL_WORLD_SIZE={local_world_size}); otherwise every local "
            f"rank tries to claim all GPUs and OOMs."
        )
        n_visible = torch.cuda.device_count()
        max_memory = (
            {i: mc.max_memory_per_gpu for i in range(n_visible)}
            if mc.max_memory_per_gpu else None
        )
        model = AutoModelForCausalLM.from_pretrained(
            mc.name, torch_dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
        )
        if is_main:
            print(f"[model_parallel] sharded across {n_visible} GPUs "
                  f"(max_memory={mc.max_memory_per_gpu or 'auto'})")
    elif mc.use_fsdp:
        # Only rank 0 loads full weights from disk; others build a meta-tensor
        # skeleton. FSDP's `sync_module_states=True` will broadcast rank 0's
        # params to all ranks during wrap. Avoids N×144 GiB CPU mem blow-up
        # when N ranks share a node.
        from transformers import AutoConfig
        from accelerate import init_empty_weights
        if dist.get_rank() == 0:
            model = AutoModelForCausalLM.from_pretrained(
                mc.name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            )
        else:
            mcfg = AutoConfig.from_pretrained(mc.name)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(mcfg, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            mc.name, torch_dtype=torch.bfloat16, device_map={"": device},
        )
    # Only grow the embedding; shrink-resize on a `device_map="auto"` model
    # destroys the accelerate dispatch hooks and corrupts lm_head's output
    # shape (CE asserts `t < n_classes` on otherwise-valid labels).
    # Qwen2.5/3 ship with padded vocab (e.g. 152064 / 151936) that already
    # covers ACT_TOKEN, so no resize is needed in practice.
    old_vocab = model.get_input_embeddings().weight.shape[0]
    new_vocab = len(tokenizer)
    if new_vocab > old_vocab:
        # FSDP path uses meta tensors on non-rank-0; resize_token_embeddings
        # would diverge across ranks and break the sync_module_states broadcast.
        # Both Qwen2.5 (152064) and Qwen3 (151936) already pad vocab → no resize.
        assert not mc.use_fsdp, (
            f"FSDP path expects model vocab ({old_vocab}) ≥ tokenizer ({new_vocab}); "
            "use a base model with sufficient padding (Qwen2.5/Qwen3 both work)."
        )
        model.resize_token_embeddings(new_vocab)
        with torch.no_grad():
            w = model.get_input_embeddings().weight
            w[old_vocab:] = w[:old_vocab].mean(dim=0, keepdim=True)
    elif is_main:
        print(f"[vocab] keeping original embedding ({old_vocab}); "
              f"len(tokenizer)={new_vocab} fits, skipping resize.")

    model.requires_grad_(False)
    # FSDP requires use_reentrant=False for activation checkpointing; reentrant
    # version uses autograd in a way that conflicts with FSDP's param gather/free.
    if mc.use_fsdp:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # needed for GC with frozen model + inputs_embeds

    lm_dim = getattr(model.config, "hidden_size", None) \
             or model.config.get_text_config().hidden_size
    if is_main:
        total = sum(p.numel() for p in model.parameters())
        print(f"base model params (frozen): {total:,}")

    # ── FSDP wrap (after vocab resize, before training) ───────────────────────
    if mc.use_fsdp:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        # Keep the input embedding OUT of FSDP. Reasons:
        #   (1) compute_loss reads it directly (raw_model.get_input_embeddings()
        #       (input_ids)) outside of model.forward(). If embedding were its
        #       own FSDP unit, that standalone call would trigger _lazy_init →
        #       _is_root=True, then the next model.forward() would fail the
        #       outer FSDP's "Non-root _is_root should not have been set" assert.
        #   (2) Embedding is only ~2.5 GiB on 72B → cheap to replicate per rank.
        # Embedding is frozen → identical bytes on every rank → no sync needed,
        # ever. Each rank reads its own copy from the safetensors shard (page
        # cache makes ranks 2..N essentially free). Avoids the rank-0 → all
        # broadcast (1.245B numel = 5GB at fp32) that has been NCCL-timing-out
        # on this cluster's IB fabric for every 72B run.
        embed = model.get_input_embeddings()
        if embed.weight.is_meta:
            embed.to_empty(device=device, recurse=False)
            _load_frozen_tensor_from_hf(mc.name, "model.embed_tokens.weight",
                                       embed.weight, device)
        else:
            embed.to(device)

        # Same treatment for lm_head: vocab×hidden = same 1.245B numel as embed
        # on Qwen2.5-72B (tie_word_embeddings=false). If left to FSDP root-unit
        # sync_module_states broadcast it's another 2.5 GiB single op → same
        # hang risk. Replicate per-rank instead (negligible vs 80 GiB GPU mem).
        lm_head = model.get_output_embeddings()
        if lm_head is not None and lm_head is not embed:
            if lm_head.weight.is_meta:
                lm_head.to_empty(device=device, recurse=False)
                _load_frozen_tensor_from_hf(mc.name, "lm_head.weight",
                                           lm_head.weight, device)
            else:
                lm_head.to(device)

        # Auto-detect transformer block class (Qwen2DecoderLayer / Qwen3DecoderLayer / ...).
        transformer_cls = type(model.model.layers[0])
        wrap_policy = ModuleWrapPolicy({transformer_cls})

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Materialize meta tensors on GPU before FSDP broadcasts rank-0 weights.
        # No-op for already-real tensors (rank 0).
        def _init_meta_params(module: torch.nn.Module) -> None:
            if any(p.is_meta for p in module.parameters(recurse=False)):
                module.to_empty(device=device, recurse=False)

        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=device,
            use_orig_params=False,
            sync_module_states=True,           # rank-0 weights → broadcast
            param_init_fn=_init_meta_params,   # materialize meta tensors on other ranks
            ignored_modules=[embed] + (
                [lm_head] if (lm_head is not None and lm_head is not embed) else []
            ),
        )
        if is_main:
            print(f"[fsdp] wrapped {transformer_cls.__name__} across {world_size} ranks "
                  f"(FULL_SHARD, bf16 mixed precision, rank-0-only load, "
                  f"embedding replicated)")

    # ── adapter (trainable) ───────────────────────────────────────────────────
    adapter = build_adapter(
        ac.type, train_ds.hidden_dim, lm_dim,
        n_tokens=ac.n_tokens, dropout=ac.dropout,
        n_hidden=ac.n_hidden, bottleneck_dim=ac.bottleneck_dim,
        n_contexts=ac.n_contexts, n_heads=ac.n_heads,
        n_layers=ac.n_layers, ffn_mult=ac.ffn_mult,
    ) \
                .to(device).to(torch.bfloat16)
    adapter = DDP(adapter, device_ids=[local_rank], find_unused_parameters=False)

    if is_main:
        adp_params = sum(p.numel() for p in adapter.parameters())
        print(f"adapter params (trainable): {adp_params:,}")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=tc.adapter_lr, weight_decay=tc.weight_decay,
    )
    total_opt_steps = optimizer_steps_pe * tc.n_epochs
    warmup_steps    = int(total_opt_steps * tc.warmup_ratio)
    scheduler       = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch, global_step = 0, 0
    skip_steps = 0
    # early-stop state — restored across resume
    best_eval_loss   = float("inf")
    bad_eval_count   = 0
    stop_patience    = getattr(tc, "early_stop_patience", 3)   # n consecutive worsening evals
    stop_min_delta   = getattr(tc, "early_stop_min_delta", 1e-4)
    es_state_path    = out_dir / "early_stop_state.json"

    if io.resume:
        ckpt = latest_checkpoint(out_dir)
        if ckpt is not None:
            start_epoch, global_step = load_checkpoint(
                ckpt, model, adapter.module, optimizer, scheduler, device
            )
            skip_steps = (global_step % optimizer_steps_pe) * ttc.grad_accum_steps
            if es_state_path.exists():
                es = json.loads(es_state_path.read_text())
                best_eval_loss = es.get("best_eval_loss", float("inf"))
                bad_eval_count = es.get("bad_eval_count", 0)

    # ── training loop ─────────────────────────────────────────────────────────
    stop_training = False
    for epoch in range(start_epoch, tc.n_epochs):
        model.train()
        adapter.train()

        loader_iter = iter(make_train_loader(epoch, skip_steps * ttc.batch_size))
        skip_steps = 0

        accum_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader_iter):
            is_last_accum = (batch_idx + 1) % ttc.grad_accum_steps == 0
            adp_ctx = contextlib.nullcontext() if is_last_accum else adapter.no_sync()

            with adp_ctx:
                loss = compute_loss(model, adapter, batch, device)
                (loss / ttc.grad_accum_steps).backward()
            accum_loss += loss.item()

            if is_last_accum:
                grad_norm = clip_grad_norm_(adapter.parameters(), tc.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss   = accum_loss / ttc.grad_accum_steps
                accum_loss = 0.0

                if is_main:
                    if global_step % 25 == 0:
                        print(f"epoch {epoch+1}/{tc.n_epochs}  step {global_step}"
                              f"  loss={avg_loss:.4f}  ppl={torch.exp(torch.tensor(avg_loss)).item():.2f}"
                              f"  lr={scheduler.get_last_lr()[0]:.2e}")

                if global_step % tc.eval_steps == 0:
                    dist.barrier()   # sync before eval to avoid cross-rank drift
                    eval_loss = _eval(model, adapter, eval_loader, device)
                    if is_main:
                        eval_ppl = torch.exp(torch.tensor(eval_loss)).item()
                        log_metric(out_dir, {"step": global_step, "epoch": epoch,
                                             "train_loss": avg_loss, "eval_loss": eval_loss,
                                             "eval_ppl": eval_ppl})
                        print(f"[eval] step {global_step}  eval_loss={eval_loss:.4f}  eval_ppl={eval_ppl:.2f}")
                        save_checkpoint(out_dir, model, tokenizer, adapter.module,
                                        optimizer, scheduler, epoch, global_step)
                        # keep only the latest step_* (plus 'best', which lives separately)
                        prune_old_step_checkpoints(out_dir, keep_step=global_step)

                        # ── early-stop: save best, track consecutive worsening ──
                        if eval_loss < best_eval_loss - stop_min_delta:
                            best_eval_loss = eval_loss
                            bad_eval_count = 0
                            save_checkpoint(out_dir, model, tokenizer, adapter.module,
                                            optimizer, scheduler, epoch, global_step, tag="best")
                        else:
                            bad_eval_count += 1
                        es_state_path.write_text(json.dumps({
                            "best_eval_loss": best_eval_loss,
                            "bad_eval_count": bad_eval_count,
                            "global_step":    global_step,
                        }))
                        if bad_eval_count >= stop_patience:
                            print(f"[early-stop] eval_loss failed to improve for "
                                  f"{stop_patience} evals (best={best_eval_loss:.4f}). stopping.")
                            stop_training = True
                    # broadcast stop decision
                    stop_tensor = torch.tensor([1 if stop_training else 0], device=device)
                    dist.broadcast(stop_tensor, src=0)
                    stop_training = bool(stop_tensor.item())
                    dist.barrier()
                    if stop_training:
                        break
                    model.train()
                    adapter.train()
        if stop_training:
            break

    # summary: best checkpoint (tag="best") and its eval_loss
    best_dir = out_dir / "best"
    if is_main:
        if best_dir.exists():
            best_state = json.loads((best_dir / "state.json").read_text())
            print("\n" + "=" * 60)
            print(f"[best checkpoint] {best_dir}")
            print(f"  global_step = {best_state['global_step']}")
            print(f"  epoch       = {best_state['epoch']}")
            print(f"  eval_loss   = {best_eval_loss:.4f}")
            print("  → will be promoted to final/")
            print("=" * 60 + "\n")
        else:
            print("[warn] no 'best' checkpoint saved — will fall back to latest step")

    # qualitative eval on the best adapter (must run BEFORE prune renames best → final).
    # All ranks must participate: under FSDP every forward triggers AllGather across
    # ranks, so rank0-only generate() deadlocks at the first collective.
    eval_adapter = adapter.module
    if best_dir.exists():
        eval_adapter.load_state_dict(
            torch.load(best_dir / "adapter.pt", map_location=device)
        )
        if is_main:
            print(f"[qualitative_eval] using adapter from {best_dir}")
    _qualitative_eval(
        model, eval_adapter, eval_ds,
        tokenizer, act_token_id, ac.n_tokens, device, out_dir,
    )

    if is_main:
        # keep only best (or latest fallback) as final/; delete all step_*
        prune_to_best_as_final(out_dir)

    dist.destroy_process_group()


def _eval(model, adapter, loader, device) -> float:
    model.eval()
    adapter.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            total += compute_loss(model, adapter, batch, device).item()
            n += 1
    t = torch.tensor([total, float(n)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1]).item()


def _qualitative_eval(
    model, adapter, eval_ds, tokenizer, act_token_id, n_tokens, device, out_dir,
    n_samples: int = 100,
) -> None:
    """Generate predictions for n_samples random eval examples and save GT vs pred."""
    model.eval()
    adapter.eval()

    tagged = TaggedDataset(eval_ds, task="qa")
    indices = random.Random(42).sample(range(len(tagged)), min(n_samples, len(tagged)))

    question_ids = tokenizer.encode(RECON_QUESTION, add_special_tokens=False)
    prefix_ids   = [act_token_id] * n_tokens + question_ids
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    results = []
    with torch.no_grad():
        base_prefix = model.get_input_embeddings()(prefix_tensor)  # (1, L_prefix, D)

        for idx in indices:
            item   = tagged[idx]
            gt     = item["label"]
            vector = item["vector"]
            if not isinstance(vector, torch.Tensor):
                vector = torch.tensor(vector)
            vector = vector.unsqueeze(0).to(device, dtype=torch.bfloat16)  # (1, D)

            soft_embs = adapter(vector)                       # (1, n_tokens, D)
            embeds    = base_prefix.clone()
            embeds[:, :n_tokens, :] = soft_embs

            attn_mask = torch.ones(1, embeds.shape[1], dtype=torch.long, device=device)
            # synced_gpus=True: under FSDP every generate() step does an AllGather;
            # if ranks hit EOS at slightly different steps (fp drift across shards)
            # they desync and watchdog SIGABRTs at 600s. synced_gpus pads finished
            # sequences and keeps every rank in lockstep through the collectives.
            out_ids   = model.generate(
                inputs_embeds  = embeds,
                attention_mask = attn_mask,
                max_new_tokens = 64,
                do_sample      = False,
                synced_gpus    = dist.is_initialized() and dist.get_world_size() > 1,
            )
            pred = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            results.append({"gt": gt, "pred": pred})

    if dist.get_rank() == 0:
        out_path = out_dir / "qualitative_eval.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"\n[qualitative_eval] {len(results)} samples → {out_path}")
        for r in results[:5]:
            print(f"  GT:   {r['gt']}")
            print(f"  PRED: {r['pred']}")
            print()


if __name__ == "__main__":
    main()
