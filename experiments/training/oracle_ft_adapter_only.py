"""
Adapter-only finetune — DDP (torchrun).

Frozen: Qwen3-4B base + v1-2 Stage 2 LoRA (loaded via --lora-ckpt).
Trainable: cross-attn (Q-Former) adapter only.

Use case: swap donor model (Llama → Gemma-3-4B / Gemma-3-12B / Yi-1.5-34B).
Since the v1-2 LoRA already learned to consume soft tokens in the Qwen
embedding space, we only retrain the adapter to map new donor activations
into that same space. Donor selection is expressed by `finetune.reps_subdir`
in the config (e.g. "finetune_diversified_gemma3_12b_it").

Launch:
    torchrun --nproc_per_node=4 experiments/training/oracle_ft_adapter_only.py \\
        --config    experiments/training/configs/qwen3_4b_v1-2_gemma3_12b_donor.yaml \\
        --task      activation \\
        --lora-ckpt checkpoints/oracle_ft_qwen3_4b_v1-2_2M_lr1e-4/best/lora
"""

import argparse
import contextlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.adapter import MLPAdapter, build_adapter
from utils.finetune_dataset import FinetuneDataset
from utils.train_config import OracleTrainConfig
from utils.train_utils import (
    ACT_TOKEN, TaggedDataset, compute_loss, log_metric, make_collator,
    prune_to_best_as_final,
)


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint_adp(
    out_dir:   Path,
    adapter:   MLPAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    step:      int,
    tag:       str = "",
) -> None:
    label = tag or f"step_{step}"
    ckpt  = out_dir / label
    ckpt.mkdir(parents=True, exist_ok=True)

    (ckpt / "adapter").mkdir(exist_ok=True)
    torch.save(adapter.state_dict(),   ckpt / "adapter" / "adapter.pt")
    torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")

    state = {"epoch": epoch, "global_step": step}
    (ckpt / "state.json").write_text(json.dumps(state))
    (out_dir / "train_state.json").write_text(
        json.dumps({**state, "latest_ckpt": label})
    )
    print(f"[save] {ckpt}")


# ── eval ──────────────────────────────────────────────────────────────────────

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


def cosine_schedule_with_min_lr(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    is_main    = (global_rank == 0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--task",       required=True, choices=["activation"])
    parser.add_argument("--lora-ckpt",  required=True,
                        help="path to frozen LoRA dir (e.g. v0-4 Stage 2 ft "
                             "final/lora); contains adapter_config.json + "
                             "adapter_model.safetensors")
    args = parser.parse_args()

    cfg      = OracleTrainConfig.from_yaml(args.config)
    task_cfg = cfg.tasks[args.task]
    mc, ac, dc, ftc = (
        cfg.model, task_cfg.adapter, task_cfg.dataset, task_cfg.finetune,
    )

    lora_dir = Path(args.lora_ckpt).resolve()
    assert lora_dir.exists(), f"LoRA ckpt not found: {lora_dir}"
    assert (lora_dir / "adapter_config.json").exists(), \
        f"missing adapter_config.json in {lora_dir}"
    assert (lora_dir / "adapter_model.safetensors").exists() \
        or (lora_dir / "adapter_model.bin").exists(), \
        f"missing adapter_model.* in {lora_dir}"

    out_dir = Path(ftc.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(Path(args.config).read_text())
        (out_dir / "lora_source.txt").write_text(str(lora_dir))
    dist.barrier()

    # ── tokenizer (same as v0-4 ft: base Qwen + ACT_TOKEN) ───────────────────
    tokenizer = AutoTokenizer.from_pretrained(mc.name)
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({"additional_special_tokens": [ACT_TOKEN]})
    act_token_id: int = tokenizer.convert_tokens_to_ids(ACT_TOKEN)

    # ── dataset (new donor's reps, picked via reps_subdir) ───────────────────
    train_ds = FinetuneDataset(split="train", subdir=ftc.reps_subdir,
                               raw_subdir=ftc.raw_subdir, qa_subdir=ftc.qa_subdir)
    eval_ds  = FinetuneDataset(split="eval",  subdir=ftc.reps_subdir,
                               raw_subdir=ftc.raw_subdir, qa_subdir=ftc.qa_subdir)

    assert len(train_ds) > 0, "FinetuneDataset returned 0 train samples"
    assert len(eval_ds)  > 0, "FinetuneDataset returned 0 eval samples"

    donor_hidden_dim = train_ds.hidden_dim
    donor_model_name = train_ds.source_model
    train_ds = TaggedDataset(train_ds, task="qa")
    eval_ds  = TaggedDataset(eval_ds,  task="qa")

    if is_main:
        print(f"[config] adapter-only ft  donor={donor_model_name}"
              f"  hidden={donor_hidden_dim}")
        print(f"[config] reps_subdir={ftc.reps_subdir}  lora_ckpt={lora_dir}")
        print(f"[config] adapter_lr={ftc.adapter_lr}  n_epochs={ftc.n_epochs}")
        print(f"[config] batch={ftc.batch_size}  grad_accum={ftc.grad_accum_steps}"
              f"  effective_global={ftc.batch_size * ftc.grad_accum_steps * world_size}")
        print(f"[config] train={len(train_ds)}  eval={len(eval_ds)}")

    collate = make_collator(tokenizer, act_token_id, ac.n_tokens, dc.max_label_len)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=global_rank,
        shuffle=True, seed=dc.seed, drop_last=True,
    )
    eval_sampler = DistributedSampler(
        eval_ds, num_replicas=world_size, rank=global_rank, shuffle=False,
    )

    def make_train_loader(epoch: int, skip_samples: int = 0) -> DataLoader:
        train_sampler.set_epoch(epoch)
        sampler = train_sampler
        if skip_samples > 0:
            sampler = list(train_sampler)[skip_samples:]
        return DataLoader(
            train_ds, batch_size=ftc.batch_size, sampler=sampler,
            collate_fn=collate, num_workers=4, pin_memory=True,
        )

    eval_loader = DataLoader(
        eval_ds, batch_size=ftc.batch_size, sampler=eval_sampler,
        collate_fn=collate, num_workers=4, pin_memory=True,
    )

    steps_per_epoch    = len(train_sampler) // ftc.batch_size
    optimizer_steps_pe = steps_per_epoch // ftc.grad_accum_steps

    # ── base LM + frozen LoRA ────────────────────────────────────────────────
    # 1. load base Qwen (bf16)
    # 2. resize for ACT_TOKEN + mean-init (matches v0-4 training-time state)
    # 3. wrap with PeftModel from saved v0-4 LoRA (is_trainable=False)
    # 4. freeze everything, enable GC for memory
    model = AutoModelForCausalLM.from_pretrained(
        mc.name, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    old_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        w = model.get_input_embeddings().weight
        w[old_vocab:] = w[:old_vocab].mean(dim=0, keepdim=True)

    model = PeftModel.from_pretrained(
        model, lora_dir, is_trainable=False, autocast_adapter_dtype=True,
    )
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # so grads flow through frozen LM to adapter

    # fail loudly if anything in the LM is accidentally trainable
    lm_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert lm_trainable == 0, (
        f"expected 0 trainable LM params with frozen base+LoRA, got {lm_trainable:,}"
    )

    # ── adapter: fresh init (donor hidden_dim likely differs from v0-4's) ────
    lm_dim  = model.config.hidden_size
    adapter = build_adapter(
        ac.type, donor_hidden_dim, lm_dim,
        n_tokens=ac.n_tokens, dropout=ac.dropout,
        n_hidden=ac.n_hidden, n_contexts=ac.n_contexts, n_heads=ac.n_heads,
        n_layers=ac.n_layers, ffn_mult=ac.ffn_mult,
    ).to(device).to(torch.bfloat16)

    if is_main:
        lm_params  = sum(p.numel() for p in model.parameters())
        adp_params = sum(p.numel() for p in adapter.parameters())
        print(f"[params] base LM + LoRA: {lm_params:,} (frozen)")
        print(f"[params] adapter:        {adp_params:,} (trainable)")

    # ── optimizer (adapter only) ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=ftc.adapter_lr, weight_decay=ftc.weight_decay,
    )
    total_opt_steps = optimizer_steps_pe * ftc.n_epochs
    warmup_steps    = int(total_opt_steps * ftc.warmup_ratio)
    scheduler = cosine_schedule_with_min_lr(
        optimizer, warmup_steps, total_opt_steps, ftc.min_lr_ratio,
    )

    # ── resume (adapter-only ckpt) ───────────────────────────────────────────
    start_epoch, global_step = 0, 0
    skip_steps = 0
    ft_state   = out_dir / "train_state.json"
    if ft_state.exists():
        _s = json.loads(ft_state.read_text())
        _d = out_dir / _s["latest_ckpt"]
        if _d.exists() and (_d / "adapter" / "adapter.pt").exists() and _s.get("global_step", 0) > 0:
            map_loc = {f"cuda:{i}": str(device) for i in range(8)}
            adapter.load_state_dict(
                torch.load(_d / "adapter" / "adapter.pt", map_location=device)
            )
            optimizer.load_state_dict(
                torch.load(_d / "optimizer.pt", map_location=map_loc)
            )
            scheduler.load_state_dict(
                torch.load(_d / "scheduler.pt", map_location="cpu")
            )
            _state      = json.loads((_d / "state.json").read_text())
            start_epoch = _state["epoch"]
            global_step = _state["global_step"]
            skip_steps  = (global_step % optimizer_steps_pe) * ftc.grad_accum_steps
            if is_main:
                print(f"[resume] {_d}  epoch={start_epoch}  step={global_step}")

    # ── DDP wrap (adapter only — LM frozen, no grad sync needed for it) ──────
    train_adapter = DDP(adapter, device_ids=[local_rank], find_unused_parameters=False)

    # best-eval tracking — not recovered across resume; first eval after restart
    # may overwrite on-disk best/ once before catching up. Accepted tradeoff.
    best_eval_loss = float("inf")

    # ── training loop ─────────────────────────────────────────────────────────
    # model.train() so HF gradient checkpointing actually fires (it gates on
    # self.training). LoRA dropout will be active on the frozen LoRA — same
    # pattern as oracle_train.py for base-only frozen + adapter training.
    for epoch in range(start_epoch, ftc.n_epochs):
        model.train()
        train_adapter.train()

        loader_iter = iter(make_train_loader(epoch, skip_steps * ftc.batch_size))
        skip_steps = 0

        accum_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader_iter):
            is_update = (batch_idx + 1) % ftc.grad_accum_steps == 0
            adp_ctx = contextlib.nullcontext() if is_update else train_adapter.no_sync()

            with adp_ctx:
                loss = compute_loss(model, train_adapter, batch, device)
                (loss / ftc.grad_accum_steps).backward()
            accum_loss += loss.item()

            if is_update:
                clip_grad_norm_(adapter.parameters(), ftc.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss   = accum_loss / ftc.grad_accum_steps
                accum_loss = 0.0

                if is_main:
                    if global_step % 10 == 0:
                        train_ppl = torch.exp(torch.tensor(avg_loss)).item()
                        print(
                            f"epoch {epoch+1}/{ftc.n_epochs}  step {global_step}"
                            f"  loss={avg_loss:.4f}  ppl={train_ppl:.2f}"
                            f"  lr_adp={optimizer.param_groups[0]['lr']:.2e}"
                        )

                if global_step % ftc.eval_steps == 0:
                    eval_loss = _eval(model, train_adapter, eval_loader, device)
                    if is_main:
                        eval_ppl = torch.exp(torch.tensor(eval_loss)).item()
                        log_metric(out_dir, {
                            "step": global_step, "epoch": epoch,
                            "train_loss": avg_loss, "eval_loss": eval_loss,
                        })
                        print(f"[eval] step {global_step}  loss={eval_loss:.4f}  ppl={eval_ppl:.2f}")
                        save_checkpoint_adp(
                            out_dir, adapter, optimizer, scheduler,
                            epoch, global_step,
                        )
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            save_checkpoint_adp(
                                out_dir, adapter, optimizer, scheduler,
                                epoch, global_step, tag="best",
                            )
                    dist.barrier()
                    model.train()
                    train_adapter.train()

    if is_main:
        print(f"[train done] best_eval_loss={best_eval_loss:.4f}")
        prune_to_best_as_final(out_dir)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
