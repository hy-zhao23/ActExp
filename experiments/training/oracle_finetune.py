"""
QA + LoRA SFT，**from-scratch**（adapter 随机初始化，与 decoder-LoRA 一起从零联合训 QA）。

对照 oracle 原 QA Stage-2（`oracle_ft`）：那个 adapter warm-start 自 Stage-1（two-stage）；
本脚本去掉 Stage-1 依赖 —— adapter 随机初始化 + decoder-LoRA 一起从零训 QA。
  - 数据 = FinetuneDataset（task="qa"，chat template，label=答案，data/raw/finetune_qa_diversified）
  - adapter 随机初始化（不传 --stage1-ckpt）、decoder 加 LoRA，两者联合微调
  - 目的：直接从 activation 答题（非反演），验证 QA 是否也不需要 Stage-1 warm-start

改自 oracle_recon_lora.py，仅换数据源 ActivationDataset(recon) → FinetuneDataset(qa) 且 task="qa"。

Launch（from-scratch = 不传 --stage1-ckpt）:
    torchrun --nproc_per_node=1 experiments/training/oracle_qa_lora.py \\
        --config experiments/training/configs/qwen3_4b_diversified_qa_lora_scratch.yaml \\
        --task activation
"""

import argparse
import contextlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.adapter import build_adapter
from utils.finetune_dataset import FinetuneDataset
from utils.train_config import OracleTrainConfig
from utils.train_utils import (
    ACT_TOKEN, TaggedDataset, compute_loss, log_metric, make_collator,
    prune_to_best_as_final,
)


def save_checkpoint(out_dir, model, tokenizer, adapter, optimizer, scheduler,
                    epoch, step, tag=""):
    label = tag or f"step_{step}"
    ckpt = out_dir / label
    ckpt.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(ckpt / "tokenizer")
    model.save_pretrained(ckpt / "lora")
    torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
    torch.save(adapter.state_dict(), ckpt / "adapter.pt")
    state = {"epoch": epoch, "global_step": step}
    (ckpt / "state.json").write_text(json.dumps(state))
    (out_dir / "train_state.json").write_text(
        json.dumps({**state, "latest_ckpt": label}))
    print(f"[save] {ckpt}")


def _eval(model, adapter, loader, device) -> float:
    model.eval(); adapter.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            total += compute_loss(model, adapter, batch, device).item()
            n += 1
    t = torch.tensor([total, float(n)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1]).item()


def cosine_schedule_with_min_lr(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    is_main = (global_rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=["activation"])
    parser.add_argument("--stage1-ckpt", default=None,
                        help="Stage-1 ckpt dir (contains adapter.pt) for warm-start; "
                             "omit → from-scratch（adapter 随机初始化，与 LoRA 联合从零训）")
    parser.add_argument("--init-ckpt", default=None,
                        help="continual ft：从已完成的 stage-2 ckpt（final/）载入 LoRA+adapter 继续在新数据上训；"
                             "fresh optimizer/schedule/epoch。与 resume（output_dir 内断点续跑）互不干扰，resume 优先。")
    parser.add_argument("--train-seed", type=int, default=None,
                        help="训练侧随机性 seed（LoRA init/dropout/shuffle/worker）。"
                             "数据 500k 子采样恒用 dataset.seed 不随此变；"
                             "解析后 != dataset.seed 时 output_dir 自动加 _seed{N} 后缀防覆盖。")
    args = parser.parse_args()

    cfg = OracleTrainConfig.from_yaml(args.config)
    task_cfg = cfg.tasks[args.task]
    mc, ac, dc, ftc = cfg.model, task_cfg.adapter, task_cfg.dataset, task_cfg.finetune

    # ── 训练侧随机性统一控制（multi-seed 实验用；优先级 CLI > config > dataset.seed）──
    #   覆盖: python/np 全局 RNG、torch CPU+CUDA RNG（LoRA 初始化、LoRA/adapter dropout）、
    #        DistributedSampler shuffle、DataLoader worker base seed。
    #   不覆盖: FinetuneDataset 500k 子采样（恒 sample_seed=dc.seed → 各 seed 同数据）；
    #          cuDNN 算法选择的非确定性（不影响统计意义上的 seed 方差）。
    #   注: LoRA init 各 rank 各自发生、DDP wrap 时从 rank0 broadcast 统一；
    #      CUDA RNG 按 rank 偏移 → 各 rank dropout mask 独立。
    train_seed = args.train_seed if args.train_seed is not None else \
                 (ftc.train_seed if ftc.train_seed is not None else dc.seed)
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)                                   # CPU + 全部 CUDA 设备
    torch.cuda.manual_seed(train_seed * 1000 + dist.get_rank())     # 本 rank 独立 dropout 流

    out_dir = Path(ftc.output_dir)
    if train_seed != dc.seed:
        out_dir = Path(f"{ftc.output_dir}_seed{train_seed}")        # 防覆盖默认 seed 结果
    if is_main:
        print(f"[seed] train_seed={train_seed} (data subsample seed={dc.seed})  out_dir={out_dir}")
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(Path(args.config).read_text())
    dist.barrier()

    # ── tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(mc.name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"additional_special_tokens": [ACT_TOKEN]})
    act_token_id = tokenizer.convert_tokens_to_ids(ACT_TOKEN)

    # ── dataset: qa（chat template，label=答案，data/raw/finetune_qa_diversified）──────
    #   物理 split：train → {qa_subdir}/，val → {qa_subdir}_val/（早停用；test 训练期不碰）。
    sources_filter = dc.source if isinstance(dc.source, list) else None
    train_base = FinetuneDataset(
        split="train", subdir=ftc.reps_subdir, raw_subdir=ftc.raw_subdir,
        qa_subdir=ftc.qa_subdir, sources_filter=sources_filter,
        train_subsample_ratio=ftc.train_subsample_ratio, sample_seed=dc.seed)
    eval_base = FinetuneDataset(
        split="val", subdir=ftc.reps_subdir, raw_subdir=ftc.raw_subdir,
        qa_subdir=ftc.qa_subdir, sources_filter=sources_filter, sample_seed=dc.seed)
    hidden_dim = train_base.hidden_dim
    train_ds = TaggedDataset(train_base, task="qa")
    eval_ds = TaggedDataset(eval_base, task="qa")
    # val 已是独立小 hold-out；再 cap 前 2000 控 eval 时长。
    # FT_EVAL_CAP=0 → 不 cap（全量 val，与 5 月主表 15016 口径一致）。
    EVAL_CAP = int(os.environ.get("FT_EVAL_CAP", "2000"))
    if EVAL_CAP > 0 and len(eval_ds) > EVAL_CAP:
        from torch.utils.data import Subset
        eval_ds = Subset(eval_ds, list(range(EVAL_CAP)))

    if is_main:
        print(f"[config] qa+LoRA (from-scratch)  model={mc.name}  in_dim={hidden_dim}")
        print(f"[config] lm_lr={ftc.lm_lr}  adapter_lr={ftc.adapter_lr}  n_epochs={ftc.n_epochs}")
        print(f"[config] train={len(train_ds)}  eval={len(eval_ds)}  "
              f"eff_global={ftc.batch_size * ftc.grad_accum_steps * world_size}")

    collate = make_collator(tokenizer, act_token_id, ac.n_tokens, dc.max_label_len)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=global_rank,
                                       shuffle=True, seed=train_seed, drop_last=True)
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=global_rank,
                                      shuffle=False)

    def make_train_loader(epoch):
        train_sampler.set_epoch(epoch)
        return DataLoader(train_ds, batch_size=ftc.batch_size, sampler=train_sampler,
                          collate_fn=collate, num_workers=4, pin_memory=True)

    eval_loader = DataLoader(eval_ds, batch_size=ftc.batch_size, sampler=eval_sampler,
                             collate_fn=collate, num_workers=4, pin_memory=True)

    steps_per_epoch = len(train_sampler) // ftc.batch_size
    optimizer_steps_pe = steps_per_epoch // ftc.grad_accum_steps

    # ── base LM (frozen) + vocab resize ─────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        mc.name, torch_dtype=torch.bfloat16, device_map={"": device})
    old_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        w = model.get_input_embeddings().weight
        w[old_vocab:] = w[:old_vocab].mean(dim=0, keepdim=True)
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lm_dim = model.config.hidden_size

    # ── adapter: warm-start from Stage-1（与 Stage-1 同结构）──────────────────────
    adapter = build_adapter(
        ac.type, hidden_dim, lm_dim,
        n_tokens=ac.n_tokens, dropout=ac.dropout,
        n_hidden=ac.n_hidden, bottleneck_dim=ac.bottleneck_dim,
        n_contexts=ac.n_contexts, n_heads=ac.n_heads,
        n_layers=ac.n_layers, ffn_mult=ac.ffn_mult,
        expansion=getattr(ac, "expansion", 0.5),
        res_gain=getattr(ac, "res_gain", 1.0),
    ).to(device).to(torch.bfloat16)
    if args.init_ckpt:
        # continual ft：adapter 从已完成的 stage-2 ckpt 载入（LoRA 在下面同源载入）
        init_ckpt = Path(args.init_ckpt)
        adp_path = init_ckpt / "adapter.pt"
        if not adp_path.exists():
            adp_path = init_ckpt / "adapter" / "adapter.pt"   # 旧 trainer 格式
        adapter.load_state_dict(torch.load(adp_path, map_location=device))
        if is_main:
            print(f"[init-ckpt] adapter ← {adp_path}")
    elif args.stage1_ckpt:
        stage1_ckpt = Path(args.stage1_ckpt)
        adapter.load_state_dict(torch.load(stage1_ckpt / "adapter.pt", map_location=device))
        if is_main:
            print(f"[warm-start] adapter ← {stage1_ckpt / 'adapter.pt'}")
    elif is_main:
        print("[from-scratch] adapter 随机初始化（无 warm-start），与 LoRA 联合从零训")

    # ── LoRA + resume ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=mc.lora_r, lora_alpha=mc.lora_alpha, lora_dropout=mc.lora_dropout,
        target_modules=ftc.lora_target_modules, bias="none", task_type="CAUSAL_LM")

    start_epoch, global_step, skip_steps = 0, 0, 0
    ft_state = out_dir / "train_state.json"
    ckpt_dir = None
    if ft_state.exists():
        _s = json.loads(ft_state.read_text())
        _d = out_dir / _s["latest_ckpt"]
        if _d.exists():
            ckpt_dir = _d

    if ckpt_dir is not None:
        model = PeftModel.from_pretrained(model, ckpt_dir / "lora",
                                          is_trainable=True, autocast_adapter_dtype=True)
    elif args.init_ckpt:
        # continual ft：LoRA 从 init ckpt 载入继续训；optimizer/scheduler/epoch 全新
        model = PeftModel.from_pretrained(model, Path(args.init_ckpt) / "lora",
                                          is_trainable=True, autocast_adapter_dtype=True)
        if is_main:
            print(f"[init-ckpt] lora ← {Path(args.init_ckpt) / 'lora'}（fresh optimizer/schedule）")
    else:
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    if is_main:
        model.print_trainable_parameters()

    # ── optimizer（联合：LoRA + adapter）──────────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(),   "lr": ftc.lm_lr},
        {"params": adapter.parameters(), "lr": ftc.adapter_lr},
    ], weight_decay=ftc.weight_decay)
    total_opt_steps = optimizer_steps_pe * ftc.n_epochs
    warmup_steps = int(total_opt_steps * ftc.warmup_ratio)
    scheduler = cosine_schedule_with_min_lr(optimizer, warmup_steps, total_opt_steps,
                                            ftc.min_lr_ratio)

    if ckpt_dir is not None:
        map_loc = {f"cuda:{i}": str(device) for i in range(8)}
        optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=map_loc))
        scheduler.load_state_dict(torch.load(ckpt_dir / "scheduler.pt", map_location="cpu"))
        adapter.load_state_dict(torch.load(ckpt_dir / "adapter.pt", map_location=device))
        _state = json.loads((ckpt_dir / "state.json").read_text())
        start_epoch = _state["epoch"]; global_step = _state["global_step"]
        skip_steps = (global_step % optimizer_steps_pe) * ftc.grad_accum_steps
        if is_main:
            print(f"[resume] {ckpt_dir}  epoch={start_epoch}  step={global_step}")

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    ddp_adapter = DDP(adapter, device_ids=[local_rank], find_unused_parameters=False)

    # ── early-stop state（best-eval 追踪 + patience，跨 resume）──────────────────────
    #   eval_loss 经 _eval 的 all_reduce 各 rank 一致 → best/bad 计数各 rank 同步、一起 break。
    best_eval_loss = float("inf")
    bad_eval_count = 0
    stop_patience  = getattr(ftc, "early_stop_patience", 0) or 0     # 0 = 只存 best 不 break
    stop_min_delta = getattr(ftc, "early_stop_min_delta", 1e-4)
    es_state_path  = out_dir / "early_stop_state.json"
    if ckpt_dir is not None and es_state_path.exists():
        _es = json.loads(es_state_path.read_text())
        best_eval_loss = _es.get("best_eval_loss", float("inf"))
        bad_eval_count = _es.get("bad_eval_count", 0)
    should_stop = False

    # ── training loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, ftc.n_epochs):
        if should_stop:
            break
        ddp_model.train(); ddp_adapter.train()
        loader_iter = iter(make_train_loader(epoch))
        for _ in range(skip_steps):
            next(loader_iter)
        skip_steps = 0
        accum_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader_iter):
            is_update = (batch_idx + 1) % ftc.grad_accum_steps == 0
            lm_ctx = contextlib.nullcontext() if is_update else ddp_model.no_sync()
            adp_ctx = contextlib.nullcontext() if is_update else ddp_adapter.no_sync()
            with lm_ctx, adp_ctx:
                loss = compute_loss(ddp_model, ddp_adapter, batch, device)
                (loss / ftc.grad_accum_steps).backward()
            accum_loss += loss.item()

            if is_update:
                clip_grad_norm_(list(model.parameters()) + list(adapter.parameters()),
                                ftc.grad_clip)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1
                avg_loss = accum_loss / ftc.grad_accum_steps
                accum_loss = 0.0

                if is_main:
                    if global_step % 10 == 0:
                        ppl = torch.exp(torch.tensor(avg_loss)).item()
                        print(f"epoch {epoch+1}/{ftc.n_epochs}  step {global_step}"
                              f"  loss={avg_loss:.4f}  ppl={ppl:.2f}"
                              f"  lr_lm={optimizer.param_groups[0]['lr']:.2e}")

                if global_step % ftc.eval_steps == 0:
                    eval_loss = _eval(ddp_model, ddp_adapter, eval_loader, device)
                    improved = eval_loss < best_eval_loss - stop_min_delta
                    if improved:
                        best_eval_loss = eval_loss
                        bad_eval_count = 0
                    else:
                        bad_eval_count += 1
                    if is_main:
                        eval_ppl = torch.exp(torch.tensor(eval_loss)).item()
                        log_metric(out_dir, {"step": global_step, "epoch": epoch,
                                             "train_loss": avg_loss, "eval_loss": eval_loss})
                        print(f"[eval] step {global_step}  eval_loss={eval_loss:.4f}  "
                              f"eval_ppl={eval_ppl:.2f}  best={best_eval_loss:.4f}  "
                              f"bad={bad_eval_count}/{stop_patience or '∞'}")
                        save_checkpoint(out_dir, model, tokenizer, adapter,
                                        optimizer, scheduler, epoch, global_step)
                        if improved:               # 存 best（收尾 prune_to_best_as_final 会 promote 成 final）
                            save_checkpoint(out_dir, model, tokenizer, adapter,
                                            optimizer, scheduler, epoch, global_step, tag="best")
                        es_state_path.write_text(json.dumps(
                            {"best_eval_loss": best_eval_loss, "bad_eval_count": bad_eval_count,
                             "global_step": global_step}))
                    dist.barrier()
                    ddp_model.train(); ddp_adapter.train()
                    if stop_patience > 0 and bad_eval_count >= stop_patience:
                        if is_main:
                            print(f"[early-stop] eval_loss {stop_patience} 次未改进"
                                  f"（best={best_eval_loss:.4f}）→ 停。")
                        should_stop = True
                        break

    if is_main:
        # best 改名 final、删 step_*；若从未改进则回退最新 step_* 作 final
        prune_to_best_as_final(out_dir)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
