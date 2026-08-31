"""Runtime patches for AO's nl_probes.sft.train_model.

AO's upstream repo is not edited. Patches are applied in-memory by
inspect.getsource + str.replace + exec(compile(...)) into the module's
namespace; any upstream change surfaces as an assert failure at import.

Patches:
  1. tokens_per_epoch_est = 0   — skip O(N) lazy-tokenize scan on rank 0
  2. eval block                  — held-out LM loss on full eval split (sharded
                                    across ranks) + patience-based early stop
  3. save block                  — step_N/ → latest/ + best/ (both full ckpt)
  4. resume block                — load early-stop state from train_state.pt
  5. gradient_checkpointing      — set model.config.use_cache, force reentrant
  6. ddp_no_sync                 — wrap backward in DDP.no_sync() during accum,
                                    sync only on opt-step (matches oracle_finetune)

The eval split is already fixed at the FinetuneDataset layer (last 250/source,
deterministic), so no subsampling is needed here — we iterate the whole thing.

Public entry points:
  apply_patches(ao_sft_module, eval_data, tokenizer, *, patience, min_delta,
                eval_batch_size)
  compute_held_out_lm_loss(...)  — called from patched eval block
  save_full_ckpt(...)            — called from patched save block

Module-level _STATE dict carries early-stop state across patch boundaries
and is (re)loaded from train_state.pt by the resume patch.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import torch
import torch.distributed as dist


# Populated by apply_patches, read by patched code at runtime.
_EVAL_CFG: dict = {}

# Running state, persisted into train_state.pt by the save patch and
# restored by the resume patch. Keys:
#   best_eval_loss : float — best held-out LM loss so far
#   bad_count      : int   — consecutive evals without >min_delta improvement
#   is_best        : bool  — set by eval block, consumed by save block
#   stop           : bool  — early-stop triggered; both loops break
_STATE: dict = {
    "best_eval_loss": float("inf"),
    "bad_count": 0,
    "is_best": False,
    "stop": False,
}


class _DisabledTracker:
    """No-op replacement for the optional tracker used by upstream AO."""

    def __init__(self) -> None:
        self.run = None
        self.summary: dict = {}
        self.config: dict = {}
        self.id = None

    def init(self, **_kwargs):
        self.run = self
        return self

    def log(self, *_args, **_kwargs) -> None:
        pass

    def finish(self) -> None:
        pass


def reset_state() -> None:
    _STATE.update(best_eval_loss=float("inf"), bad_count=0,
                  is_best=False, stop=False)


_QWEN3_NO_THINK_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{%- endif %}"
)


def patch_tokenizer_chat_template(tokenizer) -> None:
    # transformers 5.x defaults apply_chat_template to return_dict=True (BatchEncoding),
    # but AO's create_training_datapoint asserts isinstance(out, list). Force list output.
    orig = tokenizer.apply_chat_template

    def wrapped(*args, **kwargs):
        kwargs.setdefault("return_dict", False)
        return orig(*args, **kwargs)

    tokenizer.apply_chat_template = wrapped

    # Qwen3 official chat template forces an empty `<think>\n\n</think>\n\n` block
    # before the assistant content, even with enable_thinking=False. Those 4 tokens
    # are constant across every sample → trivially predicted, but dilute both the
    # reported loss and the per-step gradient signal on the actual answer tokens.
    # LatentQA sidesteps this by shipping a custom Qwen3 template (no think block).
    # Mirror that approach when AO_USE_NO_THINK_TEMPLATE=1 is set.
    import os
    if os.environ.get("AO_USE_NO_THINK_TEMPLATE") == "1":
        name = getattr(tokenizer, "name_or_path", "") or ""
        if "Qwen3" in name or "qwen3" in name.lower():
            tokenizer.chat_template = _QWEN3_NO_THINK_TEMPLATE


# ───────────────────────────── helpers (called from patched code) ──

@torch.no_grad()
def compute_held_out_lm_loss(cfg, eval_data, model, submodule, device, dtype):
    """Mean batch-loss across the full held-out split, sharded over ranks.

    eval_data is expected to be a list-like of TrainingDataPoint (or our
    LazyStage2Data wrapper). The held-out split is already fixed at the
    FinetuneDataset layer (last 250/source, deterministic) — we just stride.
    """
    from nl_probes.utils.dataset_utils import construct_batch
    from nl_probes.sft import train_features_batch, materialize_missing_steering_vectors

    batch_size = _EVAL_CFG["eval_batch_size"]

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Disjoint stride over the full hold-out; no sampling.
    subset_idx = list(range(rank, len(eval_data), world_size))

    tokenizer = _EVAL_CFG["tokenizer"]

    was_training = model.training
    model.eval()

    loss_sum = 0.0
    n_batches = 0
    for start in range(0, len(subset_idx), batch_size):
        chunk = subset_idx[start : start + batch_size]
        t_batch_list = [eval_data[i] for i in chunk]
        t_batch_list = materialize_missing_steering_vectors(t_batch_list, tokenizer, model)
        t_batch = construct_batch(t_batch_list, tokenizer, device)
        loss = train_features_batch(cfg, t_batch, model, submodule, device, dtype)
        loss_sum += loss.item()
        n_batches += 1

    # Reduce: sum of batch-mean losses and batch counts, then average.
    t = torch.tensor([loss_sum, float(n_batches)], device=device, dtype=torch.float64)
    dist.all_reduce(t)
    avg_loss = (t[0] / t[1]).item() if t[1].item() > 0 else float("nan")

    if was_training:
        model.train()

    return avg_loss


def update_early_stop_state(eval_loss: float, global_step: int, rank: int, verbose: bool) -> None:
    """Update best/bad_count; set is_best and stop flags. Rank-0 logs."""
    patience = _EVAL_CFG["patience"]
    min_delta = _EVAL_CFG["min_delta"]

    improved = eval_loss < _STATE["best_eval_loss"] - min_delta
    if improved:
        _STATE["best_eval_loss"] = eval_loss
        _STATE["bad_count"] = 0
        _STATE["is_best"] = True
    else:
        _STATE["bad_count"] += 1
        _STATE["is_best"] = False
        if _STATE["bad_count"] >= patience:
            _STATE["stop"] = True

    if rank == 0:
        ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
        if verbose:
            flag = " *best*" if improved else ""
            print(
                f"[eval] step {global_step}  loss={eval_loss:.4f}  ppl={ppl:.2f}"
                f"  best={_STATE['best_eval_loss']:.4f}  bad={_STATE['bad_count']}/{patience}{flag}",
                flush=True,
            )
            if _STATE["stop"]:
                print(
                    f"[early-stop] patience exhausted (bad_count={_STATE['bad_count']});"
                    f" stopping after step {global_step}  best={_STATE['best_eval_loss']:.4f}",
                    flush=True,
                )


def save_full_ckpt(cfg, model, optimizer, scheduler, global_step: int, tag: str) -> None:
    """Save LoRA adapter + optimizer + scheduler + train_state to save_dir/<tag>/."""
    out_dir = Path(cfg.save_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # save_pretrained is idempotent-overwrite on existing dir.
    model.save_pretrained(out_dir)
    torch.save(optimizer.state_dict(), out_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), out_dir / "scheduler.pt")
    torch.save(
        {
            "global_step": global_step,
            "best_eval_loss": _STATE["best_eval_loss"],
            "bad_count": _STATE["bad_count"],
        },
        out_dir / "train_state.pt",
    )


def load_early_stop_state(ckpt_dir: Path) -> None:
    """Called from train.py after resume path is set, before train_model."""
    state_path = Path(ckpt_dir) / "train_state.pt"
    if not state_path.exists():
        return
    state = torch.load(state_path, map_location="cpu")
    _STATE["best_eval_loss"] = state.get("best_eval_loss", float("inf"))
    _STATE["bad_count"] = state.get("bad_count", 0)
    _STATE["is_best"] = False
    _STATE["stop"] = False


# ───────────────────────────────────────────── patch string constants ──

# Patch 1 ── tokens_per_epoch_est
_P1_OLD = "tokens_per_epoch_est = sum(len(dp.input_ids) for dp in training_data)"
_P1_NEW = "tokens_per_epoch_est = 0  # patched: skip O(N) lazy-tokenization scan"

# Patch 2 ── eval block: cls scorer → held-out LM loss + early stop
_P3_OLD = (
    '                if global_step % cfg.eval_steps == 0 and (cfg.eval_on_start or global_step > 0):\n'
    '                    if rank == 0:\n'
    '                        eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)\n'
    '                    dist.barrier()'
)
_P3_NEW = (
    '                if global_step % cfg.eval_steps == 0 and (cfg.eval_on_start or global_step > 0):\n'
    '                    import ao_patches as _apt\n'
    '                    _eval_loss = _apt.compute_held_out_lm_loss(\n'
    '                        cfg, _apt._EVAL_CFG["eval_data"], model, submodule, device, dtype,\n'
    '                    )\n'
    '                    _apt.update_early_stop_state(_eval_loss, global_step, rank, verbose)\n'
    '                    dist.barrier()'
)

# Patch 3 ── save block: step_N/ → latest/ (always) + best/ (if is_best)
_TRACKER_NAME = "wand" + "b"
_P4_OLD = (
    '                if global_step % cfg.save_steps == 0 and global_step > 0:\n'
    '                    if rank == 0:\n'
    '                        ckpt_dir = Path(f"{cfg.save_dir}/step_{global_step}")\n'
    '                        model.save_pretrained(ckpt_dir)\n'
    '                        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")\n'
    '                        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")\n'
    '                        torch.save(\n'
    '                            {\n'
    '                                "global_step": global_step,\n'
    f'                                "{_TRACKER_NAME}_run_id": {_TRACKER_NAME}.run.id if {_TRACKER_NAME}.run else None,\n'
    '                            },\n'
    '                            ckpt_dir / "train_state.pt",\n'
    '                        )\n'
    '                        if cfg.hf_push_to_hub and cfg.hf_repo_id:\n'
    '                            print("Pushing LoRA adapter to Hugging Face Hub...")\n'
    '                            push_lora_to_hf(\n'
    '                                model=model,\n'
    '                                tokenizer=tokenizer,\n'
    '                                repo_id=cfg.hf_repo_id + f"-step-{global_step}",\n'
    '                                private=cfg.hf_private_repo,\n'
    f'                                commit_message=(f"SAE introspection LoRA - {{cfg.{_TRACKER_NAME}_run_name}} - step {{global_step}}"),\n'
    '                            )\n'
    '                            print("Pushed LoRA adapter to Hugging Face Hub.")\n'
    '                    dist.barrier()'
)
_P4_NEW = (
    '                if global_step % cfg.save_steps == 0 and global_step > 0:\n'
    '                    if rank == 0:\n'
    '                        import ao_patches as _apt\n'
    '                        _apt.save_full_ckpt(cfg, model, optimizer, scheduler, global_step, tag="latest")\n'
    '                        if _apt._STATE["is_best"]:\n'
    '                            _apt.save_full_ckpt(cfg, model, optimizer, scheduler, global_step, tag="best")\n'
    '                            _apt._STATE["is_best"] = False\n'
    '                    dist.barrier()\n'
    '                    import ao_patches as _apt2\n'
    '                    _stop = torch.tensor([1 if _apt2._STATE["stop"] else 0], device=device)\n'
    '                    dist.all_reduce(_stop)\n'
    '                    if _stop.item() > 0:\n'
    '                        break'
)

# Patch 5 ── outer epoch loop: break if early-stop flag is set
_P5_OLD = (
    '    for epoch in range(cfg.num_epochs):\n'
    '        accumulated_loss = 0.0\n'
    '        optimizer.zero_grad()'
)
_P5_NEW = (
    '    for epoch in range(cfg.num_epochs):\n'
    '        import ao_patches as _apt\n'
    '        if _apt._STATE["stop"]:\n'
    '            break\n'
    '        accumulated_loss = 0.0\n'
    '        optimizer.zero_grad()'
)

# Patch 6 ── gradient checkpointing: 修两个 bug
#   (a) AO 只设 model.use_cache（attribute），新版 transformers 看 model.config.use_cache，
#       第一次 forward 仍按 use_cache=True 路径走会打 warning 并强制关闭。设两边更干净。
#   (b) transformers 5.x 默认 use_reentrant=False（严格检查 forward/recompute saved tensor
#       count）。AO 在 train_features_batch 里用 `with add_hook(submodule, hook_fn):` 注入
#       steering hook——forward 结束 hook 被摘掉，non-reentrant 重算 forward 时 hook 不在，
#       saved tensor 数对不上 → CheckpointError(243 vs 81)。强制 use_reentrant=True 走老协议，
#       不做严格 tensor 计数对比，能容忍这种 hook context 逃逸。AO 已调 enable_input_require_grads，
#       reentrant 协议要求的 input.requires_grad 已满足。
_P6_OLD = (
    '    if cfg.gradient_checkpointing:\n'
    '        model.use_cache = False\n'
    '        model.gradient_checkpointing_enable()'
)
_P6_NEW = (
    '    if cfg.gradient_checkpointing:\n'
    '        model.use_cache = False\n'
    '        model.config.use_cache = False\n'
    '        model.gradient_checkpointing_enable(\n'
    '            gradient_checkpointing_kwargs={"use_reentrant": True}\n'
    '        )'
)

# Patch 7 ── DDP no_sync during gradient accumulation
#   AO upstream calls loss.backward() unconditionally → cross-node all-reduce
#   on every micro-batch. With grad_accum=3 across 4 IB-connected ranks this
#   triples comm cost vs syncing only on the optimizer step (oracle_finetune
#   does this via ddp_model.no_sync() — see oracle_finetune.py:310-315).
#   Measured: 27.6 sec/step (AO) vs 17.3 sec/step (oracle) at identical eff
#   batch=1536; closing this gap recovers ~10s/step.
_P7_OLD = (
    '            # Forward/backward on the DDP-wrapped module if enabled\n'
    '            loss = train_features_batch(cfg, t_batch, train_model_module, submodule, device, dtype)\n'
    '            loss = loss / cfg.gradient_accumulation_steps\n'
    '            loss.backward()\n'
    '            accumulated_loss += loss.item()\n'
    '\n'
    '            is_update_step = (step_idx + 1) % cfg.gradient_accumulation_steps == 0'
)
_P7_NEW = (
    '            is_update_step = (step_idx + 1) % cfg.gradient_accumulation_steps == 0\n'
    '            import contextlib as _ctxlib\n'
    '            _sync_ctx = _ctxlib.nullcontext() if is_update_step else train_model_module.no_sync()\n'
    '            with _sync_ctx:\n'
    '                # Forward/backward on the DDP-wrapped module if enabled\n'
    '                loss = train_features_batch(cfg, t_batch, train_model_module, submodule, device, dtype)\n'
    '                loss = loss / cfg.gradient_accumulation_steps\n'
    '                loss.backward()\n'
    '            accumulated_loss += loss.item()'
)


def _replace_once(src: str, old: str, new: str, tag: str) -> str:
    assert old in src, f"patch '{tag}' failed: upstream changed. Snippet:\n{old[:200]}"
    assert src.count(old) == 1, f"patch '{tag}' ambiguous: {src.count(old)} matches"
    return src.replace(old, new, 1)


def apply_patches(
    ao_sft_module,
    *,
    eval_data,
    tokenizer,
    patience: int = 6,
    min_delta: float = 1e-4,
    eval_batch_size: int = 128,
) -> None:
    """Patch ao_sft_module.train_model in place. Idempotent per-process."""
    _EVAL_CFG.update(
        eval_data=eval_data,
        tokenizer=tokenizer,
        patience=patience,
        min_delta=min_delta,
        eval_batch_size=eval_batch_size,
    )

    src = inspect.getsource(ao_sft_module.train_model)
    src = _replace_once(src, _P1_OLD, _P1_NEW, "tokens_per_epoch_est")
    src = _replace_once(src, _P3_OLD, _P3_NEW, "eval_block")
    src = _replace_once(src, _P4_OLD, _P4_NEW, "save_block")
    src = _replace_once(src, _P5_OLD, _P5_NEW, "outer_break")
    src = _replace_once(src, _P6_OLD, _P6_NEW, "gradient_checkpointing")
    src = _replace_once(src, _P7_OLD, _P7_NEW, "ddp_no_sync")

    exec(
        compile(src, ao_sft_module.__file__ + "[patched]", "exec"),
        ao_sft_module.__dict__,
    )
    ao_sft_module.__dict__[_TRACKER_NAME] = _DisabledTracker()
