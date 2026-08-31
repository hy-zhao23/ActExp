"""
Shared training utilities for oracle pretraining.

Imported by experiments/training/oracle_train.py.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils.adapter import MLPAdapter

ACT_TOKEN = "<activation>"

# ── prompt templates ──────────────────────────────────────────────────────────
# Stage 1 (task="recon") — flat sequence, no chat template:
#
#   [<activation>×n] [question_tokens] [label_tokens] [EOS]
#   labels: [-100 × (n + q_len)]      [label_ids]    [eos_id]
#
# Stage 2 (task="qa") — Qwen3 chat template:
#
#   <|im_start|>user\n<activation>×n\n{question}<|im_end|>
#   <|im_start|>assistant\n{answer}<|im_end|>
#
#   act_pos: positions of <activation> tokens inside full_ids (found by scan).

RECON_QUESTION = "What does this representation encode?"


# ── dataset helpers ───────────────────────────────────────────────────────────

class TaggedDataset(Dataset):
    """Stamps every item from a base dataset with a task tag ("recon" or "qa")."""

    def __init__(self, base: Dataset, task: str):
        self.base = base
        self.task = task

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        return {**self.base[i], "task": self.task}


def subsplit(dataset: Dataset, recon_ratio: float, seed: int):
    """Split dataset into (reconstruction, qa) subsets — seeded, deterministic."""
    n    = len(dataset)
    perm = np.random.default_rng(seed).permutation(n).tolist()
    cut  = int(n * recon_ratio)
    return (
        TaggedDataset(Subset(dataset, perm[:cut]),  task="recon"),
        TaggedDataset(Subset(dataset, perm[cut:]),  task="qa"),
    )


# ── tokenisation ──────────────────────────────────────────────────────────────

def build_datapoint(
    item:          dict,
    tokenizer:     PreTrainedTokenizerBase,
    act_token_id:  int,
    n_tokens:      int,
    max_label_len: int | None = None,
) -> dict:
    """
    Build one training datapoint.

    Stage 1 (task="recon"): flat sequence — no chat template.
        [<activation>×n] [question_ids] [label_ids] [EOS]
        act_pos = [0 … n-1]

    Stage 2 (task="qa"): Qwen3 chat template.
        <|im_start|>user\\n<activation>×n\\n{question}<|im_end|>
        <|im_start|>assistant\\n{answer}<|im_end|>
        act_pos = positions of act_token_id found by scanning full_ids.
    """
    question = item.get("question") or RECON_QUESTION

    if item["task"] == "recon":
        # ── Stage 1: flat ──────────────────────────────────────────────────
        question_ids: list[int] = tokenizer.encode(question,      add_special_tokens=False)
        label_ids:    list[int] = tokenizer.encode(item["label"], add_special_tokens=False)
        if max_label_len is not None:
            label_ids = label_ids[:max_label_len]
        eos = tokenizer.eos_token_id
        prefix_ids = [act_token_id] * n_tokens + question_ids
        input_ids  = prefix_ids + label_ids + [eos]
        labels     = [-100] * len(prefix_ids) + label_ids + [eos]
        return {
            "input_ids": input_ids,
            "labels":    labels,
            "act_pos":   list(range(n_tokens)),
            "vector":    item["vector"],
        }

    # ── Stage 2: chat template (token-level splicing) ──────────────────────
    # Render chat template to STRING with a sentinel marking the activation
    # slot, then tokenize around it and splice in [act_token_id]*n_tokens.
    # This makes act_pos computed rather than scanned, and doesn't rely on
    # `<activation>` surviving apply_chat_template's internal re-tokenize.
    act_slot = "\x00ACT_SLOT\x00"   # sentinel; safe to split on

    answer_ids = tokenizer.encode(item["label"], add_special_tokens=False)
    if max_label_len is not None:
        answer_ids = answer_ids[:max_label_len]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    user_content = act_slot + "\n" + question
    messages_full   = [{"role": "user", "content": user_content},
                       {"role": "assistant", "content": answer}]
    messages_prompt = [{"role": "user", "content": user_content}]

    full_text: str = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    prompt_text: str = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    def _splice(text: str) -> tuple[list[int], int]:
        pre, post = text.split(act_slot)
        pre_ids  = tokenizer.encode(pre,  add_special_tokens=False)
        post_ids = tokenizer.encode(post, add_special_tokens=False)
        return pre_ids + [act_token_id] * n_tokens + post_ids, len(pre_ids)

    full_ids,   pre_len_full   = _splice(full_text)
    prompt_ids, pre_len_prompt = _splice(prompt_text)
    assert pre_len_full == pre_len_prompt, (
        f"act slot offset mismatch: full={pre_len_full} prompt={pre_len_prompt}"
    )

    assistant_start = len(prompt_ids)
    labels  = [-100] * assistant_start + full_ids[assistant_start:]
    act_pos = list(range(pre_len_full, pre_len_full + n_tokens))

    return {
        "input_ids": full_ids,
        "labels":    labels,
        "act_pos":   act_pos,
        "vector":    item["vector"],
    }


def make_collator(
    tokenizer:     PreTrainedTokenizerBase,
    act_token_id:  int,
    n_tokens:      int,
    max_label_len: int | None = None,
):
    """Returns a collate_fn that tokenizes and right-pads a batch."""
    pad_id = tokenizer.pad_token_id

    def collate(items: list[dict]) -> dict:
        built   = [build_datapoint(it, tokenizer, act_token_id, n_tokens, max_label_len) for it in items]
        max_len = max(len(b["input_ids"]) for b in built)

        input_ids, labels, attn_masks, act_positions, vectors = [], [], [], [], []
        for b in built:
            L, pad = len(b["input_ids"]), max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"]       + [-100]   * pad)
            attn_masks.append([1] * L       + [0]      * pad)
            act_positions.append(b["act_pos"])
            vectors.append(b["vector"])

        return {
            "input_ids":      torch.tensor(input_ids,     dtype=torch.long),
            "labels":         torch.tensor(labels,        dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks,    dtype=torch.bool),
            "act_positions":  torch.tensor(act_positions, dtype=torch.long),  # (B, n_tokens)
            "vectors":        torch.stack(vectors),
        }

    return collate


# ── forward pass ──────────────────────────────────────────────────────────────

def compute_loss(
    model:   torch.nn.Module,
    adapter: MLPAdapter,
    batch:   dict,
    device:  torch.device,
) -> torch.Tensor:
    """Forward pass with soft-token injection via masked addition."""
    input_ids = batch["input_ids"].to(device)
    labels    = batch["labels"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    vectors   = batch["vectors"].to(device, dtype=torch.bfloat16)
    pos       = batch["act_positions"].to(device)   # (B, n_tokens)

    B, L  = input_ids.shape
    n_tok = pos.shape[1]

    raw_model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        base_embeds = raw_model.get_input_embeddings()(input_ids)   # (B, L, D)

    soft_embs = adapter(vectors)   # (B, n_tokens, D)

    pos_mask  = torch.zeros(B, L, device=device, dtype=base_embeds.dtype)
    pos_mask.scatter_(1, pos.view(B, n_tok), 1.0)
    pos_mask  = pos_mask.unsqueeze(-1)

    soft_grid = torch.zeros_like(base_embeds)
    soft_grid.scatter_(
        1, pos.unsqueeze(-1).expand(B, n_tok, base_embeds.shape[-1]), soft_embs
    )

    embeds = base_embeds * (1.0 - pos_mask) + soft_grid

    # Don't let HF compute loss internally — when the model is sharded
    # (`device_map="auto"`), its loss_function's view/shift path spuriously
    # triggers `nll_loss: t < n_classes` on otherwise-valid labels (kernel
    # asserts on -100 ignore entries after some device-dispatch hop).
    # Compute CE manually with logits in our control instead.
    logits = model(inputs_embeds=embeds, attention_mask=attn_mask).logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ── checkpoint helpers ────────────────────────────────────────────────────────

def latest_checkpoint(out_dir: Path) -> Path | None:
    ckpts = sorted(
        [p for p in out_dir.glob("step_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    return ckpts[-1] if ckpts else None


def save_checkpoint(
    out_dir:   Path,
    model,
    tokenizer,
    adapter:   MLPAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    step:      int,
    tag:       str = "",
) -> None:
    label = tag or f"step_{step}"
    ckpt  = out_dir / label
    ckpt.mkdir(exist_ok=True)
    # model is frozen — only adapter and optimiser state need saving
    tokenizer.save_pretrained(ckpt / "tokenizer")
    # atomic write: tmp file → rename, so a SIGKILL mid-save can't corrupt the existing ckpt
    def _atomic_save(obj, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.rename(path)
    _atomic_save(adapter.state_dict(),   ckpt / "adapter.pt")
    _atomic_save(optimizer.state_dict(), ckpt / "optimizer.pt")
    _atomic_save(scheduler.state_dict(), ckpt / "scheduler.pt")
    state = {"epoch": epoch, "global_step": step}
    (ckpt / "state.json").write_text(json.dumps(state))
    (out_dir / "train_state.json").write_text(
        json.dumps({**state, "latest_ckpt": label})
    )
    print(f"[save] {ckpt}")


def load_checkpoint(
    ckpt:      Path,
    model,
    adapter:   MLPAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device:    torch.device,
) -> tuple[int, int]:
    """Returns (epoch, global_step)."""
    adapter.load_state_dict(torch.load(ckpt / "adapter.pt",    map_location=device))
    optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location=device))
    scheduler.load_state_dict(torch.load(ckpt / "scheduler.pt"))
    state = json.loads((ckpt / "state.json").read_text())
    print(f"[resume] {ckpt}  epoch={state['epoch']}  step={state['global_step']}")
    return state["epoch"], state["global_step"]


def log_metric(out_dir: Path, record: dict) -> None:
    with (out_dir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def prune_old_step_checkpoints(out_dir: Path, keep_step: int) -> None:
    """Delete all step_* checkpoints except step_{keep_step}. 'best' untouched."""
    import shutil
    for d in out_dir.glob("step_*"):
        if not d.is_dir():
            continue
        step = int(d.name.split("_")[1])
        if step != keep_step:
            shutil.rmtree(d)
            print(f"[prune] rm {d.name}")


def prune_to_best_as_final(out_dir: Path) -> None:
    """Training 收尾：把 'best' 改名成 'final'，删掉所有 step_*。

    若 'best' 不存在（eval 从未改进），回退到最新的 step_* 作为 final。
    最终 out_dir 下只剩 final/ 一个 ckpt 目录。
    """
    import shutil

    best  = out_dir / "best"
    final = out_dir / "final"

    if not best.exists():
        step_dirs = sorted(
            [p for p in out_dir.glob("step_*") if p.is_dir()],
            key=lambda p: int(p.name.split("_")[1]),
        )
        if not step_dirs:
            print("[cleanup] no checkpoints found — nothing to prune")
            return
        best = step_dirs[-1]
        print(f"[cleanup] no 'best' — promoting latest {best.name} as final")

    if final.exists():
        shutil.rmtree(final)
    shutil.move(str(best), str(final))
    print(f"[cleanup] {best.name} → final")

    for d in out_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            shutil.rmtree(d)
            print(f"[cleanup] rm {d.name}")

    final_state = json.loads((final / "state.json").read_text())
    (out_dir / "train_state.json").write_text(
        json.dumps({**final_state, "latest_ckpt": "final"})
    )
