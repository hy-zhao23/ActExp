"""
Open-generation eval — OURS method (cross-attn / MLP adapter via ACT_TOKEN).

Loads N stage2-finetune ckpts (each = LoRA + MLPAdapter) and greedy-generates
one answer per (source, sampled activation, sampled QA slot). Writes one JSONL
per source; schema is compatible with `eval_score.py`.

Usage:
    python experiments/eval/eval_gen_ours.py \\
        --config       experiments/training/configs/qwen3_4b_v1-2_2M_lr1e-4.yaml \\
        --stage1-ckpt  checkpoints/oracle_act_qwen3_4b_v1-2/ep16/final \\
        --ft-dirs      v1-2_2M_lr1e-4:checkpoints/oracle_ft_qwen3_4b_v1-2_2M_lr1e-4 \\
        --output-dir   out/eval/stage2_v1-2_2M_lr1e-4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.adapter import build_adapter
from utils.train_config import OracleTrainConfig
from utils.train_utils import ACT_TOKEN

from experiments.eval._eval_common import (
    OPEN_GEN_SOURCES,
    build_source_samples,
    hidden_dim_for,
)


def best_checkpoint(ft_dir: Path) -> Path:
    """Resolve the best-val ckpt for a finetune run.

    Training's `prune_to_best_as_final` renames the best ckpt to `final/` and
    deletes all `step_*/`. So for completed runs we just use `final/`. For
    still-running or TIMEOUT'd runs we fall back to `best/` then latest step_*.
    """
    for name in ("final", "best"):
        ckpt = ft_dir / name
        if ckpt.exists():
            best_val = None
            mp = ft_dir / "metrics.jsonl"
            if mp.exists():
                recs = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
                seen = {r["step"]: r for r in recs}
                if seen:
                    key = "val_loss" if "val_loss" in next(iter(seen.values())) else "eval_loss"
                    b = min(seen.values(), key=lambda x: x[key])
                    best_val = f"step={b['step']} {key}={b[key]:.4f} ppl={float(np.exp(b[key])):.2f}"
            print(f"[ckpt] {ft_dir.name}/{name}  ({best_val or 'no metrics'})")
            return ckpt
    step_dirs = sorted(ft_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    assert step_dirs, f"no final/ best/ step_*/ ckpt in {ft_dir}"
    print(f"[ckpt] {ft_dir.name}/{step_dirs[-1].name}  (no final/best, latest step)")
    return step_dirs[-1]


def load_base(stage1_ckpt: Path, model_name: str, hidden_dim: int,
              adapter_cfg, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(stage1_ckpt / "tokenizer")
    tokenizer.padding_side = "right"
    act_token_id = tokenizer.convert_tokens_to_ids(ACT_TOKEN)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    adapter_kwargs = adapter_cfg.model_dump() if hasattr(adapter_cfg, "model_dump") else dict(adapter_cfg)
    adapter_type   = adapter_kwargs.pop("type")
    # Gemma3 multimodal config nests text_config.hidden_size; HF text-only configs expose hidden_size directly.
    lm_dim = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    adapter_s1 = build_adapter(adapter_type, hidden_dim, lm_dim,
                               **adapter_kwargs)
    adapter_s1.load_state_dict(torch.load(stage1_ckpt / "adapter.pt", map_location=device))
    adapter_s1.to(device).to(torch.bfloat16).eval()

    return model, tokenizer, adapter_s1, act_token_id


def load_ft_variant(ft_ckpt: Path, base_model, hidden_dim: int, lm_dim: int,
                    adapter_cfg, device: torch.device, adapter_name: str,
                    random_adapter: bool = False, rand_seed: int = 0):
    """Wrap base_model with this variant's LoRA and load its trained adapter.

    When random_adapter=True we still attach the LoRA from ft_ckpt, but the
    Q-Former adapter is freshly initialised (no state dict load); rand_seed
    seeds torch's RNG just for that init so different seeds yield distinct
    random adapters.
    """
    peft_model = PeftModel.from_pretrained(
        base_model, ft_ckpt / "lora", adapter_name=adapter_name,
        is_trainable=False, autocast_adapter_dtype=True,
    )
    peft_model.eval()

    adapter_kwargs = adapter_cfg.model_dump() if hasattr(adapter_cfg, "model_dump") else dict(adapter_cfg)
    adapter_type   = adapter_kwargs.pop("type")
    if random_adapter:
        gen = torch.Generator(device="cpu").manual_seed(rand_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(rand_seed)
            adapter = build_adapter(adapter_type, hidden_dim, lm_dim, **adapter_kwargs)
        print(f"[ablation] random_adapter seed={rand_seed} (no state_dict loaded)")
    else:
        adapter = build_adapter(adapter_type, hidden_dim, lm_dim, **adapter_kwargs)
        adp_path = ft_ckpt / "adapter" / "adapter.pt"   # 5 月 trainer 格式
        if not adp_path.exists():
            adp_path = ft_ckpt / "adapter.pt"           # 恢复版 trainer 格式（ckpt 根目录）
        adapter.load_state_dict(torch.load(adp_path, map_location=device))
    adapter.to(device).to(torch.bfloat16).eval()
    return peft_model, adapter


def _chat_stop_ids(tokenizer) -> list[int]:
    """Stop ids = tokenizer.eos + any chat-template end-of-turn token the tokenizer knows.
    Gemma-3's tokenizer.eos is `<eos>` (id 1) but chat template ends with `<end_of_turn>`
    (id 106), so generate() never stops if we only pass eos_token_id. This helper unions
    in known chat-template terminators so every decoder family (Llama/Qwen/Gemma) halts."""
    ids = {tokenizer.eos_token_id}
    unk = tokenizer.unk_token_id
    for tok in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end_of_text|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != unk and tid >= 0:
            ids.add(tid)
    return sorted(ids)


@torch.inference_mode()
def generate_answer(model, tokenizer, adapter,
                    vector: torch.Tensor, question: str,
                    act_token_id: int, n_tokens: int,
                    device: torch.device, max_new_tokens: int,
                    adapter_name: str | None,
                    stop_ids: list[int] | None,
                    mode: str = "normal") -> str:
    """mode:
      "normal"      — ACT_TOKEN*n + question, soft-token patching (default).
      "no_soft"     — drop ACT_TOKEN entirely; prompt is just the question.
                      Adapter is not invoked; LoRA still applies.
      "stage1_only" — same prompt/patching as "normal", caller passes
                      stage1 adapter; adapter_name must be None (no LoRA).
    """
    if mode == "no_soft":
        prompt_ids: list[int] = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=False,
        )
        inp    = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        embeds = model.get_input_embeddings()(inp)
    else:
        user_content = ACT_TOKEN * n_tokens + "\n" + question
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=False,
        )
        inp       = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        base_emb  = model.get_input_embeddings()(inp)
        soft_embs = adapter(vector.unsqueeze(0).to(device, dtype=torch.bfloat16))

        embeds  = base_emb.clone()
        act_pos = [i for i, t in enumerate(prompt_ids) if t == act_token_id]
        for k, pos in enumerate(act_pos):
            embeds[0, pos] = soft_embs[0, k]

    attn = torch.ones(1, embeds.shape[1], dtype=torch.bool, device=device)
    if adapter_name is not None:
        model.set_adapter(adapter_name)
    gen_kwargs = dict(
        inputs_embeds=embeds, attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    if stop_ids is not None:
        gen_kwargs["eos_token_id"] = stop_ids
    out = model.generate(**gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",         required=True)
    ap.add_argument("--stage1-ckpt",    required=True)
    ap.add_argument("--ft-dirs",        nargs="*", default=[],
                    metavar="NAME:DIR",
                    help="ft dir containing metrics.jsonl; best step auto-selected")
    ap.add_argument("--exact-ckpts",    nargs="*", default=[],
                    metavar="NAME:PATH",
                    help="pin exact checkpoint step dir (skips metrics.jsonl lookup)")
    ap.add_argument("--n-samples",      type=int, default=50)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--output-dir",     required=True)
    ap.add_argument("--sources",        nargs="*", default=OPEN_GEN_SOURCES)
    ap.add_argument(
        "--ablation",
        choices=["no_soft", "random_adapter", "random_base", "stage1_only", "base_only"],
        default=None,
        help="Ablation mode. "
             "no_soft: drop ACT_TOKEN; LoRA-only QA on plain question. "
             "random_adapter: re-init the Q-Former adapter (uses --rand-seed); LoRA kept. "
             "random_base: re-init Q-Former adapter (--rand-seed) AND drop LoRA — pure base "
             "chat model + random ACT context (--ft-dirs ignored). "
             "stage1_only: stage1 adapter, NO LoRA (--ft-dirs ignored). "
             "base_only: pure base chat model, no soft tokens, no LoRA, no adapter "
             "(floor baseline; --ft-dirs ignored).",
    )
    ap.add_argument(
        "--rand-seed", type=int, default=0,
        help="Seed for random_adapter init (ignored unless --ablation=random_adapter).",
    )
    ap.add_argument(
        "--no-stop-ids", action="store_true",
        help="Do not pass explicit eos_token_id to generate(); fall back to "
             "model.generation_config defaults. Matches pre-5/23 eval behavior "
             "(Qwen3 then stops on both <|im_end|> and <|endoftext|>).",
    )
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ours] device={device}  output={out_dir}", flush=True)

    cfg = OracleTrainConfig.from_yaml(args.config)
    actc = cfg.tasks["activation"]
    n_tokens    = actc.adapter.n_tokens
    reps_subdir = actc.finetune.reps_subdir

    ablation = args.ablation
    variants_paths: dict[str, Path] = {}
    for spec in args.ft_dirs:
        name, dir_str = spec.split(":", 1)
        variants_paths[name] = best_checkpoint(Path(dir_str))
    for spec in args.exact_ckpts:
        name, path_str = spec.split(":", 1)
        variants_paths[name] = Path(path_str)
    if ablation in ("stage1_only", "base_only", "random_base"):
        # No LoRA / ft ckpt used.
        if variants_paths:
            print(f"[ablation] {ablation}: ignoring {len(variants_paths)} --ft-dirs/--exact-ckpts")
            variants_paths = {}
    else:
        assert variants_paths, "must pass at least one --ft-dirs or --exact-ckpts"

    hidden_dim = hidden_dim_for(reps_subdir)

    print("[ours] loading base + stage1 adapter…", flush=True)
    base_model, tokenizer, adapter_s1, act_token_id = load_base(
        Path(args.stage1_ckpt), cfg.model.name, hidden_dim, actc.adapter, device,
    )
    lm_dim = getattr(base_model.config, "hidden_size", None) or base_model.config.text_config.hidden_size
    if args.no_stop_ids:
        stop_ids = None
        print("[ours] generate stop_ids = None (fall back to model.generation_config)", flush=True)
    else:
        stop_ids = _chat_stop_ids(tokenizer)
        print(f"[ours] generate stop_ids = {stop_ids}", flush=True)
    if ablation:
        print(f"[ablation] mode = {ablation}"
              + (f"  rand_seed={args.rand_seed}" if ablation == "random_adapter" else ""),
              flush=True)

    # ── Build per-variant (model, adapter, adapter_name) tuples ─────────────
    # variant_models maps an output-column name → (model, adapter, lora_name|None, gen_mode)
    variant_models: dict[str, tuple] = {}
    if ablation == "stage1_only":
        # No LoRA. Use the raw base model with the stage1 adapter that was
        # already loaded by load_base(). adapter_name=None tells generate_answer
        # to skip set_adapter (no PEFT wrapper).
        variant_models["stage1_only"] = (base_model, adapter_s1, None, "normal")
    elif ablation == "base_only":
        # Floor: pure base chat model, plain question prompt (mode=no_soft),
        # no LoRA, no adapter call. adapter arg is None.
        variant_models["base_only"] = (base_model, None, None, "no_soft")
    elif ablation == "random_base":
        # Random Q-Former adapter (no state_dict) feeding ACT context into the
        # raw base chat model — no LoRA. Tests whether random soft tokens act
        # as noise on a non-finetuned decoder.
        adapter_kwargs = actc.adapter.model_dump() if hasattr(actc.adapter, "model_dump") else dict(actc.adapter)
        adapter_type   = adapter_kwargs.pop("type")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.rand_seed)
            rand_adapter = build_adapter(adapter_type, hidden_dim, lm_dim, **adapter_kwargs)
        rand_adapter.to(device).to(torch.bfloat16).eval()
        print(f"[ablation] random_base seed={args.rand_seed} (random Q-Former, no LoRA)")
        variant_models["random_base"] = (base_model, rand_adapter, None, "normal")
    else:
        gen_mode = "no_soft" if ablation == "no_soft" else "normal"
        for name, ckpt in variants_paths.items():
            print(f"[ours] loading variant '{name}' from {ckpt}…", flush=True)
            peft_model, adapter = load_ft_variant(
                ckpt, base_model, hidden_dim, lm_dim, actc.adapter, device,
                adapter_name=name,
                random_adapter=(ablation == "random_adapter"),
                rand_seed=args.rand_seed,
            )
            variant_models[name] = (peft_model, adapter, name, gen_mode)

    # Build per-source eval samples
    datasets: list[tuple[str, list[dict]]] = []
    for src in args.sources:
        samples = build_source_samples(src, args.n_samples, args.seed, reps_subdir)
        datasets.append((src, samples))
        print(f"[data] {src:24s}  {len(samples)} samples")

    summary_lines: list[str] = []
    for src, samples in datasets:
        out_path = out_dir / f"{src}.jsonl"
        # Source-level resume: skip if jsonl exists with the right number of rows.
        # Coarse-grained (file or nothing) but enough for preemption + manual reruns.
        if out_path.exists():
            existing = sum(1 for _ in out_path.open()) if out_path.stat().st_size else 0
            if existing >= len(samples):
                print(f"\n[ours] source={src}  skip (already have {existing}/{len(samples)} rows)")
                summary_lines.append(f"{src}: {existing} → {out_path.name} (resumed)")
                continue
            else:
                print(f"\n[ours] source={src}  redoing (existing={existing}/{len(samples)} incomplete)")
        print(f"\n{'='*70}\n[ours] source={src}  n={len(samples)}")
        with out_path.open("w") as fout:
            for i, s in enumerate(samples):
                record = {
                    "source": src, "idx": s["idx"], "slot_kind": s["slot_kind"],
                    "question": s["question"], "gt": s["gt"], "input_text": s["input_text"],
                }
                for name, (model, adapter, adapter_name, gen_mode) in variant_models.items():
                    record[name] = generate_answer(
                        model, tokenizer, adapter, s["vector"], s["question"],
                        act_token_id, n_tokens, device, args.max_new_tokens,
                        adapter_name=adapter_name, stop_ids=stop_ids,
                        mode=gen_mode,
                    )
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                if i < 2:
                    print(f"  [{i+1}] slot={s['slot_kind']}  Q: {s['question'][:90]}")
                    print(f"       GT: {s['gt'][:120]}")
                    for name in variant_models:
                        print(f"       {name:18s}: {record[name][:120]}")
        summary_lines.append(f"{src}: {len(samples)} → {out_path.name}")
        print(f"[ours] saved {out_path}")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(
        f"method: ours\n"
        f"variants: {', '.join(variant_models.keys())}\n"
        f"n_samples: {args.n_samples}  seed: {args.seed}\n"
        f"reps_subdir: {reps_subdir}\n\n"
        + "\n".join(summary_lines)
    )
    print(f"\n[ours] ✓ summary → {summary_path}")


if __name__ == "__main__":
    main()
