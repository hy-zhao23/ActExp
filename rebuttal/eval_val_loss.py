"""
R2 cross-eval: 每个 stage-2 ckpt 在不同 val 子集上的 teacher-forcing loss。

把三个模型放到同一张考卷上，分解 "数据类型效应" vs "val 难度差异"：
    ckpt ∈ {gistcls, factonly, mix_ref(1027643)}
    val  ∈ {nonwiki(8 src), wiki(7 src), full(15 src)}

单 GPU 运行（无 DDP）；loss 口径与 oracle_finetune._eval 一致（batch-mean 的平均）。

Usage:
    python rebuttal/eval_val_loss.py --out rebuttal/crosseval_val_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.adapter import build_adapter
from utils.finetune_dataset import FinetuneDataset
from utils.train_config import OracleTrainConfig
from utils.train_utils import ACT_TOKEN, TaggedDataset, compute_loss, make_collator

NONWIKI = ["ag_news", "dair_emotion", "latentqa_control", "lmsys_user",
           "scientific", "sst2", "tweeteval_emotion", "tweeteval_sentiment"]
WIKI    = ["wikipedia_concept", "wikipedia_event", "wikipedia_generic",
           "wikipedia_organization", "wikipedia_person",
           "wikipedia_place", "wikipedia_work"]
VAL_SETS = {"nonwiki": NONWIKI, "wiki": WIKI, "full": NONWIKI + WIKI}

CKPTS = {
    # name: (ft final dir, config yaml)
    "gistcls": ("checkpoints/rebuttal/oracle_ft_qwen3_instr_self_500k_gistcls/final",
                "experiments/training/configs/rebuttal/qwen3_4b_instr_self_500k_gistcls.yaml"),
    "factonly": ("checkpoints/rebuttal/oracle_ft_qwen3_instr_self_500k_factonly/final",
                 "experiments/training/configs/rebuttal/qwen3_4b_instr_self_500k_factonly.yaml"),
    "mix_ref_1027643": ("checkpoints/oracle_ft_qwen3_4b_l27_diversified/final",
                        "checkpoints/oracle_ft_qwen3_4b_l27_diversified/config.yaml"),
}


def eval_ckpt(name: str, ckpt: Path, cfg_path: str, device, batch_size: int) -> dict:
    cfg = OracleTrainConfig.from_yaml(cfg_path)
    tc  = cfg.tasks["activation"]
    mc, ac, dc, ftc = cfg.model, tc.adapter, tc.dataset, tc.finetune

    tokenizer = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
    tokenizer.padding_side = "right"
    act_token_id = tokenizer.convert_tokens_to_ids(ACT_TOKEN)
    assert act_token_id is not None and act_token_id >= 0

    model = AutoModelForCausalLM.from_pretrained(
        mc.name, torch_dtype=torch.bfloat16, device_map={"": device})
    old_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        w = model.get_input_embeddings().weight
        w[old_vocab:] = w[:old_vocab].mean(dim=0, keepdim=True)
    model = PeftModel.from_pretrained(model, ckpt / "lora",
                                      is_trainable=False, autocast_adapter_dtype=True)
    model.eval()

    results = {}
    adapter = None
    for val_name, sources in VAL_SETS.items():
        base_ds = FinetuneDataset(
            split="val", subdir=ftc.reps_subdir, raw_subdir=ftc.raw_subdir,
            qa_subdir=ftc.qa_subdir, sources_filter=sources, sample_seed=dc.seed)
        ds = TaggedDataset(base_ds, task="qa")   # 与 oracle_finetune 训练时一致
        if adapter is None:
            adapter = build_adapter(
                ac.type, base_ds.hidden_dim, model.config.hidden_size,
                n_tokens=ac.n_tokens, dropout=ac.dropout,
                n_hidden=ac.n_hidden, bottleneck_dim=ac.bottleneck_dim,
                n_contexts=ac.n_contexts, n_heads=ac.n_heads,
                n_layers=ac.n_layers, ffn_mult=ac.ffn_mult,
                expansion=getattr(ac, "expansion", 0.5),
                res_gain=getattr(ac, "res_gain", 1.0),
            ).to(device).to(torch.bfloat16)
            adp_path = ckpt / "adapter.pt"
            if not adp_path.exists():
                adp_path = ckpt / "adapter" / "adapter.pt"   # 旧格式 (1027643)
            adapter.load_state_dict(torch.load(adp_path, map_location=device))
            adapter.eval()
        collate = make_collator(tokenizer, act_token_id, ac.n_tokens, dc.max_label_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=4, pin_memory=True)
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                total += compute_loss(model, adapter, batch, device).item()
                n += 1
        loss = total / n
        results[val_name] = {"loss": round(loss, 4), "ppl": round(float(torch.exp(torch.tensor(loss))), 3),
                             "n_samples": len(ds)}
        print(f"[{name}] val={val_name:8s} n={len(ds):6d} loss={loss:.4f} ppl={results[val_name]['ppl']}",
              flush=True)

    del model, adapter
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rebuttal/crosseval_val_results.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ckpt", action="append", default=None, metavar="NAME:CKPT_DIR:CONFIG",
                    help="覆盖内置 CKPTS，可重复。例：mix_s43:checkpoints/.../final:experiments/.../x.yaml")
    args = ap.parse_args()

    ckpts = CKPTS
    if args.ckpt:
        ckpts = {}
        for spec in args.ckpt:
            name, ckpt_dir, cfg_path = spec.split(":", 2)
            ckpts[name] = (ckpt_dir, cfg_path)

    device = torch.device("cuda:0")
    all_results = {}
    for name, (ckpt, cfg) in ckpts.items():
        print(f"===== {name}  ckpt={ckpt} =====", flush=True)
        all_results[name] = eval_ckpt(name, Path(ckpt), cfg, device, args.batch_size)

    Path(args.out).write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
