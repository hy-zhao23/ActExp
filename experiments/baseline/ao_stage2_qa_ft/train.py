"""
AO method baseline on our Stage 2 QA data (Qwen3-4B self-interpretation).

Imports nl_probes.* from the baseline AO repo as a module; AO code is NOT
modified — runtime patches live in ao_patches.py and are applied in-memory.

Donor & Oracle  : Qwen/Qwen3-4B (self-interp; 36 layers, 75% → layer 27)
Data            : data/raw/finetune + data/raw/finetune_qa (16 sources, 3.24M QA samples)
Reps            : data/representations/finetune_qwen3_l27 (Qwen3-4B layer 27 last-token)
Method          : AO steering-vector hook at oracle layer 1 (AO default)
LoRA            : r=64, α=128, dropout=0.1, target="all-linear", lr=5e-5
Batch           : 128 per-rank × grad_accum=1 × 4 ranks → effective 512
Epochs          : 5 (early-stopped via held-out LM loss, patience=6, min_delta=1e-4)
Eval/Save       : every 500 opt steps; latest/ always, best/ when eval_loss improves

Launch (from project root):
    torchrun --nproc_per_node=4 experiments/baseline/ao_stage2_qa_ft/train.py
"""

import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist

PROJ = Path(__file__).resolve().parents[3]
AO_ROOT = PROJ / "experiments" / "baseline" / "activation_oracles"
sys.path.insert(0, str(AO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for ao_patches

# AO's classification_dataset_manager does `os.listdir("datasets/...")` at module
# import time (relative to CWD). Chdir into AO_ROOT so that side-effect resolves.
# FinetuneDataset / save_dir use absolute paths so CWD change is safe.
os.chdir(AO_ROOT)

import nl_probes.sft as _ao_sft
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.common import load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint

# Our project
from utils.finetune_dataset import FinetuneDataset

import ao_patches


# AO's oom_preflight iterates the whole pre-shard training_data to find the
# longest prompt. With 3.24M samples that is a 30-minute tokenization scan on
# every rank. We skip it; Qwen3-4B bf16 + LoRA + grad_ckpt + per-rank 128 fits.
_ao_sft.oom_preflight_check = lambda *a, **kw: None


REPS_SUBDIR = "finetune_qwen3_l27"
DONOR_LAYER = 27
MODEL_NAME = "Qwen/Qwen3-4B"

# Env var overrides (set in sbatch to switch target model / layer):
#   AO_STAGE2_MODEL_NAME   - HF model id   (default Qwen/Qwen3-4B)
#   AO_STAGE2_DONOR_LAYER  - hook layer    (default 27)
MODEL_NAME  = os.environ.get("AO_STAGE2_MODEL_NAME", MODEL_NAME)
DONOR_LAYER = int(os.environ.get("AO_STAGE2_DONOR_LAYER", DONOR_LAYER))


def _rec_to_tdp(rec: dict, tokenizer, layer: int) -> TrainingDataPoint:
    vec = rec["vector"]                                   # (hidden_dim,) bf16
    acts_BD = vec.unsqueeze(0).cpu().clone().detach()     # (1, D)
    return create_training_datapoint(
        datapoint_type="oracle_stage2",
        prompt=rec["question"],
        target_response=rec["label"],
        layer=layer,
        num_positions=1,
        tokenizer=tokenizer,
        acts_BD=acts_BD,
        feature_idx=-1,
    )


class LazyStage2Data:
    """List-like view over FinetuneDataset — TrainingDataPoint built on access.

    AO's train_model uses slicing (``td[rank::world_size]``, ``td[:n]``), len,
    and iteration. Materialising the full 3.24M TrainingDataPoint list upfront
    costs ~30 GB RAM; this wrapper keeps it memory-light.
    """

    def __init__(self, fd: FinetuneDataset, tokenizer, layer: int, indices=None):
        self.fd = fd
        self.tokenizer = tokenizer
        self.layer = layer
        self.indices = indices if indices is not None else list(range(len(fd)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return LazyStage2Data(self.fd, self.tokenizer, self.layer, self.indices[key])
        if isinstance(key, int):
            return _rec_to_tdp(self.fd[self.indices[key]], self.tokenizer, self.layer)
        raise TypeError(f"Unsupported key type: {type(key)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Silence stdout on non-rank-0: AO's load_tokenizer/load_model and PEFT's
    # print_trainable_parameters all call bare print() unconditionally. We keep
    # stderr (HF warnings, tracebacks, tqdm) so real errors still surface.
    if rank != 0:
        import builtins
        builtins.print = lambda *a, **k: None

    dtype = torch.bfloat16

    # Env-var overrides (used by debug/40G submit; unset → production defaults).
    _i = lambda k, d: int(os.environ[k]) if k in os.environ else d
    _s = lambda k, d: os.environ[k] if k in os.environ else d
    _save_dir = _s("AO_STAGE2_SAVE_DIR",
                   str(PROJ / "checkpoints" / "ao_stage2_qa_ft_Qwen3-4B"))

    cfg = SelfInterpTrainingConfig(
        model_name                  = MODEL_NAME,
        hook_onto_layer             = 1,                           # AO default
        act_layers                  = [DONOR_LAYER],
        layer_percents              = [75],
        use_lora                    = True,
        lora_r                      = 64,
        lora_alpha                  = 128,
        lora_dropout                = 0.1,
        lora_target_modules         = "all-linear",
        num_epochs                  = _i("AO_STAGE2_EPOCHS", 10),  # early-stopped
        lr                          = float(os.environ.get("AO_STAGE2_LR", 5.0e-5)),  # match oracle ft v0-4_ep8_2M; override via env
        train_batch_size            = _i("AO_STAGE2_TRAIN_BS", 128),
        gradient_accumulation_steps = _i("AO_STAGE2_GRAD_ACCUM", 2),
        eval_batch_size             = _i("AO_STAGE2_EVAL_BS", 128),
        eval_steps                  = _i("AO_STAGE2_EVAL_STEPS", 500),
        save_steps                  = _i("AO_STAGE2_SAVE_STEPS", 500),
        max_grad_norm               = 1.0,
        gradient_checkpointing      = True,
        save_dir                    = _save_dir,
        seed                        = 42,
    )

    # Auto-resume: prefer latest/ (written every save_steps).
    # AO_STAGE2_DISABLE_RESUME=1 → force fresh start (debug runs).
    latest = Path(cfg.save_dir) / "latest"
    if os.environ.get("AO_STAGE2_DISABLE_RESUME") != "1" and (latest / "train_state.pt").exists():
        cfg.load_lora_path = str(latest)
        ao_patches.load_early_stop_state(latest)
        if rank == 0:
            print(f"[resume] detected ckpt {cfg.load_lora_path}  "
                  f"best_eval_loss={ao_patches._STATE['best_eval_loss']:.4f}  "
                  f"bad_count={ao_patches._STATE['bad_count']}")

    cfg.dataset_configs = []

    tokenizer = load_tokenizer(MODEL_NAME)
    ao_patches.patch_tokenizer_chat_template(tokenizer)

    _f = lambda k, d: float(os.environ[k]) if k in os.environ else d
    _subsample = _f("AO_STAGE2_SUBSAMPLE_RATIO", None)
    # Per-job dataset layout overrides — diverse / new finetune corpora live
    # under different subdirs than the original "finetune" / "finetune_qa".
    _reps_subdir = _s("AO_STAGE2_REPS_SUBDIR", REPS_SUBDIR)
    _raw_subdir  = _s("AO_STAGE2_RAW_SUBDIR",  "finetune")
    _qa_subdir   = os.environ.get("AO_STAGE2_QA_SUBDIR")  # None → falls back to f"{raw_subdir}_qa"
    train_fd = FinetuneDataset(split="train", subdir=_reps_subdir,
                               raw_subdir=_raw_subdir, qa_subdir=_qa_subdir,
                               train_subsample_ratio=_subsample)
    eval_fd  = FinetuneDataset(split="val",   subdir=_reps_subdir,
                               raw_subdir=_raw_subdir, qa_subdir=_qa_subdir)

    if rank == 0:
        print(f"[data] train acts={sum(train_fd.per_source_counts.values()):,}  "
              f"train samples={len(train_fd):,}  eval samples={len(eval_fd):,}")
        print(f"[data] sources: {list(train_fd.per_source_counts.keys())}")

    train_data = LazyStage2Data(train_fd, tokenizer, DONOR_LAYER)
    eval_data  = LazyStage2Data(eval_fd,  tokenizer, DONOR_LAYER)

    # Global shuffle (deterministic) before AO's per-rank strided shard
    random.seed(cfg.seed)
    random.shuffle(train_data.indices)

    # Install runtime patches into AO's sft module (step log, eval, save, break).
    # Eval iterates the full FinetuneDataset eval split (deterministic hold-out).
    ao_patches.apply_patches(
        _ao_sft,
        eval_data       = eval_data,
        tokenizer       = tokenizer,
        patience        = 6,
        min_delta       = 1e-4,
        eval_batch_size = _i("AO_STAGE2_EVAL_BS", 128),
    )

    # eval_datasets is kept empty — AO's cls-scorer path is replaced by the
    # held-out LM loss patch, which reads eval_data from ao_patches._EVAL_CFG.
    _ao_sft.train_model(
        cfg=cfg,
        training_data=train_data,
        eval_datasets={},
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        model_kwargs={},
        verbose=(rank == 0),
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
