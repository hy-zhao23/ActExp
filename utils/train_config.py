"""
Training configuration schema (pydantic).

One YAML file per decoder model; tasks ("activation", "sae") are sub-keys
that share the top-level model and training settings.

Load:
    cfg = OracleTrainConfig.from_yaml("configs/qwen3_4b_v0-1.yaml")
    task_cfg = cfg.tasks["activation"]
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name:         str   = "Qwen/Qwen3-4B"
    lora_r:       int   = 32
    lora_alpha:   int   = 64
    lora_dropout: float = 0.05
    # Shard a frozen LM across all visible GPUs in the process via HF
    # `device_map="auto"` (naive pipeline parallel). Use with
    # `torchrun --nproc_per_node=1` so one process owns all local GPUs.
    model_parallel:     bool         = False
    max_memory_per_gpu: str | None   = None  # e.g. "70GiB"; None → HF default
    # FSDP (FULL_SHARD) across all ranks. Mutually exclusive with model_parallel.
    # Launch with `torchrun --nproc_per_node=N` (per-GPU-per-rank). Frozen base
    # is sharded; trainable adapter remains separate (its own DDP, see train).
    use_fsdp:           bool         = False


class AdapterConfig(BaseModel):
    type:       str   = "mlp"   # "mlp" | "cross_attn"
    n_tokens:   int               # 64 for activation
    dropout:    float = 0.0
    # MLP-specific
    n_hidden:       int        = 1    # Linear+GELU+Dropout blocks (only for type="mlp")
    bottleneck_dim: int | None = None # if set, insert Linear(hidden→bot) before output (mlp only)
    # Cross-attention-specific
    n_contexts: int   = 8         # number of context tokens (only for type="cross_attn")
    n_heads:    int   = 8
    n_layers:   int   = 1         # stacked (CrossAttn + FFN) blocks
    ffn_mult:   int   = 4         # FFN hidden = lm_dim × ffn_mult; 0 disables FFN


class DatasetConfig(BaseModel):
    source:         str | list[str] | None = None  # single, list, or None=auto-discover *.jsonl in subdir
    n_samples:      int | None = None
    seed:           int        = 42
    test_ratio:     float      = 0.1
    recon_ratio:    float      = 0.4
    max_label_len:  int | None = None                # truncate label tokens; None = no limit
    stage1_ratio:   float      = 1.0                 # fraction of train split used in stage 1; remainder held out for stage 2
    subdir:         str        = "finetune_diversified"  # data/raw/<subdir>/{source}.jsonl
    reps_subdir:    str | None = None                # data/representations/<reps_subdir>/; None → falls back to subdir


class TrainingConfig(BaseModel):
    lm_lr:                float = 5e-5
    adapter_lr:           float = 3e-4
    weight_decay:         float = 0.01
    warmup_ratio:         float = 0.05
    grad_clip:            float = 1.0
    n_epochs:             int   = 3
    eval_steps:           int   = 200
    early_stop_patience:  int   = 3       # stop after N consecutive worsening evals
    early_stop_min_delta: float = 1e-4    # min improvement required to reset patience counter


class TaskTrainingConfig(BaseModel):
    """Per-task training knobs that differ between ACT and SAE jobs."""
    batch_size:       int = 8
    grad_accum_steps: int = 1   # effective batch = batch_size × grad_accum_steps


class FinetuneConfig(BaseModel):
    output_dir:          str        = ""
    lm_lr:               float      = 5e-6
    adapter_lr:          float      = 5e-5
    weight_decay:        float      = 0.01
    warmup_ratio:        float      = 0.1
    min_lr_ratio:        float      = 0.0    # cosine decay floor as fraction of peak; 0 = decay to zero
    grad_clip:           float      = 1.0
    n_epochs:            int        = 3
    eval_steps:          int        = 100
    batch_size:          int        = 32
    grad_accum_steps:    int        = 1
    lora_target_modules: list[str] | str = Field(default_factory=lambda: ["q_proj", "v_proj"])
    reps_subdir:         str        = "finetune_diversified"      # data/representations/{reps_subdir}; switch per base model
    raw_subdir:          str        = "finetune_diversified"      # data/raw/{raw_subdir}/{source}.jsonl
    qa_subdir:           str | None = "finetune_qa_diversified"   # data/raw/{qa_subdir}/{source}.jsonl; None → falls back to f"{raw_subdir}_qa"
    train_subsample_ratio: float | None = None     # if set, per-source stratified subsample to this fraction of train rows (eval untouched)
    train_seed:          int | None = None         # 训练侧随机性 seed（LoRA init/dropout/shuffle）；None → 沿用 dataset.seed。数据子采样恒用 dataset.seed，不受此影响
    early_stop_patience:  int        = 0           # 0 = disabled; >0 = stop after N consecutive worsening evals
    early_stop_min_delta: float      = 1e-4        # min improvement required to reset patience counter


class IOConfig(BaseModel):
    output_dir: str
    resume:     bool = True


class TaskConfig(BaseModel):
    adapter:       AdapterConfig
    dataset:       DatasetConfig
    io:            IOConfig
    task_training: TaskTrainingConfig = Field(default_factory=TaskTrainingConfig)
    finetune:      FinetuneConfig     = Field(default_factory=FinetuneConfig)


class OracleTrainConfig(BaseModel):
    model:    ModelConfig    = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tasks:    dict[str, TaskConfig]   # key: "activation"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OracleTrainConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
