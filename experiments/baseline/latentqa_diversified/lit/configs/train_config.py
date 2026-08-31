from dataclasses import dataclass


@dataclass
class train_config:
    # ── Decoder model (target activation comes from disk cache, see below) ─
    target_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    load_model_checkpoint: str = ""

    # ── Cached read activation (v2 pipeline) ─────────────────────────────
    # data/representations/<cache_subdir>/{source}_rank{r}_of_{W}.npy
    #   Qwen3-4B-Instruct-2507  ↔ finetune_diversified_qwen3_l27  (hidden=2560, layer=27)
    #   Llama-3.1-8B-Instruct   ↔ finetune_diversified            (hidden=4096, layer=27)
    cache_subdir: str = "finetune_diversified_qwen3_l27"
    raw_subdir:   str = "finetune_diversified"
    qa_subdir:    str = "finetune_qa_diversified"
    train_subsample_ratio: float = 0.0   # 0/negative = no subsample (use all train QA)

    # ── Write-side / mask behavior ───────────────────────────────────────
    modify_chat_template: bool = True

    # ── Eval / logging ───────────────────────────────────────────────────
    eval_ppl:           bool = True
    eval_every_n_steps: int  = 1_000      # opt-step cadence
    output_dir:         str  = "out/runs"
    save_model:         bool = True
    save_every_n_steps: int  = 2_000
    run_name:           str  = ""

    # ── Early stop on eval PPL ───────────────────────────────────────────
    early_stop_enabled:    bool  = True
    early_stop_patience:   int   = 6        # == AO stage2
    early_stop_min_delta:  float = 1.0e-4

    # ── Write-side patching ──────────────────────────────────────────────
    # Decoder layer that receives the cached activation via forward hook.
    # 0 keeps the patchscope-style early-injection that upstream LatentQA used.
    layer_to_write: int = 0

    # ── Training ─────────────────────────────────────────────────────────
    # Effective batch per opt-step = batch_size_training × gas × world_size.
    # Aligned with ours v1-2 / AO stage-2 (eff=1536), NOT upstream LatentQA
    # paper (eff=128). Rationale: with the cached-activation pipeline the
    # GPU memory profile == ours v1-2 (single 4B/8B decoder + LoRA + short
    # seq), so ours' batch settings transfer directly. Holding eff_batch
    # equal across the three baselines is the only way to make their data-
    # exposure and lr-schedule comparable; locking the upstream recipe at
    # eff=128 would make ours/AO see 12× more samples per step than LatentQA.
    #
    #     Qwen3-4B  : bs=128 / gas=3 / 4 GPU → eff=1536  (= ours v1-2 4B)
    #     Llama-8B  : bs=64  / gas=6 / 4 GPU → eff=1536  (= ours v1-2 8B)
    batch_size_training:           int   = 128
    gradient_accumulation_steps:   int   = 3
    gradient_clipping:             bool  = False
    gradient_clipping_threshold:   float = 1.0
    num_epochs:                    int   = 5
    num_workers_dataloader:        int   = 2
    lr:                            float = 5.0e-5    # == AO / ours v1-2 lm_lr
    ema_decay:                     float = 1.0
    warmup_steps:                  int   = 0
    weight_decay:                  float = 0.01
    gamma:                         float = 0.85
    seed:                          int   = 42

    # ── PEFT / parallelism ───────────────────────────────────────────────
    peft_method: str  = "lora"
    use_peft:    bool = True
    use_fsdp:    bool = False
