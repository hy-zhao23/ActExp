# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field
from typing import List


@dataclass
class lora_config:
    # Aligned to AO stage2 LoRA config (r=64, α=128, all-linear, dropout=0.1).
    # Was upstream LatentQA default r=16/α=32/dropout=0.05 — bumped so the
    # three baselines share the same parameter budget on the decoder.
    r: int = 64
    lora_alpha: int = 128
    target_modules: str = "all-linear"
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.1
    inference_mode: bool = False
