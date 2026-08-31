"""Adapters mapping a single activation vector → n_tokens soft-token embeddings.

Factory: build_adapter(type, ...) returns the requested class.
Types:
  "mlp"        — MLPAdapter (baseline, depth-configurable)
  "cross_attn" — CrossAttentionAdapter (parameter-efficient Q-Former style)
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


# ══════════════════════════════════════════════════════════════════════════
# MLP adapter (baseline)
# ══════════════════════════════════════════════════════════════════════════

class MLPAdapter(nn.Module):
    """LayerNorm → Linear(in→2*lm) → GELU → Dropout → ... → Linear(2*lm → n·lm).

    n_hidden controls depth (≥1). When bottleneck_dim is set, an extra
    Linear(hidden→bottleneck)+GELU+Dropout is inserted before the output
    projection, making output Linear(bottleneck → n·lm) — this is the
    parameter-saving path: at n_tokens=32, bottleneck=512, total ≈ 66M
    (vs 860M for the no-bottleneck n_tokens=64 config).
    """

    def __init__(
        self,
        in_dim:        int,
        lm_dim:        int,
        n_tokens:      int        = 1,
        dropout:       float      = 0.0,
        n_hidden:      int        = 1,
        bottleneck_dim: int | None = None,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.lm_dim   = lm_dim
        assert n_hidden >= 1

        hidden = lm_dim * 2
        layers: list = [
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_hidden - 1):
            layers += [
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]

        if bottleneck_dim is not None:
            layers += [
                nn.Linear(hidden, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            final_in = bottleneck_dim
        else:
            final_in = hidden

        layers.append(nn.Linear(final_in, lm_dim * n_tokens))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, in_dim) → (B, n_tokens, lm_dim)"""
        B = x.shape[0]
        return self.net(x).view(B, self.n_tokens, self.lm_dim)


# ══════════════════════════════════════════════════════════════════════════
# Cross-attention adapter (Q-Former style)
# ══════════════════════════════════════════════════════════════════════════

class _CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention: queries attend to fixed context. Residual on q."""

    def __init__(self, lm_dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert lm_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = lm_dim // n_heads
        self.ln_q   = nn.LayerNorm(lm_dim)
        self.ln_ctx = nn.LayerNorm(lm_dim)
        self.q_proj   = nn.Linear(lm_dim, lm_dim)
        self.kv_proj  = nn.Linear(lm_dim, 2 * lm_dim)
        self.out_proj = nn.Linear(lm_dim, lm_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: Tensor, ctx: Tensor) -> Tensor:
        B, N, lm = q.shape
        M = ctx.shape[1]
        qn = self.ln_q(q)
        cn = self.ln_ctx(ctx)
        Q = self.q_proj(qn)
        K, V = self.kv_proj(cn).chunk(2, dim=-1)

        def split(t: Tensor, L: int) -> Tensor:
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        Qh, Kh, Vh = split(Q, N), split(K, M), split(V, M)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn  = torch.softmax(Qh @ Kh.transpose(-1, -2) * scale, dim=-1)
        attn  = self.drop(attn)
        out   = (attn @ Vh).transpose(1, 2).contiguous().view(B, N, lm)
        return q + self.drop(self.out_proj(out))


class _FFNBlock(nn.Module):
    """Pre-norm FFN: LN → Linear(lm→lm·mult) → GELU → Linear(→lm). Residual."""

    def __init__(self, lm_dim: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln  = nn.LayerNorm(lm_dim)
        self.fc1 = nn.Linear(lm_dim, lm_dim * ffn_mult)
        self.fc2 = nn.Linear(lm_dim * ffn_mult, lm_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop(self.fc2(torch.nn.functional.gelu(self.fc1(self.ln(x)))))


class CrossAttentionAdapter(nn.Module):
    """Q-Former style adapter with configurable depth.

    Step 1 — activation → M context tokens (Linear + LN).
    Step 2 — N learnable query tokens pass through n_layers × (CrossAttn + FFN).
    Step 3 — final LayerNorm + dropout.

    n_layers=1, ffn_mult=0  (v4a legacy geometry, no FFN): ~110M
    n_layers=2, ffn_mult=4  (v4b):                         ~241M
      context_proj : 4096 × 8·2560                        = 83.9M
      per layer    : cross-attn (26.2M) + FFN (52.4M)     = 78.6M × 2 = 157.2M

    Set ffn_mult=0 to disable the FFN path (cross-attn only).
    """

    def __init__(
        self,
        in_dim:      int,
        lm_dim:      int,
        n_tokens:    int   = 1,
        dropout:     float = 0.0,
        n_contexts:  int   = 8,
        n_heads:     int   = 8,
        n_layers:    int   = 1,
        ffn_mult:    int   = 4,
    ):
        super().__init__()
        assert lm_dim % n_heads == 0, f"lm_dim {lm_dim} not divisible by n_heads {n_heads}"
        assert n_layers >= 1
        self.n_tokens   = n_tokens
        self.lm_dim     = lm_dim
        self.n_contexts = n_contexts

        self.context_proj = nn.Linear(in_dim, n_contexts * lm_dim)
        self.context_ln   = nn.LayerNorm(lm_dim)
        self.queries      = nn.Parameter(torch.randn(n_tokens, lm_dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.ModuleDict({"cross_attn": _CrossAttnBlock(lm_dim, n_heads, dropout)})
            if ffn_mult > 0:
                block["ffn"] = _FFNBlock(lm_dim, ffn_mult, dropout)
            self.layers.append(block)

        self.out_ln = nn.LayerNorm(lm_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, in_dim) → (B, n_tokens, lm_dim)"""
        B = x.shape[0]
        ctx = self.context_ln(self.context_proj(x).view(B, self.n_contexts, self.lm_dim))
        q   = self.queries.unsqueeze(0).expand(B, -1, -1)
        for block in self.layers:
            q = block["cross_attn"](q, ctx)
            if "ffn" in block:
                q = block["ffn"](q)
        return self.drop(self.out_ln(q))


# ══════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════

def build_adapter(adapter_type: str, in_dim: int, lm_dim: int, **kwargs) -> nn.Module:
    """Instantiate an adapter by type. Unknown kwargs for a given type are ignored."""
    if adapter_type == "mlp":
        allowed = {"n_tokens", "dropout", "n_hidden", "bottleneck_dim"}
        return MLPAdapter(in_dim, lm_dim, **{k: v for k, v in kwargs.items() if k in allowed})
    if adapter_type == "cross_attn":
        allowed = {"n_tokens", "dropout", "n_contexts", "n_heads", "n_layers", "ffn_mult"}
        return CrossAttentionAdapter(in_dim, lm_dim, **{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"unknown adapter_type: {adapter_type!r}")
