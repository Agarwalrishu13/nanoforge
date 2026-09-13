"""The prebuilt block library — every block is spelled out, no nn.Transformer.

Two families share this file:
  exportable  — RMSNorm / Attention / SwiGLU, mathematically identical to
                nanollama.c's C code (same rope pairing, same eps, same
                SwiGLU), so weights can be lowered to the engine 1:1.
  studio-only — GELU-MLP, causal conv1d: great for custom architectures,
                refuse politely at export time.

Naming convention: a canonical [attention, swiglu_ffn] pair compiles to a
PairedBlock whose state-dict keys are exactly nanobrain's
(blocks.{i}.att_norm.weight, blocks.{i}.att.wq.weight, ...) — that's what
makes the exporter a straight tensor dump.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, *_ ) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def rope_rotate(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """Adjacent-pair rotation (2i, 2i+1) — nanollama.c's convention, NOT the
    half-split one HF uses, so exported weights run bit-compatibly in C."""
    batch, seq, n_heads, head_size = x.shape
    pair = torch.arange(0, head_size, 2, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (pair / head_size))
    angles = pos.float()[:, None] * inv_freq[None, :]
    cos, sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    even = x_even * cos - x_odd * sin
    odd = x_even * sin + x_odd * cos
    return torch.stack((even, odd), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_size = heads, kv_heads, dim // heads
        self.wq = nn.Linear(dim, heads * self.head_size, bias=False)
        self.wk = nn.Linear(dim, kv_heads * self.head_size, bias=False)
        self.wv = nn.Linear(dim, kv_heads * self.head_size, bias=False)
        self.wo = nn.Linear(heads * self.head_size, dim, bias=False)
        self.dropout = dropout
        self.wo.NL_RESIDUAL = True

    def forward(self, x: torch.Tensor, pos: torch.Tensor, rope_base: float) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, self.n_heads, self.head_size)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.head_size)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.head_size)
        q, k = rope_rotate(q, pos, rope_base), rope_rotate(k, pos, rope_base)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.n_kv_heads == self.n_heads:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            try:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            except TypeError:
                rep = self.n_heads // self.n_kv_heads
                out = F.scaled_dot_product_attention(
                    q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1),
                    is_causal=True,
                )
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.wo(out)
        return F.dropout(out, self.dropout, self.training)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2.NL_RESIDUAL = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GELUMLP(nn.Module):
    """Classic MLP — studio-only (not in the engine's weight format)."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(dim, hidden)
        self.proj = nn.Linear(hidden, dim)
        self.proj.NL_RESIDUAL = True
        self.dropout = dropout

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        h = F.gelu(self.fc(x))
        return x + F.dropout(self.proj(h), self.dropout, self.training)


class CausalConv1d(nn.Module):
    """Depthwise causal temporal conv + pointwise mix — studio-only."""

    def __init__(self, dim: int, kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        self.kernel = kernel
        self.depthwise = nn.Conv1d(dim, dim, kernel, groups=dim)
        self.pointwise = nn.Linear(dim, dim)
        self.pointwise.NL_RESIDUAL = True
        self.norm = RMSNorm(dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        xc = x.transpose(1, 2)                       # (b, d, s)
        xc = F.pad(xc, (self.kernel - 1, 0))         # causal
        h = self.depthwise(xc).transpose(1, 2)
        h = F.gelu(self.pointwise(h))
        return x + F.dropout(h, self.dropout, self.training)


class PairedBlock(nn.Module):
    """The canonical transformer block: attention sub-layer + SwiGLU sub-layer,
    each behind an RMSNorm with a residual. State-dict-compatible with nanobrain."""

    def __init__(self, dim: int, heads: int, kv_heads: int, hidden: int,
                 eps: float, dropout: float = 0.0):
        super().__init__()
        self.att_norm = RMSNorm(dim, eps)
        self.att = Attention(dim, heads, kv_heads, dropout)
        self.ffn_norm = RMSNorm(dim, eps)
        self.ffn = SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, rope_base: float) -> torch.Tensor:
        x = x + self.att(self.att_norm(x), pos, rope_base)
        x = x + self.ffn(self.ffn_norm(x))
        return x
