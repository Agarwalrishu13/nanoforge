"""Spec -> PyTorch: the first of nanospec's two compilation targets.

compile(spec) returns an nn.Module whose state-dict names match nanobrain's
whenever the layer stack is canonical ([attention, swiglu_ffn]*n) — which is
exactly the condition exporter.py needs to lower the weights to nanollama.c.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import PairedBlock, Attention, SwiGLU, GELUMLP, CausalConv1d, RMSNorm
from .spec import NanoSpec


class ForgeModel(nn.Module):
    """Assembled architecture. LM head is tied by default (the classifier IS
    the embedding table); classification heads pool over the sequence."""

    def __init__(self, spec: NanoSpec, vocab_size: int, n_features: int | None = None):
        super().__init__()
        self.spec = spec
        ms = spec.model
        self.vocab_size = vocab_size
        self.is_lm = ms.head == "lm"

        if not self.is_lm:
            if n_features is None:
                raise ValueError("classification models need n_features (from the csv)")
            self.input_proj = nn.Linear(n_features, ms.dim)

        self.token_embedding = nn.Embedding(vocab_size, ms.dim) if self.is_lm else None
        self.stages = self._build_stages()
        self.final_norm = RMSNorm(ms.dim, ms.rmsnorm_eps)
        if self.is_lm:
            self.lm_head = None if ms.tied else nn.Linear(ms.dim, vocab_size, bias=False)
        else:
            self.head = nn.Linear(ms.dim, ms.n_classes)

        self.apply(self._init_weights)

    # ---- assembly ----------------------------------------------------------
    def _build_stages(self) -> nn.ModuleList:
        ms = self.spec.model
        layers = self.spec.resolved_layers()
        stages: list[nn.Module] = []
        i = 0
        while i < len(layers):
            if (
                i + 1 < len(layers)
                and layers[i].kind == "attention"
                and layers[i + 1].kind == "swiglu_ffn"
            ):
                a, f = layers[i], layers[i + 1]
                stages.append(
                    PairedBlock(
                        ms.dim,
                        a.params.get("heads", ms.heads),
                        ms.kv_heads,
                        f.params.get("hidden", ms.hidden),
                        ms.rmsnorm_eps,
                        ms.dropout,
                    )
                )
                i += 2
            else:
                b = layers[i]
                if b.kind == "attention":
                    stages.append(_SoloAttention(ms, b))
                elif b.kind == "swiglu_ffn":
                    stages.append(_SoloFFN(ms, b))
                elif b.kind == "gelu_ffn":
                    stages.append(GELUMLP(ms.dim, b.params.get("hidden", ms.hidden), ms.dropout))
                elif b.kind == "conv1d":
                    stages.append(CausalConv1d(ms.dim, b.params.get("kernel", 3), ms.dropout))
                elif b.kind == "rmsnorm":
                    stages.append(RMSNorm(ms.dim, ms.rmsnorm_eps))
                else:
                    raise ValueError(f"cannot compile block '{b.kind}'")
                i += 1
        return nn.ModuleList(stages)

    # ---- init ----------------------------------------------------------------
    def _init_weights(self, m: nn.Module) -> None:
        """GPT-2 init; residual-output projections scaled down by depth. Skip
        this and the loss starts at ~ln(vocab)*40 and NaNs within 20 steps."""
        n_layers = max(1, len(self.stages))
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "NL_RESIDUAL", False):
                std = 0.02 / math.sqrt(2 * n_layers)
            m.weight.data.normal_(0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.02)

    # ---- forward ---------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_lm:
            b, s = x.shape
            pos = torch.arange(s, device=x.device)
            h = self.token_embedding(x)
            for stage in self.stages:
                h = stage(h, pos, self.spec.model.rope_base)
        else:
            h = self.input_proj(x).unsqueeze(1)      # (b, 1, d) — features as a length-1 sequence
            pos = torch.zeros(1, device=x.device, dtype=torch.long)
            for stage in self.stages:
                h = stage(h, pos, self.spec.model.rope_base)
        h = self.final_norm(h)
        if self.is_lm:
            if self.lm_head is None:
                return F.linear(h, self.token_embedding.weight)   # tied classifier
            return self.lm_head(h)
        pooled = h.mean(dim=1) if self.spec.model.pool == "mean" else h[:, -1]
        return self.head(pooled)

    # ---- sampling ----------------------------------------------------------------
    @torch.no_grad()
    def generate(self, tokens: list[int], max_new: int, temperature: float = 0.8,
                 top_k: int = 0) -> list[int]:
        if not self.is_lm:
            raise ValueError("generate() is only for language models")
        self.eval()
        out = list(tokens)
        device = next(self.parameters()).device
        for _ in range(max_new):
            window = out[-self.spec.model.seq_len:]
            logits = self.forward(torch.tensor([window], device=device))[0, -1]
            logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits[logits < kth] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            out.append(int(nxt))
        return out


class _SoloAttention(nn.Module):
    """A standalone attention block outside a canonical pair (custom stacks)."""

    def __init__(self, ms, b):
        super().__init__()
        self.norm = RMSNorm(ms.dim, ms.rmsnorm_eps)
        self.att = Attention(ms.dim, b.params.get("heads", ms.heads), ms.kv_heads, ms.dropout)

    def forward(self, x, pos, rope_base):
        return x + self.att(self.norm(x), pos, rope_base)


class _SoloFFN(nn.Module):
    """A standalone SwiGLU block outside a canonical pair (custom stacks)."""

    def __init__(self, ms, b):
        super().__init__()
        self.norm = RMSNorm(ms.dim, ms.rmsnorm_eps)
        self.ffn = SwiGLU(ms.dim, b.params.get("hidden", ms.hidden))

    def forward(self, x, *_) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


def compile_spec(spec: NanoSpec, vocab_size: int, n_features: int | None = None) -> ForgeModel:
    return ForgeModel(spec, vocab_size, n_features)


def estimate_params(spec: NanoSpec, vocab_size: int, n_features: int | None = None) -> int:
    """Parameter count without building the model — powers the UI + agent."""
    ms = spec.model
    d, h = ms.dim, ms.hidden
    total = vocab_size * d + d  # embedding + final norm
    if ms.head == "lm" and not ms.tied:
        total += vocab_size * d
    if ms.head == "classification":
        total += (n_features or 0) * d + d * ms.n_classes
    for b in spec.resolved_layers():
        if b.kind == "attention":
            hs = d // b.params.get("heads", ms.heads)
            total += d * hs * b.params.get("heads", ms.heads) * 2 + d * hs * ms.kv_heads * 2 + d
        elif b.kind == "swiglu_ffn":
            hd = b.params.get("hidden", ms.hidden)
            total += 3 * d * hd + d
        elif b.kind == "gelu_ffn":
            hd = b.params.get("hidden", ms.hidden)
            total += d * hd + hd + hd * d + d
        elif b.kind == "conv1d":
            total += d * b.params.get("kernel", 3) + d * d + d
        elif b.kind == "rmsnorm":
            total += d
    return total
