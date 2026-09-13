"""The exporter: header bytes, tensor order, and the engine's -q emulation."""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch  # noqa: E402

from nanoforge.compiler import compile_spec  # noqa: E402
from nanoforge.exporter import (  # noqa: E402
    MAGIC, GS, export_fp32, emulate_engine_int8, estimate_int8_size,
)
from nanoforge.spec import load_spec  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_LM = os.path.join(ROOT, "templates", "tiny-char-lm", "model.nanospec")


def _tiny_model(tmpdir):
    spec = load_spec(TEMPLATE_LM)
    spec.model.dim, spec.model.hidden, spec.model.n_layers = 32, 88, 2
    spec.model.seq_len = 64
    torch.manual_seed(0)
    model = compile_spec(spec, spec.vocab_size)
    return spec, model


def test_header_and_size(tmpdir=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        spec, model = _tiny_model(td)
        out = export_fp32(model, spec, os.path.join(td, "m.bin"))
        blob = open(out, "rb").read()
        magic, version, dim, hidden, n_layers, heads, kv_heads, vocab, seq_len = struct.unpack(
            "<9i", blob[:36]
        )
        (shared,) = struct.unpack("<i", blob[36:40])
        assert magic == MAGIC and version == 1
        assert (dim, hidden, n_layers, heads, kv_heads) == (32, 88, 2, 4, 4)
        assert vocab == 259 and shared == 1
        n_floats = (len(blob) - 40) // 4
        # emb + per-layer [att_norm, wq, wk, wv, wo, ffn_norm, w1, w2, w3] + final_norm
        per_layer = 32 + 4 * 32 * 32 + 32 + 3 * 88 * 32
        expected = 259 * 32 + 2 * per_layer + 32
        assert n_floats == expected, f"{n_floats} != {expected}"


def test_int8_emulation_matches_engine_rules():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        spec, model = _tiny_model(td)
        q = emulate_engine_int8(model, spec)
        sd = model.state_dict()
        # embedding + norms stay fp32
        assert torch.equal(q["token_embedding.weight"], sd["token_embedding.weight"])
        assert torch.equal(q["final_norm.weight"], sd["final_norm.weight"])
        # linear layers get grouped-quantized: changed but numerically close
        diff = (q["stages.0.att.wq.weight"] - sd["stages.0.att.wq.weight"]).abs().max().item()
        assert 0 < diff < 0.5
        # size estimate is smaller than fp32
        assert estimate_int8_size(model, spec) < sum(v.numel() for v in sd.values()) * 4


def test_export_rejects_custom_stacks():
    import copy
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        spec, model = _tiny_model(td)
        spec.model.layers = None
        spec = copy.copy(spec)
        spec.model = copy.copy(spec.model)
        from nanoforge.spec import Block
        spec.model.layers = [Block("attention"), Block("gelu_ffn")]
        try:
            export_fp32(model, spec, os.path.join(td, "no.bin"))
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "pairs" in str(e)
