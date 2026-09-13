"""The compiler: spec -> nn.Module with nanobrain-compatible state dicts."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from nanoforge.compiler import compile_spec, estimate_params  # noqa: E402
from nanoforge.spec import load_spec  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_LM = os.path.join(ROOT, "templates", "tiny-char-lm", "model.nanospec")
TEMPLATE_CLF = os.path.join(ROOT, "templates", "tiny-classifier", "model.nanospec")


def test_canonical_state_dict_matches_nanobrain_layout():
    spec = load_spec(TEMPLATE_LM)
    model = compile_spec(spec, spec.vocab_size)
    keys = set(model.state_dict().keys())
    for expected in (
        "token_embedding.weight",
        "stages.0.att_norm.weight",
        "stages.0.att.wq.weight",
        "stages.0.att.wk.weight",
        "stages.0.att.wv.weight",
        "stages.0.att.wo.weight",
        "stages.0.ffn_norm.weight",
        "stages.0.ffn.w1.weight",
        "stages.0.ffn.w2.weight",
        "stages.0.ffn.w3.weight",
        "final_norm.weight",
    ):
        assert expected in keys, f"missing {expected}"
    # tied classifier: no separate head
    assert not any("lm_head" in k for k in keys)


def test_lm_forward_shapes():
    spec = load_spec(TEMPLATE_LM)
    model = compile_spec(spec, spec.vocab_size)
    x = torch.randint(0, 259, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 259)


def test_classification_forward():
    spec = load_spec(TEMPLATE_CLF)
    model = compile_spec(spec, 256, n_features=4)
    x = torch.randn(8, 4)
    out = model(x)
    assert out.shape == (8, 3)


def test_custom_stack_compiles():
    spec = load_spec(TEMPLATE_LM)
    spec.model.layers = None
    import copy
    spec = copy.copy(spec)
    spec.model = copy.copy(spec.model)
    from nanoforge.spec import Block
    spec.model.layers = [Block("attention"), Block("swiglu_ffn"),
                         Block("gelu_ffn"), Block("conv1d"), Block("rmsnorm")]
    spec.model.n_layers = None
    model = compile_spec(spec, spec.vocab_size)
    x = torch.randint(0, 259, (2, 12))
    assert model(x).shape == (2, 12, 259)


def test_param_estimate_is_close():
    spec = load_spec(TEMPLATE_LM)
    model = compile_spec(spec, spec.vocab_size)
    actual = sum(p.numel() for p in model.parameters())
    est = estimate_params(spec, spec.vocab_size)
    assert abs(actual - est) / actual < 0.05, f"estimate {est} vs actual {actual}"


def test_gqa_compiles():
    spec = load_spec(TEMPLATE_LM)
    spec.model.kv_heads = 2
    model = compile_spec(spec, spec.vocab_size)
    x = torch.randint(0, 259, (1, 8))
    assert model(x).shape == (1, 8, 259)
