"""nanospec v0 validation — the contract everything else compiles from."""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoforge.spec import load_spec  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_LM = os.path.join(ROOT, "templates", "tiny-char-lm", "model.nanospec")
TEMPLATE_CLF = os.path.join(ROOT, "templates", "tiny-classifier", "model.nanospec")


def _write(tmpdir: str, doc: dict) -> str:
    path = os.path.join(tmpdir, "model.nanospec")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f)
    return path


def test_valid_template_loads():
    spec = load_spec(TEMPLATE_LM)
    assert spec.name == "tiny-char-lm"
    assert spec.vocab_size == 259  # engine convention: 3 specials + 256 bytes
    assert spec.model.heads == 4
    assert len(spec.resolved_layers()) == 6  # 3 canonical pairs


def test_shorthand_expands_to_pairs():
    spec = load_spec(TEMPLATE_LM)
    kinds = [b.kind for b in spec.resolved_layers()]
    assert kinds == ["attention", "swiglu_ffn"] * 3


def test_classification_template():
    spec = load_spec(TEMPLATE_CLF)
    assert spec.model.head == "classification"
    assert spec.model.n_classes == 3
    ok, _ = spec.is_exportable()
    assert not ok  # classifiers don't map to the engine yet — and that's honest


def test_rejects_undivisible_heads(tmpdir=""):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        doc = yaml.safe_load(open(TEMPLATE_LM))
        doc["model"]["dim"], doc["model"]["heads"] = 65, 4
        try:
            load_spec(_write(td, doc))
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "divide" in str(e)


def test_rejects_unknown_block():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        doc = yaml.safe_load(open(TEMPLATE_LM))
        doc["model"] = {"layers": [{"kind": "quantum_fusion"}]}
        doc["data"] = {"path": os.path.relpath(
            os.path.join(ROOT, "templates", "tiny-char-lm", "data", "stories.txt"), td)}
        try:
            load_spec(_write(td, doc))
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "quantum_fusion" in str(e)


def test_rejects_missing_data():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        doc = yaml.safe_load(open(TEMPLATE_LM))
        doc["data"] = {"path": "data/does-not-exist.txt"}
        try:
            load_spec(_write(td, doc))
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "not found" in str(e)


def test_rejects_bad_warmup():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        doc = yaml.safe_load(open(TEMPLATE_LM))
        doc["train"]["warmup"] = 10_000
        try:
            load_spec(_write(td, doc))
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "warmup" in str(e)


def test_exportable_flag_matches_pairs():
    import tempfile
    spec = load_spec(TEMPLATE_LM)
    ok, why = spec.is_exportable()
    assert ok, why

    with tempfile.TemporaryDirectory() as td:
        doc = yaml.safe_load(open(TEMPLATE_LM))
        doc["model"] = {"layers": [
            {"kind": "attention"},
            {"kind": "gelu_ffn"},  # breaks the canonical pairing
        ]}
        doc["data"] = {"path": os.path.relpath(
            os.path.join(ROOT, "templates", "tiny-char-lm", "data", "stories.txt"), td)}
        spec2 = load_spec(_write(td, doc))
        ok, why = spec2.is_exportable()
        assert not ok and "pairs" in why
