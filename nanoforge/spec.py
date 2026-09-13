"""nanospec v0 — a model is a declarative file, not hidden code.

One file, two compilation targets:
    spec -> PyTorch      (train / evaluate, via nanoforge.compiler)
    spec -> engine .bin  (ship, via nanoforge.exporter)

The spec is the single source of truth: the studio's architecture canvas is
only a view over it, and every run snapshots the spec it was trained with.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

import yaml

# Blocks that the nanollama.c exporter can lower to the engine's format.
# Everything else trains fine but refuses to export (with a clear message).
EXPORTABLE_BLOCKS = {"attention", "swiglu_ffn", "rmsnorm"}

KNOWN_BLOCKS = EXPORTABLE_BLOCKS | {"gelu_ffn", "conv1d"}

TOKENIZERS = ("byte", "bpe")


@dataclass
class Block:
    kind: str
    params: dict = field(default_factory=dict)


@dataclass
class ModelSpec:
    vocab: int | None = None          # None = inferred from the tokenizer (byte -> 256)
    dim: int = 64
    hidden: int = 176
    heads: int = 4
    kv_heads: int | None = None       # None = same as heads
    seq_len: int = 128
    n_layers: int | None = None       # shorthand: n_layers x [attention, swiglu_ffn]
    layers: list[Block] | None = None  # explicit stack — the "own architecture" path
    head: str = "lm"                  # lm | classification
    n_classes: int | None = None      # classification only
    pool: str = "mean"                # classification pooling: mean | last
    tied: bool = True                 # tie classifier to the embedding (lm only)
    rope_base: float = 10000.0        # must match nanollama.c's rope()
    rmsnorm_eps: float = 1e-5         # must match nanollama.c's rmsnorm()
    dropout: float = 0.0


@dataclass
class TrainSpec:
    steps: int = 300
    batch: int = 32
    lr: float = 3e-3
    warmup: int = 30
    schedule: str = "cosine"          # cosine | constant
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 20
    sample_every: int = 100           # 0 = never (lm only)
    val_split: float = 0.1
    seed: int = 0


@dataclass
class DataSpec:
    path: str = "data/stories.txt"    # text (lm) or csv (classification)
    label_column: str = "label"       # csv only


@dataclass
class ExportSpec:
    formats: list[str] = field(default_factory=lambda: ["nanollama"])
    quantize: list[str] = field(default_factory=list)  # e.g. ["int8"]


@dataclass
class NanoSpec:
    name: str
    tokenizer: str = "byte"           # byte | bpe (bpe imports a nanobrain tokenizer)
    bpe_model: str | None = None      # path to a sentencepiece .model (bpe only)
    model: ModelSpec = field(default_factory=ModelSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    data: DataSpec = field(default_factory=DataSpec)
    export: ExportSpec = field(default_factory=ExportSpec)

    # ---- layer stack -----------------------------------------------------
    def resolved_layers(self) -> list[Block]:
        if self.model.layers:
            return self.model.layers
        n = self.model.n_layers or 3
        pair = (Block("attention"), Block("swiglu_ffn"))
        return [b for _ in range(n) for b in pair]

    def stack_summary(self) -> str:
        parts = []
        for b in self.resolved_layers():
            if b.kind == "attention":
                parts.append(f"attn({b.params.get('heads', self.model.heads)}h)")
            elif b.kind == "swiglu_ffn":
                parts.append(f"swiglu({b.params.get('hidden', self.model.hidden)})")
            elif b.kind == "gelu_ffn":
                parts.append(f"gelu({b.params.get('hidden', self.model.hidden)})")
            else:
                extra = ",".join(f"{k}={v}" for k, v in b.params.items())
                parts.append(f"{b.kind}({extra})")
        return " -> ".join(parts)

    def is_exportable(self) -> tuple[bool, str]:
        """Whether this exact architecture can lower to the nanollama.c format."""
        layers = self.resolved_layers()
        if self.model.head != "lm":
            return False, "only the `lm` head maps to the engine (classification ships via ONNX later)"
        if not self.model.tied:
            return False, "untied classifiers are not in the engine format yet"
        i = 0
        while i < len(layers):
            if i + 1 >= len(layers):
                return False, f"layer stack must be [attention, swiglu_ffn] pairs; got a trailing {layers[i].kind}"
            a, f = layers[i], layers[i + 1]
            if a.kind != "attention" or f.kind != "swiglu_ffn":
                return False, (
                    f"engine export needs repeating [attention, swiglu_ffn] pairs; "
                    f"found {a.kind}+{f.kind} at layer pair {i // 2}"
                )
            if a.params.get("rope", True) is False or a.params.get("causal", True) is False:
                return False, "engine attention is causal + rope"
            if a.params.get("heads", self.model.heads) != self.model.heads:
                return False, "per-layer head overrides change n_heads; the engine format has one global n_heads"
            i += 2
        return True, "ok"

    # ---- io ----------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        if d["model"].get("layers"):
            d["model"]["layers"] = [
                {"kind": b["kind"], "params": b["params"]} for b in d["model"]["layers"]
            ]
        else:
            d["model"].pop("layers", None)
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @property
    def vocab_size(self) -> int:
        if self.model.vocab:
            return self.model.vocab
        return 259 if self.tokenizer == "byte" else 0  # bpe: real vocab comes from the tokenizer file


def _block_from(d: dict) -> Block:
    if not isinstance(d, dict) or "kind" not in d:
        raise ValueError(f"each layer needs a `kind`; got: {d!r}")
    kind = d["kind"]
    if kind not in KNOWN_BLOCKS:
        raise ValueError(f"unknown block '{kind}' (known: {sorted(KNOWN_BLOCKS)})")
    params = {k: v for k, v in (d.get("params") or {}).items()}
    return Block(kind, params)


def _positive(spec: NanoSpec, section: str, key: str, value, minimum: float = 0) -> None:
    if value is None or value <= minimum:
        raise ValueError(f"model.{section}.{key if section != 'model' else key} must be > {minimum}")


def load_spec(path: str) -> NanoSpec:
    """Parse + validate a model.nanospec. Raises ValueError with a human message."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("spec file must be a YAML mapping")

    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("`name` is required")

    tok = raw.get("tokenizer", "byte")
    if tok not in TOKENIZERS:
        raise ValueError(f"tokenizer must be one of {TOKENIZERS}, got '{tok}'")
    if tok == "bpe" and not raw.get("bpe_model"):
        raise ValueError("tokenizer: bpe requires `bpe_model: path/to/tokenizer.model`")

    m = raw.get("model") or {}
    spec = NanoSpec(
        name=name,
        tokenizer=tok,
        bpe_model=raw.get("bpe_model"),
        model=ModelSpec(
            vocab=m.get("vocab"),
            dim=m.get("dim", 64),
            hidden=m.get("hidden", 176),
            heads=m.get("heads", 4),
            kv_heads=m.get("kv_heads"),
            seq_len=m.get("seq_len", 128),
            n_layers=m.get("n_layers"),
            layers=[_block_from(b) for b in m["layers"]] if m.get("layers") else None,
            head=m.get("head", "lm"),
            n_classes=m.get("n_classes"),
            pool=m.get("pool", "mean"),
            tied=m.get("tied", True),
            rope_base=m.get("rope_base", 10000.0),
            rmsnorm_eps=m.get("rmsnorm_eps", 1e-5),
            dropout=m.get("dropout", 0.0),
        ),
        train=TrainSpec(**{**TrainSpec().__dict__, **(raw.get("train") or {})}),
        data=DataSpec(**{**DataSpec().__dict__, **(raw.get("data") or {})}),
        export=ExportSpec(**{**ExportSpec().__dict__, **(raw.get("export") or {})}),
    )

    ms = spec.model
    if ms.dim <= 0 or ms.hidden <= 0 or ms.seq_len <= 0:
        raise ValueError("model.dim, model.hidden, model.seq_len must be positive")
    if ms.heads <= 0 or ms.dim % ms.heads != 0:
        raise ValueError(f"model.heads must divide model.dim ({ms.dim} % {ms.heads} != 0)")
    ms.kv_heads = ms.kv_heads or ms.heads
    if ms.heads % ms.kv_heads != 0:
        raise ValueError(f"kv_heads ({ms.kv_heads}) must divide heads ({ms.heads})")
    if ms.head == "classification":
        if not ms.n_classes or ms.n_classes < 2:
            raise ValueError("head: classification requires model.n_classes >= 2")
        if ms.pool not in ("mean", "last"):
            raise ValueError("model.pool must be 'mean' or 'last'")
    if spec.model.vocab is not None and spec.model.vocab <= 0:
        raise ValueError("model.vocab must be positive (or omit it: byte tokenizer -> 256)")
    if not spec.resolved_layers():
        raise ValueError("model needs `layers:` or `n_layers:`")
    for b in spec.resolved_layers():
        if b.kind not in KNOWN_BLOCKS:
            raise ValueError(f"unknown block '{b.kind}'")
    if ms.dropout < 0 or ms.dropout >= 1:
        raise ValueError("model.dropout must be in [0, 1)")

    t = spec.train
    if t.steps <= 0 or t.batch <= 0:
        raise ValueError("train.steps and train.batch must be positive")
    if t.lr <= 0:
        raise ValueError("train.lr must be positive")
    if t.warmup < 0 or t.warmup > t.steps:
        raise ValueError(f"train.warmup must be in [0, steps={t.steps}]")
    if not 0 < t.val_split < 1:
        raise ValueError("train.val_split must be in (0, 1)")
    if t.schedule not in ("cosine", "constant"):
        raise ValueError("train.schedule must be 'cosine' or 'constant'")

    spec.export.formats = [f for f in spec.export.formats if f]
    unknown_q = [q for q in spec.export.quantize if q not in ("int8",)]
    if unknown_q:
        raise ValueError(f"unsupported quantize formats: {unknown_q} (v0 supports int8)")

    # data path exists? resolve relative to the spec file
    data_abs = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), spec.data.path))
    if not os.path.exists(data_abs):
        raise ValueError(f"data file not found: {spec.data.path} (looked at {data_abs})")

    return spec
