"""Ship: lower trained weights into the nanollama.c binary format.

The engine's loader reads a magic int (0x616B3432 'ak42'), a version, a 7-int
config, a shared_classifier flag, then every tensor back-to-back as
little-endian fp32, grouped by tensor type in exactly this order:

    token_embedding, rms_att*, wq*, wk*, wv*, wo*, rms_ffn*, w1*, w2*, w3*,
    rms_final

(*per layer). With a tied classifier there is no separate wcls — the flag
`shared_classifier = 1` tells the C engine to reuse the embedding table.

int8: the engine has no int8 *file* format — it quantizes at load time
(`nanollama -q`), linear layers only, per row, groups of up to 64 weights
sharing one fp32 scale (embedding + norms stay fp32). quality_report()
reproduces that exact rule — including the engine's (dim, kv_dim) grouping
for wk/wv — so the report is the receipt for what `-q` will cost you.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import torch

from . import registry
from .compiler import compile_spec

MAGIC = 0x616B3432  # 'ak42'
GS = 64             # engine's QGS — quantization group size


def _check_exportable(spec) -> None:
    ok, why = spec.is_exportable()
    if not ok:
        raise ValueError(f"this architecture can't be exported to nanollama.c yet: {why}")


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().to(torch.float32).contiguous().numpy().tobytes()


def export_fp32(model, spec, out_path: str, vocab_size: int | None = None) -> str:
    """Write the engine .bin. Byte-for-byte the layout nanobrain exports."""
    _check_exportable(spec)
    ms = spec.model
    sd = model.state_dict()
    n_layers = len(model.stages)
    vocab = vocab_size or spec.vocab_size
    parts = [
        struct.pack(
            "<9i", MAGIC, 1,
            ms.dim, ms.hidden, n_layers,
            ms.heads, ms.kv_heads, vocab, ms.seq_len,
        ),
        struct.pack("<i", 1 if ms.tied else 0),
        _tensor_bytes(sd["token_embedding.weight"]),
    ]
    # grouped by tensor type across layers — the engine maps weights in this order
    for sub in ("att_norm.weight",
                "att.wq.weight", "att.wk.weight", "att.wv.weight", "att.wo.weight",
                "ffn_norm.weight",
                "ffn.w1.weight", "ffn.w2.weight", "ffn.w3.weight"):
        for i in range(n_layers):
            parts.append(_tensor_bytes(sd[f"stages.{i}.{sub}"]))
    parts.append(_tensor_bytes(sd["final_norm.weight"]))

    with open(out_path, "wb") as f:
        for p in parts:
            f.write(p)
    return out_path


# ---- engine int8 emulation ---------------------------------------------------
def _quantize_rows(w: np.ndarray) -> np.ndarray:
    """Per row, groups of up to GS weights share one scale (engine's
    quantize_matrix, remainder rows included). Returns the dequantized matrix —
    the quality report only needs what the engine would *compute* with."""
    rows, n = w.shape
    out = np.empty_like(w)
    for r in range(rows):
        i = 0
        while i < n:
            g = min(GS, n - i)
            chunk = w[r, i : i + g]
            amax = float(np.max(np.abs(chunk)))
            if amax == 0.0:
                out[r, i : i + g] = 0.0
            else:
                d = 127.0 / amax
                q = np.clip(np.round(chunk * d), -127, 127).astype(np.int8)
                out[r, i : i + g] = q.astype(np.float32) / d
            i += g
    return out


def emulate_engine_int8(model, spec) -> dict:
    """A state dict as the engine would compute with `-q`: linear layers
    quantized per row, embedding and norms untouched. wk/wv are grouped as
    (dim, kv_dim) — the engine's view of the flat blob — not PyTorch's
    (kv_dim, dim)."""
    ms = spec.model
    kv_dim = ms.kv_heads * (ms.dim // ms.heads)
    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        w = v.detach().to(torch.float32).numpy()
        key = k.split(".", 2)[-1] if k.startswith("stages.") else k
        if key in ("att.wq.weight", "att.wo.weight"):
            out[k] = torch.from_numpy(_quantize_rows(w))
        elif key in ("att.wk.weight", "att.wv.weight"):
            out[k] = torch.from_numpy(_quantize_rows(w.reshape(ms.dim, kv_dim)).reshape(w.shape))
        elif key in ("ffn.w1.weight", "ffn.w3.weight"):
            out[k] = torch.from_numpy(_quantize_rows(w))       # (hidden, dim)
        elif key in ("ffn.w2.weight",):
            out[k] = torch.from_numpy(_quantize_rows(w))       # (dim, hidden)
        else:
            out[k] = v.detach().clone()
    return out


def estimate_int8_size(model, spec) -> float:
    """Bytes the engine's quantized weights would occupy (linear layers at
    1 byte + GS-scale, embedding + norms fp32)."""
    ms = spec.model
    kv_dim = ms.kv_heads * (ms.dim // ms.heads)
    linear = {
        "att.wq.weight": ms.dim * ms.dim,
        "att.wk.weight": ms.dim * kv_dim,
        "att.wv.weight": ms.dim * kv_dim,
        "att.wo.weight": ms.dim * ms.dim,
        "ffn.w1.weight": ms.hidden * ms.dim,
        "ffn.w2.weight": ms.dim * ms.hidden,
        "ffn.w3.weight": ms.hidden * ms.dim,
    }
    n_layers = len(model.stages)
    q_floats = sum(n_layers * n for n in linear.values())
    q_bytes = q_floats + (q_floats + GS - 1) // GS * 4  # int8 values + one scale per group
    fp32_floats = spec.vocab_size * ms.dim + n_layers * 2 * ms.dim + ms.dim
    return q_bytes + fp32_floats * 4


# ---- quality report ----------------------------------------------------------
@torch.no_grad()
def quality_report(model, spec, data, run_dir: str, exported: dict) -> str:
    """The receipt: val metrics + samples, fp32 vs what `-q` computes."""
    from .tokenizer import ByteTokenizer
    from .trainer import evaluate, sample_text

    lines = ["# quality report", "",
             "int8 row = the engine's load-time quantization (`nanollama -q`) reproduced",
             "weight-for-weight in PyTorch.", ""]

    def table(rows, header):
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(x) for x in r) + " |")
        lines.append("")

    device = next(model.parameters()).device
    ev = evaluate(model, data, spec, device, spec.train.eval_batches)
    is_lm = spec.model.head == "lm"
    fp32_size = os.path.getsize(exported["fp32"])
    rows = [["fp32 (shipped .bin)", f"{fp32_size / 1e6:.2f} MB"]]

    def metric_cells(e):
        if is_lm:
            return [f"{e['val_loss']:.4f}", f"{e['val_ppl']:.2f}"]
        return [f"{e.get('val_accuracy', 0) * 100:.1f}% acc", "-"]

    rows[0] += metric_cells(ev)

    int8_model = None
    try:
        q_sd = emulate_engine_int8(model, spec)
        int8_model = compile_spec(spec, model.vocab_size, getattr(data, "n_features", None))
        int8_model.load_state_dict(q_sd, strict=True)
        int8_model.to(device)
        ev8 = evaluate(int8_model, data, spec, device, spec.train.eval_batches)
        rows.append(["int8 (engine `-q`)", f"{estimate_int8_size(model, spec) / 1e6:.2f} MB*"] + metric_cells(ev8))
        lines.append("")
    except Exception as e:  # noqa: BLE001 — a failed emulation shouldn't kill the export
        lines.append(f"_(int8 emulation skipped: {type(e).__name__}: {e})_")
        lines.append("")

    table(rows, ["weights", "size", "val loss/acc", "val ppl"])
    if int8_model is not None:
        lines.append("\\* estimated in-memory size — the engine quantizes at load time; the shipped .bin stays fp32.")
        lines.append("")

    if is_lm:
        tok = ByteTokenizer()
        lines += ["## samples — prompt: `Once upon a time`", "", "**fp32**", "```",
                  sample_text(model, "Once upon a time", 160, 0.8), "```"]
        if int8_model is not None:
            lines += ["**int8**", "```",
                      sample_text(int8_model, "Once upon a time", 160, 0.8), "```"]

    report = "\n".join(lines)
    with open(os.path.join(run_dir, "export", "quality_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    return report


def export_run(project: str, run_id: str, quantize: list[str] | None = None) -> dict:
    """Export a run's best checkpoint into the engine format. Returns {files, report}."""
    run_dir, run = registry.read_run(run_id, project)
    if run.get("status") == "running":
        raise RuntimeError("that run is still training")
    sd = run["spec"]
    from .spec import NanoSpec, ModelSpec, TrainSpec, DataSpec, ExportSpec, Block

    m = dict(sd["model"])
    layers = [Block(b["kind"], b.get("params", {})) for b in (m.get("layers") or [])]
    m.pop("layers", None)
    spec = NanoSpec(
        name=sd["name"],
        tokenizer=sd.get("tokenizer", "byte"),
        bpe_model=sd.get("bpe_model"),
        model=ModelSpec(**m, layers=layers or None),
        train=TrainSpec(**sd.get("train", {})),
        data=DataSpec(**sd.get("data", {})),
        export=ExportSpec(**sd.get("export", {})),
    )

    ckpt_path = os.path.join(run_dir, "ckpt_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(run_dir, "ckpt_last.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model = compile_spec(spec, run.get("vocab_size") or spec.vocab_size, run.get("n_features"))
    model.load_state_dict(ck["model"])
    model.eval()

    out_dir = os.path.join(run_dir, "export")
    os.makedirs(out_dir, exist_ok=True)
    vocab = run.get("vocab_size") or spec.vocab_size
    exported = {"fp32": export_fp32(model, spec, os.path.join(out_dir, "model.nanollama.bin"), vocab)}

    meta = {
        "run": run["id"],
        "step": ck.get("step"),
        "config": spec.to_dict()["model"],
        "files": ["model.nanollama.bin"],
        "engine": "github.com/Agarwalrishu13/nanollama.c",
        "usage": "nanollama run -m model.nanollama.bin -i \"your prompt\"   "
                 "(add -q for the engine's int8 in-memory quantization)",
    }
    with open(os.path.join(out_dir, "export.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report = None
    if spec.model.head == "lm":
        from .data import make_data
        spec_file = os.path.join(project, "model.nanospec")
        data_path = spec.data.path
        if os.path.isabs(data_path) is False:
            data_path = os.path.normpath(os.path.join(project, data_path))
        data, _ = make_data(spec, data_path)
        report = quality_report(model, spec, data, run_dir, exported)

    return {"files": exported, "report": report, "meta": meta}
