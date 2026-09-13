# nanospec v0 — reference

A model is a declarative file, not hidden code. One spec, two compilation
targets:

| target | module | what it gives you |
|---|---|---|
| PyTorch | `nanoforge.compiler` | train, evaluate, generate |
| nanollama.c `.bin` | `nanoforge.exporter` | ship to the C engine |

The spec is the single source of truth. The studio's Canvas is a *view* over
it; every run snapshots the spec it trained with (`run.json`).

## Example

```yaml
name: tiny-char-lm
tokenizer: byte            # byte (engine-ready ids) | bpe (nanobrain tokenizer)
model:
  dim: 64                  # residual width
  hidden: 176              # default FFN width
  heads: 4                 # must divide dim
  kv_heads: 4              # optional GQA; must divide heads
  seq_len: 128
  n_layers: 3              # shorthand: 3 × [attention, swiglu_ffn]
  # layers:                # explicit stack — the "own architecture" path
  #   - kind: attention
  #     params: {heads: 4}
  #   - kind: swiglu_ffn
  #     params: {hidden: 176}
  #   - kind: gelu_ffn     # studio-only blocks welcome; see export rules
  head: lm                 # lm | classification
  tied: true               # tie classifier to embedding (engine-compatible)
train:
  steps: 300
  batch: 32
  lr: 0.003
  warmup: 30
  schedule: cosine         # cosine | constant
  eval_every: 50
  sample_every: 100
  val_split: 0.1
data:
  path: data/stories.txt   # text for lm; csv for classification
  label_column: label      # csv only
export:
  formats: [nanollama]
  quantize: [int8]         # quality report via engine-faithful emulation
```

## Block library (v0)

| block | what it is | exportable to the engine |
|---|---|---|
| `attention` | causal multi-head self-attention, RoPE (adjacent-pair convention, base 10000) | ✅ |
| `swiglu_ffn` | SwiGLU feed-forward | ✅ |
| `rmsnorm` | standalone RMSNorm | ✅ (as part of canonical pairs) |
| `gelu_ffn` | classic GELU MLP | ❌ trains in-studio only |
| `conv1d` | causal depthwise temporal conv | ❌ trains in-studio only |

Custom stacks compile and train fine. Export requires the canonical repeating
pattern `[attention, swiglu_ffn]` — the engine's weight format is exactly that
architecture, and nanoforge refuses to pretend otherwise.

## Engine conventions (why 259? why +3?)

nanollama.c hardcodes the Llama-2 id convention: ids 0/1/2 are
`<unk>/<s>/</s>` and raw bytes live at `3..258` (its byte-fallback rule is
`token = byte + 3`). nanoforge's `byte` tokenizer bakes that in: **every
byte model trains with engine-ready ids from step one**, so an exported
`.bin` decodes through the engine's llama2-style tokenizer with no extra
work. Vocab is 259.

For subword models, set `tokenizer: bpe` + `bpe_model: <path>` and use a
tokenizer trained by nanobrain — its ID_MAP uses the same convention.

## Validation contract

`load_spec()` raises `ValueError` with a human-readable message for: unknown
blocks, `dim % heads != 0`, `kv_heads` not dividing `heads`, missing data
files, bad warmup/lr/val_split ranges, classification without `n_classes`,
and unsupported quantize formats. The studio saves every edit but shows the
validation result next to the editor — the file stays the source of truth.
