<div align="center">

# nanoforge

**The offline IDE for tiny AI — build your own architecture, train it, and ship it to the engine you already run.**

A declarative model spec (`nanospec`) that compiles two ways · a from-scratch
PyTorch trainer · a local model registry · a browser studio with live loss
curves and an architecture canvas · an offline agent that fixes your configs —
and an exporter whose `.bin` files run on
[nanollama.c](https://github.com/Agarwalrishu13/nanollama.c), the C inference
engine written by the same author.

No accounts. No API keys. No telemetry. Airplane-mode approved.

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10+-58a6ff.svg)]()
[![dependencies](https://img.shields.io/badge/runtime%20deps-torch%20·%20numpy%20·%20pyyaml-f0883e.svg)]()
[![engine](https://img.shields.io/badge/ships%20to-nanollama.c-f0883e.svg)](https://github.com/Agarwalrishu13/nanollama.c)

</div>

---

## The three-repo story

| repo | role |
|---|---|
| [nanollama.c](https://github.com/Agarwalrishu13/nanollama.c) | **the engine** — from-scratch C inference: forward pass, KV cache, sampler, thread pool, int8 |
| [nanobrain](https://github.com/Agarwalrishu13/nanobrain) | **the brain factory** — trains one hand-built Llama-2, token by token |
| **nanoforge** (this repo) | **the studio** — turns that entire craft into a tool anyone can drive: *your* architecture, *your* data, *your* model, shipped |

nanobrain proved one person could train a model and run it on their own engine.
nanoforge asks: what if anyone could — offline, on a laptop, guided by an agent
that also runs locally?

## Verified end to end

```
$ python -m nanoforge train workspace/demo
  [train] step=300 | loss=0.9807          ← 167k params, 25 s on a laptop GPU
  done — best val: 1.90

$ python -m nanoforge export workspace/demo
  shipped: .../model.nanollama.bin (0.67 MB)

$ nanollama generate -m model.nanollama.bin -z models/tokenizer.bin -p "" -n 120
  unde a slig the sined to nis. The mars bulk on. Pip to a sus the son bull…
  loaded in 11.1 ms · 1265.9 tok/s          ← trained in the studio, running in C
```

A model that was **described in YAML, trained in a browser studio, and is now
being decoded by a hand-written C engine.** Every token comes from weights
that never touched a cloud.

## What's inside

| component | file | what it does |
|---|---|---|
| `nanospec` v0 | `nanoforge/spec.py` | declarative model definitions + hard validation |
| compiler | `nanoforge/compiler.py`, `blocks.py` | spec → PyTorch; canonical stacks are state-dict-identical to nanobrain |
| trainer | `nanoforge/trainer.py` | AdamW + warmup + cosine, eval, checkpointing, live samples |
| registry | `nanoforge/registry.py`, `runner.py` | every run is a folder: `run.json`, `metrics.jsonl`, checkpoints, exports |
| ship | `nanoforge/exporter.py` | engine `.bin` export (magic `0x616B3432`), engine-faithful int8 emulation, quality report |
| agent | `nanoforge/agent/` | constrained loop: whitelist actions, diff-previewed writes, heuristic brain (pluggable engine backend) |
| studio | `studio/` + `nanoforge/server.py` | browser IDE on Python's stdlib http.server — spec editor, canvas, live curves, chat |

## Quickstart

```bash
pip install -r requirements.txt        # torch, numpy, pyyaml — that's all

python -m nanoforge init my-model --template tiny-char-lm
python -m nanoforge train my-model                    # or open the studio:
python -m nanoforge serve                             # → http://127.0.0.1:8501

python -m nanoforge generate my-model -p "Once upon a time"
python -m nanoforge export   my-model                 # → model.nanollama.bin
python -m nanoforge selftest                          # 18 tests, no pytest needed
```

In the studio: edit the spec or drag blocks on the **Canvas** (edits rewrite
the YAML live), hit **Train** and watch the loss curve, browse **Runs**,
**Ship** to the engine format with a quantization quality report, and ask the
**Agent** to explain your spec, suggest hyperparameters, or diagnose a failed
run — it proposes diffs, you click Apply. It writes one file:
`model.nanospec`. That's the whole sandbox.

## The idea: a model is a file

```yaml
model:
  dim: 64
  heads: 4
  layers:
    - kind: attention
    - kind: swiglu_ffn
    - kind: attention
    - kind: swiglu_ffn
    - kind: gelu_ffn        # your own stack = your own architecture
```

Two compilers read that file: one builds PyTorch for training, one lowers the
weights into nanollama.c's binary format (fp32, magic `0x616B3432`, tied
classifier, bytes at ids 3..258 — the engine's own convention, so exports are
decode-ready). Blocks that don't map to the engine (GELU MLPs, causal convs)
train happily in the studio and refuse to ship with a clear message.

Full reference: [docs/nanospec-v0.md](docs/nanospec-v0.md) ·
status board: [docs/roadmap.md](docs/roadmap.md)

## Why offline

Because your models should belong to you. nanoforge binds to `127.0.0.1`,
ships no telemetry, requires no account, and runs on a machine that has never
seen the internet. The agent behind your chat panel is rules + your own files —
or a quantized model on the engine, if you point it there. Nothing else.

## License

MIT — same as the rest of the nano-stack.
