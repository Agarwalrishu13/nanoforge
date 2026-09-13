# roadmap — where nanoforge stands

The full plan lives in the project notes; this is the honest status board.
Everything marked ✅ is in the repo and verified end to end.

| phase | what | status |
|---|---|---|
| 1. spec compiler + training core | `nanospec` v0, spec→PyTorch compiler, trainer (AdamW, warmup+cosine, eval, samples), run registry, CLI | ✅ |
| 2. engine v2 + agent toolkit | constrained agent loop, whitelist sandbox, heuristic + engine backends, agent read/write protocol | ✅ loop ✅ engine backend · ⏳ nanollama.c v2: Qwen-class arch (RMSNorm/SwiGLU/pretrained-vocab BPE) so a real 1B coder model runs on the engine |
| 3. studio shell | stdlib web studio: spec editor, live loss curves, run registry, generation, agent chat, localhost-only | ✅ |
| 4. architecture canvas | block-stack editor (add/remove/reorder/params → rewrites `model.nanospec` live) | ✅ v0 · ⏳ freeform DAG canvas |
| 5. ship layer | engine `.bin` export, engine-faithful int8 emulation + quality report | ✅ · ⏳ GGUF + ONNX exporters |
| 6. templates + launch | tiny-char-lm + tiny-classifier templates, tests, CI, docs | ✅ · ⏳ demo gif + Show HN |

## verified end to end (this machine)

- `train` 167k-param LM → loss 4.8 → 0.98 (25 s, CUDA)
- `export` → 0.67 MB `.bin` → **loads and generates on nanollama.c** (fp32 1266 tok/s, `-q` int8 859 tok/s)
- classifier template → 100% val accuracy in 4 s
- agent: explain spec / suggest parameters / diagnose failed run, with diff-confirm writes
- 18/18 selftests (`python -m nanoforge selftest`)

## next up

1. nanollama.c v2 — Qwen-class architecture support → run a 1.5B coder model → agent brain upgrade
2. GGUF / ONNX exporters (generosity = adoption)
3. freeform DAG canvas (the stack editor is the power path; the graph is the demo)
4. tokenizer.bin export so byte models can also *encode prompts* inside the engine
5. process-isolated training + resume from `ckpt_last.pt`
