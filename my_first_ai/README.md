# tiny-char-lm

A ~168k-parameter character-level Llama-2 that learns to write the bundled
micro-stories in ~25 s on a laptop GPU (a couple of minutes on CPU).

- tokenizer: `byte` — engine-ready ids (bytes at 3..258), so the exported
  `.bin` decodes directly through nanollama.c's llama2-style tokenizer
- architecture: 3 canonical `[attention, swiglu_ffn]` pairs, dim 64
- export: fp32 `.bin` + engine-faithful int8 quality report

```bash
python -m nanoforge train .        # from this folder
python -m nanoforge generate . -p "Once upon a time"
python -m nanoforge export .
```

Open the studio (`python -m nanoforge serve`) and try the Canvas: add a
`gelu_ffn` block and watch the export badge flip to *studio-only* — the
engine format only accepts the canonical transformer pairs, and nanoforge
won't pretend otherwise.
