# tiny-classifier

Proof that nanoforge is not only language models: a two-block GELU MLP
(~8.7k params) that classifies the bundled 3-class tabular dataset to
~100% val accuracy in ~4 s.

- head: `classification` (mean-pool over features, 3 classes)
- architecture: custom stack of two `gelu_ffn` blocks
- export: studio-only — classifiers don't map to the engine's weight format
  yet (ONNX export is on the roadmap), and the studio says so honestly

```bash
python -m nanoforge train .
python -m nanoforge eval .
```
