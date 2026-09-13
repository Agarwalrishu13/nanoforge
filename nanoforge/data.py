"""Datasets. Two shapes, chosen by the model head:

    lm             -> raw text: utf-8 byte stream, random-cropped batches
    classification -> csv: numeric feature columns + a label column

Both return plain torch tensors and (train, val) splits. No torchvision,
no HuggingFace datasets — reading your own bytes is not a library problem.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch


class LMData:
    """Random-crop language-model batches over a token stream (nanoGPT style)."""

    def __init__(self, text_bytes: bytes, tokenizer, val_split: float, seq_len: int, seed: int = 0):
        if tokenizer.kind == "byte":
            ids = [b + tokenizer.BYTE_OFFSET for b in text_bytes]
        else:
            ids = tokenizer.encode(text_bytes.decode("utf-8", errors="replace"))
        data = np.array(ids, dtype=np.int64)
        n_val = max(1, int(len(data) * val_split))
        self.val = torch.from_numpy(data[:n_val])
        self.train = torch.from_numpy(data[n_val:])
        self.seq_len = seq_len
        self.seed = seed
        self._rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return 10**9  # stream-style: always more crops

    def batch(self, split: str, batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        stream = self.train if split == "train" else self.val
        # if the dataset is smaller than one window, tile it (tiny demos still work)
        if len(stream) <= self.seq_len + 1:
            reps = (self.seq_len + 2) // len(stream) + 1
            stream = stream.repeat(reps)
        ix = torch.randint(len(stream) - self.seq_len - 1, (batch,), generator=self._rng)
        x = torch.stack([stream[i : i + self.seq_len] for i in ix])
        y = torch.stack([stream[i + 1 : i + 1 + self.seq_len] for i in ix])
        return x.to(device), y.to(device)


class ClassificationData:
    """CSV with numeric feature columns + a label column. Features are
    z-scored with train-set statistics; labels are remapped to 0..n-1."""

    def __init__(self, csv_path: str, label_column: str, val_split: float, seed: int = 0):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"csv has no rows: {csv_path}")
        if label_column not in rows[0]:
            raise ValueError(f"label column '{label_column}' not in csv (has: {list(rows[0])})")

        feature_names = [c for c in rows[0] if c != label_column]
        try:
            X = np.array([[float(r[c]) for c in feature_names] for r in rows], dtype=np.float32)
        except ValueError as e:
            raise ValueError(f"csv feature columns must be numeric ({e})") from e
        raw_labels = [r[label_column] for r in rows]
        classes = sorted(set(raw_labels))
        label_to_id = {c: i for i, c in enumerate(classes)}
        y = np.array([label_to_id[c] for c in raw_labels], dtype=np.int64)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * val_split))
        vi, ti = idx[:n_val], idx[n_val:]

        mean, std = X[ti].mean(0), X[ti].std(0) + 1e-6
        self.feature_names = feature_names
        self.classes = classes
        self.train = (
            torch.from_numpy((X[ti] - mean) / std),
            torch.from_numpy(y[ti]),
        )
        self.val = (
            torch.from_numpy((X[vi] - mean) / std),
            torch.from_numpy(y[vi]),
        )
        self.n_features = X.shape[1]
        self._perm = torch.randperm(len(self.train[0]), generator=torch.Generator().manual_seed(seed))
        self._cursor = 0

    def batch(self, split: str, batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = self.train if split == "train" else self.val
        if split == "train":
            if self._cursor + batch > len(X):
                self._cursor = 0
                self._perm = self._perm[torch.randperm(len(self._perm))]
            ix = self._perm[self._cursor : self._cursor + batch]
            self._cursor += batch
        else:
            ix = torch.arange(len(X))
        return X[ix].to(device), y[ix].to(device)


def make_data(spec, data_path: str, device: str = "cpu"):
    """Build the right dataset for a spec. `data_path` overrides the spec's
    (used when training a project whose cwd differs)."""
    from .tokenizer import make_tokenizer

    if spec.model.head == "lm":
        with open(data_path, "rb") as f:
            text = f.read()
        tok = make_tokenizer(spec.tokenizer, spec.bpe_model)
        return LMData(text, tok, spec.train.val_split, spec.model.seq_len, spec.train.seed), tok
    return ClassificationData(
        data_path, spec.data.label_column, spec.train.val_split, spec.train.seed
    ), None


def resolve_data_path(spec_path: str, spec) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(spec_path)), spec.data.path))
