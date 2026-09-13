"""The training loop: AdamW + warmup + cosine, eval, checkpoints, samples.

Deliberately transparent — this is the same loop nanobrain uses to train its
models, generalized to any compiled nanospec. The run manager (runner.py)
executes it on a thread and the studio watches the metrics file stream by.
"""

from __future__ import annotations

import math
import os
import time
import traceback

import torch
import torch.nn.functional as F

from . import registry
from .compiler import ForgeModel
from .tokenizer import ByteTokenizer


def lr_at(step: int, total: int, base_lr: float, warmup: int, schedule: str) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if schedule == "constant":
        return base_lr
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model: ForgeModel, data, spec, device: str, max_batches: int) -> dict:
    model.eval()
    if spec.model.head == "lm":
        losses = []
        for _ in range(max_batches):
            x, y = data.batch("val", spec.train.batch, device)
            logits = model(x)
            losses.append(
                F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
                ).item()
            )
        model.train()
        return {"val_loss": sum(losses) / len(losses), "val_ppl": float(torch.exp(torch.tensor(sum(losses) / len(losses))))}
    correct = total = 0
    for _ in range(max_batches if spec.model.head == "lm" else 1):
        x, y = data.batch("val", spec.train.batch, device)
        pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return {"val_accuracy": correct / max(1, total)}


@torch.no_grad()
def sample_text(model: ForgeModel, prompt: str, n_tokens: int, temperature: float) -> str:
    tok = ByteTokenizer()
    ids = tok.encode(prompt)
    out = model.generate(ids, max_new=n_tokens, temperature=temperature, top_k=40)
    return tok.decode(out[len(ids):])


def train_run(
    run_dir: str,
    spec,
    model: ForgeModel,
    data,
    device: str,
    overrides: dict | None = None,
    stop_flag=None,
    on_event=None,
) -> dict:
    """Run training to completion (or stop_flag). Writes metrics/checkpoints
    into run_dir as it goes. Returns the final run.json dict."""

    def emit(kind: str, **kw):
        if on_event:
            try:
                on_event(kind, **kw)
            except Exception:
                pass

    t = spec.train
    if overrides:
        for k, v in overrides.items():
            if hasattr(t, k) and v is not None:
                setattr(t, k, v)

    torch.manual_seed(t.seed)
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=t.lr, weight_decay=t.weight_decay, betas=(0.9, 0.95)
    )

    is_lm = spec.model.head == "lm"
    step, best_val = 0, float("inf")
    t0 = time.time()
    model.train()
    emit("start", steps=t.steps, device=device)

    try:
        while step < t.steps:
            if stop_flag is not None and stop_flag.is_set():
                emit("stopped", step=step)
                break

            lr = lr_at(step, t.steps, t.lr, t.warmup, t.schedule)
            for g in opt.param_groups:
                g["lr"] = lr

            x, y = data.batch("train", t.batch, device)
            logits = model(x)
            if is_lm:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            else:
                loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if t.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            opt.step()
            step += 1

            if step % 10 == 0 or step == 1:
                dt = time.time() - t0
                rec = {
                    "step": step,
                    "loss": round(loss.item(), 4),
                    "lr": round(lr, 6),
                    "tok_s": round(step * t.batch * (spec.model.seq_len if is_lm else 1) / max(dt, 1e-6)),
                }
                registry.append_metric(run_dir, rec)
                emit("metric", **rec)

            if is_lm and t.sample_every and step % t.sample_every == 0:
                text = sample_text(model, "Once upon a time", 120, 0.8)
                registry.append_sample(run_dir, step, text)
                emit("sample", step=step)

            if step % t.eval_every == 0 or step == t.steps:
                ev = evaluate(model, data, spec, device, t.eval_batches)
                ev["step"] = step
                registry.append_metric(run_dir, ev)
                emit("eval", **ev)
                val_key = "val_loss" if is_lm else "val_error"
                val = ev.get("val_loss") if is_lm else 1.0 - ev.get("val_accuracy", 0.0)
                if val < best_val:
                    best_val = val
                    torch.save(
                        {"model": model.state_dict(), "step": step, "val": val},
                        os.path.join(run_dir, "ckpt_best.pt"),
                    )

            torch.save(
                {"model": model.state_dict(), "step": step, "val": None, "opt": opt.state_dict()},
                os.path.join(run_dir, "ckpt_last.pt"),
            )

    except Exception as e:  # noqa: BLE001 — the run folder is the user's error log
        _, run = registry.read_run(run_dir)
        run["status"] = "failed"
        run["error"] = f"{type(e).__name__}: {e}"
        run["traceback"] = traceback.format_exc(limit=4)
        run["metrics"]["steps_done"] = step
        registry.update_run(run_dir, run)
        emit("failed", error=run["error"])
        return run

    _, run = registry.read_run(run_dir)
    run["status"] = "done"
    run["metrics"]["steps_done"] = step
    run["metrics"]["best_val"] = best_val if best_val != float("inf") else None
    run["metrics"]["wall_seconds"] = round(time.time() - t0, 1)
    run["metrics"]["n_params"] = sum(p.numel() for p in model.parameters())
    registry.update_run(run_dir, run)
    emit("done", step=step)
    return run
