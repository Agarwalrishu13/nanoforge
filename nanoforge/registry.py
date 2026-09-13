"""The model registry: every run is a folder, every folder is self-describing.

    <project>/runs/<id>/
        run.json         spec snapshot + config + status + final metrics
        metrics.jsonl    one line per log: {step, loss, lr, val_loss, tok_s, ...}
        samples.txt      periodic generations while training (lm runs)
        ckpt_last.pt     resume point (state dict + step + optimizer state)
        ckpt_best.pt     lowest val loss so far
        export/          shipped artifacts (written by exporter.py)

Folders over a database: you can open them, grep them, and ship them. The
studio's Runs tab and the CLI both read exactly this layout.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime


def runs_dir(project: str) -> str:
    return os.path.join(project, "runs")


def new_run(project: str, spec_dict: dict, config_note: str = "") -> str:
    rid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + spec_dict["name"].replace(" ", "-")
    d = os.path.join(runs_dir(project), rid)
    os.makedirs(d, exist_ok=True)
    run = {
        "id": rid,
        "project": os.path.basename(os.path.normpath(project)),
        "status": "running",
        "created": time.time(),
        "spec": spec_dict,
        "config_note": config_note,
        "metrics": {},
        "error": None,
    }
    update_run(d, run)
    return d


def run_path(run_id_or_dir: str, project: str | None = None) -> str:
    if os.path.isdir(run_id_or_dir):
        return run_id_or_dir
    if project:
        p = os.path.join(runs_dir(project), run_id_or_dir)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"run not found: {run_id_or_dir}")


def update_run(run_dir: str, run: dict) -> None:
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)


def read_run(run_id_or_dir: str, project: str | None = None) -> tuple[str, dict]:
    d = run_path(run_id_or_dir, project)
    with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
        return d, json.load(f)


def append_metric(run_dir: str, record: dict) -> None:
    with open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_metrics(run_id_or_dir: str, project: str | None = None) -> list[dict]:
    d = run_path(run_id_or_dir, project)
    path = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn write while training is live
    return out


def append_sample(run_dir: str, step: int, text: str) -> None:
    with open(os.path.join(run_dir, "samples.txt"), "a", encoding="utf-8") as f:
        f.write(f"--- step {step} ---\n{text}\n\n")


def list_runs(project: str) -> list[dict]:
    d = runs_dir(project)
    if not os.path.isdir(d):
        return []
    out = []
    for rid in sorted(os.listdir(d), reverse=True):
        p = os.path.join(d, rid)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "run.json")):
            try:
                with open(os.path.join(p, "run.json"), encoding="utf-8") as f:
                    out.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    return out


def latest_run(project: str, status: str | None = None) -> dict | None:
    for r in list_runs(project):
        if status is None or r.get("status") == status:
            return r
    return None
