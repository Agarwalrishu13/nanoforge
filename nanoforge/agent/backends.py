"""Agent backends. Two, behind one interface:

  HeuristicBackend — fully offline rule-based brain. It can explain a spec,
  suggest hyperparameters, and diagnose failed runs from run.json's error and
  traceback, proposing a fixed spec as a confirmed diff. This is what ships
  today: it always works, needs no model at all, and demonstrates the
  constrained-action loop end to end.

  EngineBackend — shells out to a nanollama.c binary for freeform replies.
  Wired and real (it talks to the engine you built), but a 15M TinyStories
  model is a poet, not a sysadmin — heuristic actions stay the default.

The interface: handle(message, context) -> {"reply": str, "actions": [...]}.
"""

from __future__ import annotations

import os
import subprocess

import yaml


# ----------------------------------------------------------------- heuristic --
class HeuristicBackend:
    name = "heuristic"

    def handle(self, message: str, ctx: dict) -> dict:
        msg = message.lower()
        if any(w in msg for w in ("error", "fail", "wrong", "nan", "broken", "crash", "why")):
            return self.diagnose(ctx)
        if any(w in msg for w in ("suggest", "hyper", "parameter", "tune", "improve", "better")):
            return self.suggest(ctx)
        if any(w in msg for w in ("explain", "what does", "spec", "architecture", "how")):
            return self.explain(ctx)
        return {
            "reply": (
                "I'm the offline nanoforge agent — no API behind me, just rules and "
                "your project files. I can:\n"
                "  • `explain my spec` — walk through model.nanospec\n"
                "  • `suggest parameters` — propose better hyperparameters (as a "
                "reviewable diff, I never write without your confirm)\n"
                "  • `why did it fail?` — diagnose the last failed run\n"
                "Point `agent_backend: engine` in forge.json at a nanollama.c binary "
                "to swap in a real model."
            ),
            "actions": [],
        }

    # -- skills ----------------------------------------------------------------
    def explain(self, ctx: dict) -> dict:
        spec = ctx["spec"]
        ms = spec["model"]
        stack = " -> ".join(
            b["kind"] + (f"({b['params']['heads']}h)" if b["kind"] == "attention" and "heads" in b["params"] else "")
            for b in (ms.get("layers") or [])
        ) or f"{ms.get('n_layers')} canonical blocks"
        reply = (
            f"**{spec['name']}** — {spec.get('tokenizer', 'byte')} tokenizer, "
            f"{'language model' if ms.get('head', 'lm') == 'lm' else 'classifier'}.\n"
            f"- dim {ms['dim']} · heads {ms['heads']} · seq {ms['seq_len']}\n"
            f"- stack: {stack}\n"
            f"- training: {spec['train']['steps']} steps, batch {spec['train']['batch']}, "
            f"lr {spec['train']['lr']}, {spec['train']['schedule']} schedule\n"
            f"- export: {', '.join(spec.get('export', {}).get('formats', ['nanollama']))}\n"
            "The stack is your architecture — add/reorder blocks in the Canvas tab "
            "and this file rewrites itself."
        )
        return {"reply": reply, "actions": [{"kind": "read_spec"}]}

    def suggest(self, ctx: dict) -> dict:
        spec = ctx["spec"]
        t = spec["train"]
        ms = spec["model"]
        fixes = {}
        notes = []
        if t["steps"] >= 10 * t["warmup"] is False:
            fixes["warmup"] = max(10, t["steps"] // 20)
            notes.append(f"warmup was tiny; {fixes['warmup']} steps (~5%) is healthier")
        if t["lr"] > 1e-2:
            fixes["lr"] = 3e-3
            notes.append("lr > 1e-2 NaNs small transformers; 3e-3 with clip is the safe zone")
        if t["lr"] < 1e-4 and t["steps"] < 2000:
            fixes["lr"] = 1e-3
            notes.append("with <2k steps, lr <1e-4 barely moves the loss")
        if t["batch"] * ms["seq_len"] > 32768:
            fixes["batch"] = max(8, 32768 // ms["seq_len"])
            notes.append(f"batch×seq is large for a laptop; batch {fixes['batch']} is friendlier")
        if not notes:
            notes.append("hyperparameters look sane — spend your budget on data, not tuning")
        proposed = _patch_train(ctx["spec_yaml"], fixes)
        actions = []
        if fixes:
            actions.append({"kind": "write_spec", "content": proposed})
        return {"reply": "**Suggestions**\n- " + "\n- ".join(notes) +
                         ("\n\nI've attached a spec patch — review it and hit Apply if you agree."
                          if fixes else ""), "actions": actions}

    def diagnose(self, ctx: dict) -> dict:
        run = ctx.get("last_failed_run")
        if not run:
            return {"reply": "I don't see a failed run in this project's registry — "
                             "train once and I'll read its run.json.", "actions": []}
        error = run.get("error") or ""
        fixes, notes = _diagnose_error(error, run)
        reply = f"**Run `{run['id']}` failed** with `{error}`\n"
        if notes:
            reply += "\n".join(f"- {n}" for n in notes)
        actions = []
        if fixes:
            proposed = _patch_train(ctx["spec_yaml"], fixes)
            actions.append({"kind": "write_spec", "content": proposed})
            reply += "\n\nProposed a spec patch — review the diff, then Apply."
        return {"reply": reply, "actions": actions}


def _diagnose_error(error: str, run: dict) -> tuple[dict, list[str]]:
    e = error.lower()
    fixes, notes = {}, []
    if "nan" in e or "loss is nan" in e:
        fixes.update(lr=1e-3, warmup=50)
        notes.append("NaN losses on tiny transformers are almost always lr too high / warmup too short")
    if "out of memory" in e or "cuda oom" in e:
        fixes.update(batch=16)
        notes.append("OOM → halve the batch (or seq_len); small models don't need big batches")
    if "divides" in e or "%" in e and "0" in e:
        notes.append("heads must divide dim exactly — the engine requires it too")
    if "not found" in e:
        notes.append("a file path in the spec doesn't exist — check data.path against the file tree")
    if "vocab" in e and "mismatch" in e:
        notes.append("checkpoint vocab ≠ spec vocab — retrain after changing the tokenizer or vocab")
    if "expected" in e and "state_dict" in e:
        notes.append("architecture changed after training — export the run that matches the spec")
    if not notes:
        notes.append("no known pattern matched; open run.json's traceback for the full stack")
    return fixes, notes


def _patch_train(spec_yaml: str, fixes: dict) -> str:
    doc = yaml.safe_load(spec_yaml)
    doc.setdefault("train", {}).update(fixes)
    return yaml.safe_dump(doc, sort_keys=False)


# -------------------------------------------------------------------- engine --
class EngineBackend:
    """Freeform replies from a nanollama.c binary. Real, offline, and — with a
    toy-sized model — best enjoyed as performance art. Configure in forge.json:

        {"agent_backend": "engine",
         "engine": {"bin": "../nanollama.c/nanollama.exe",
                    "model": "../nanollama.c/models/nanobrain-tiny.bin"}}
    """

    name = "engine"

    def __init__(self, bin_path: str, model_path: str):
        self.bin_path = os.path.abspath(bin_path)
        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.bin_path):
            raise FileNotFoundError(f"engine binary not found: {self.bin_path}")

    def handle(self, message: str, ctx: dict) -> dict:
        prompt = f"Instruction: {message}\nAnswer:"
        try:
            out = subprocess.run(
                [self.bin_path, "run", "-m", self.model_path, "-i", prompt, "-n", "60", "-t", "0.8"],
                capture_output=True, text=True, timeout=120, shell=False,
            )
            reply = (out.stdout or out.stderr).strip() or "(the engine said nothing)"
        except Exception as e:  # noqa: BLE001
            reply = f"engine backend failed: {type(e).__name__}: {e}"
        return {"reply": reply, "actions": []}


def make_backend(workspace: str):
    cfg_path = os.path.join(workspace, "forge.json")
    cfg = {}
    if os.path.exists(cfg_path):
        import json
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    if cfg.get("agent_backend") == "engine":
        eng = cfg.get("engine") or {}
        try:
            return EngineBackend(eng.get("bin", ""), eng.get("model", ""))
        except FileNotFoundError as e:
            return HeuristicBackend(), str(e)
    return HeuristicBackend(), None
