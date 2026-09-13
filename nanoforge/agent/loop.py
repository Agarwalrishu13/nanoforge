"""The constrained agent loop.

Flow: message -> backend.handle() -> {reply, actions}. Read actions execute
immediately (they can't hurt). Write actions render as diffs and require an
explicit confirm=true round trip — the human is always the one holding the pen.
Caps: max 4 actions per turn, 16KB per file — a copilot, not an autopilot.
"""

from __future__ import annotations

import os

from . import tools
from .backends import make_backend

MAX_ACTIONS = 4
MAX_FILE_BYTES = 16384


class AgentSession:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.backend, self.backend_warning = make_backend(workspace)

    def step(self, project: str, message: str, confirm: bool = False) -> dict:
        ctx = self._context(project)
        try:
            result = self.backend.handle(message, ctx)
        except Exception as e:  # noqa: BLE001 — never crash the chat
            return {"reply": f"agent backend error: {type(e).__name__}: {e}", "actions": [],
                    "results": []}

        results, executed_writes = [], []
        for action in (result.get("actions") or [])[:MAX_ACTIONS]:
            kind = action.get("kind")
            try:
                if kind == "write_spec":
                    content = action["content"]
                    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                        results.append({"kind": kind, "status": "refused",
                                        "error": "file too large for the agent sandbox"})
                        continue
                    res = tools.execute(project, action, confirm=confirm)
                    res["kind"] = kind
                    results.append(res)
                    if res.get("status") == "applied":
                        executed_writes.append("model.nanospec")
                else:
                    res = tools.execute(project, action)
                    res["kind"] = kind
                    results.append(res)
            except tools.SandboxError as e:
                results.append({"kind": kind, "status": "refused", "error": str(e)})

        return {
            "reply": result.get("reply", ""),
            "actions": [{"kind": a.get("kind")} for a in (result.get("actions") or [])],
            "results": results,
            "backend": getattr(self.backend, "name", "unknown"),
            "backend_warning": self.backend_warning,
            "wrote": executed_writes,
        }

    def _context(self, project: str) -> dict:
        from .. import registry
        ctx = {"project": project}
        spec_path = tools._resolve(project, "model.nanospec")
        if os.path.exists(spec_path):
            with open(spec_path, encoding="utf-8") as f:
                ctx["spec_yaml"] = f.read()
            try:
                from ..spec import load_spec
                ctx["spec"] = _spec_to_ctx(load_spec(spec_path))
            except ValueError as e:
                ctx["spec"] = {"name": "(invalid spec)", "model": {}, "train": {},
                              "error": str(e)}
        else:
            ctx["spec_yaml"] = ""
            ctx["spec"] = {"name": "(no spec yet)", "model": {}, "train": {}}
        failed = registry.latest_run(project, status="failed")
        if failed:
            ctx["last_failed_run"] = failed
        return ctx


def _spec_to_ctx(spec) -> dict:
    return {
        "name": spec.name,
        "tokenizer": spec.tokenizer,
        "model": {
            "dim": spec.model.dim, "heads": spec.model.heads, "seq_len": spec.model.seq_len,
            "head": spec.model.head, "n_classes": spec.model.n_classes,
            "layers": [{"kind": b.kind, "params": b.params} for b in spec.resolved_layers()],
        },
        "train": {
            "steps": spec.train.steps, "batch": spec.train.batch, "lr": spec.train.lr,
            "warmup": spec.train.warmup, "schedule": spec.train.schedule,
            "val_split": spec.train.val_split,
        },
        "export": {"formats": spec.export.formats, "quantize": spec.export.quantize},
    }
