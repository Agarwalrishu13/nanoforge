"""The agent's hands: a small whitelist of actions, executed in a sandbox.

Every action the model can take is one of these — no shell, no network, no
freeform code execution. Write actions produce a diff preview and only apply
on explicit confirmation. Paths are resolved inside the project folder and
validated against escapes.
"""

from __future__ import annotations

import difflib
import os

READ_ACTIONS = ("read_spec", "list_files", "read_run")
WRITE_ACTIONS = ("write_spec",)


class SandboxError(Exception):
    pass


def _resolve(project: str, rel: str) -> str:
    root = os.path.abspath(project)
    target = os.path.abspath(os.path.join(root, rel or "."))
    if target != root and not target.startswith(root + os.sep):
        raise SandboxError(f"path escapes the project: {rel}")
    return target


def read_spec(project: str, rel: str = "model.nanospec") -> str:
    path = _resolve(project, rel)
    if not os.path.exists(path):
        raise SandboxError(f"no such file: {rel}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def list_files(project: str) -> list[str]:
    root = os.path.abspath(project)
    out = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        rel = os.path.relpath(base, root)
        for f in files:
            if f.endswith((".pt", ".pyc")):
                continue
            out.append(os.path.normpath(os.path.join(rel, f)).replace("\\", "/"))
    return sorted(out)[:80]


def read_run(project: str, run_id: str) -> dict:
    from . import registry
    _, run = registry.read_run(run_id, project)
    return {
        "id": run["id"], "status": run["status"], "error": run.get("error"),
        "metrics": run.get("metrics"), "name": run.get("name"),
    }


def write_spec(project: str, content: str, confirm: bool) -> dict:
    """Replace model.nanospec — only ever after the diff was shown + confirmed."""
    path = _resolve(project, "model.nanospec")
    old = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = f.read()
    diff = "".join(
        difflib.unified_diff(old.splitlines(True), content.splitlines(True),
                             "model.nanospec (current)", "model.nanospec (proposed)")
    )
    if not confirm:
        return {"status": "preview", "diff": diff}
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "applied", "diff": diff}


def execute(project: str, action: dict, confirm: bool = False) -> dict:
    """Execute one whitelisted action. Unknown actions are refused, not guessed."""
    kind = action.get("kind")
    if kind == "read_spec":
        return {"status": "ok", "content": read_spec(project, action.get("path", "model.nanospec"))}
    if kind == "list_files":
        return {"status": "ok", "files": list_files(project)}
    if kind == "read_run":
        return {"status": "ok", "run": read_run(project, action["run_id"])}
    if kind == "write_spec":
        return write_spec(project, action["content"], confirm)
    raise SandboxError(f"action '{kind}' is not in the whitelist — the agent can only "
                       f"read files, read runs, and (with your confirmation) rewrite the spec")
