"""The studio server: Python's stdlib HTTP server, nothing else.

No Flask, no FastAPI, no websockets — REST + polling, bound to 127.0.0.1.
That's not nostalgia; it's the brand: nanoforge runs with the same dependency
list it documents (torch + numpy + pyyaml), and the studio works on a machine
that has never seen the internet.
"""

from __future__ import annotations

import json
import os
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import yaml

from . import __version__, registry
from .agent.loop import AgentSession
from .runner import MANAGER
from .spec import load_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO = os.path.join(ROOT, "studio")
TEMPLATES = os.path.join(ROOT, "templates")
WORKSPACE = os.path.join(os.getcwd(), "workspace")
MAX_BODY = 256 * 1024


class Api:
    """All /api handlers. Each returns (status_code, payload)."""

    def __init__(self, workspace: str):
        self.workspace = os.path.abspath(workspace)
        os.makedirs(self.workspace, exist_ok=True)
        self.session = AgentSession(self.workspace)
        self._model_cache: dict = {}

    # -- helpers -----------------------------------------------------------
    def _project(self, name: str) -> str:
        path = os.path.abspath(os.path.join(self.workspace, name or ""))
        if not path.startswith(self.workspace + os.sep) or not os.path.isdir(path):
            raise FileNotFoundError(f"no such project: {name}")
        return path

    def _spec_path(self, name: str) -> str:
        return os.path.join(self._project(name), "model.nanospec")

    def _model_for(self, project: str, run_id: str):
        from .cli import load_run_model
        key = (project, run_id)
        if key not in self._model_cache:
            self._model_cache[key] = load_run_model(project, run_id or None)
        return self._model_cache[key]

    # -- endpoints -----------------------------------------------------------
    def state(self, qs):
        projects = sorted(
            d for d in os.listdir(self.workspace)
            if os.path.isdir(os.path.join(self.workspace, d))
            and os.path.exists(os.path.join(self.workspace, d, "model.nanospec"))
        )
        templates = sorted(
            d for d in os.listdir(TEMPLATES)
            if os.path.isdir(os.path.join(TEMPLATES, d))
        )
        return 200, {"version": __version__, "workspace": self.workspace,
                     "projects": projects, "templates": templates,
                     "active": MANAGER.status()}

    def project(self, qs):
        name = (qs.get("name") or [""])[0]
        path = self._project(name)
        spec_path = os.path.join(path, "model.nanospec")
        with open(spec_path, encoding="utf-8") as f:
            spec_yaml = f.read()
        validation = {"ok": True, "error": None}
        extra = {}
        try:
            spec = load_spec(spec_path)
            from .compiler import estimate_params
            ok, why = spec.is_exportable()
            extra = {
                "exportable": ok, "export_block": None if ok else why,
                "stack": spec.stack_summary(),
                "params": estimate_params(spec, spec.vocab_size, None),
            }
        except ValueError as e:
            validation = {"ok": False, "error": str(e)}

        files = []
        for base, dirs, fs in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
            rel = os.path.relpath(base, path)
            for fn in fs:
                p = os.path.normpath(os.path.join(rel, fn)).replace("\\", "/")
                files.append(p)
        spec_struct = None
        if validation["ok"]:
            spec = load_spec(spec_path)
            spec_struct = {
                "name": spec.name,
                "tokenizer": spec.tokenizer,
                "head": spec.model.head,
                "dim": spec.model.dim, "heads": spec.model.heads,
                "hidden": spec.model.hidden, "seq_len": spec.model.seq_len,
                "vocab": spec.vocab_size,
                "layers": [{"kind": b.kind, "params": b.params} for b in spec.resolved_layers()],
                "train": {"steps": spec.train.steps, "batch": spec.train.batch,
                          "lr": spec.train.lr, "warmup": spec.train.warmup},
            }
        return 200, {"name": name, "spec_yaml": spec_yaml, "validation": validation,
                     "spec_struct": spec_struct,
                     "runs": registry.list_runs(path), "files": sorted(files), **extra}

    def file(self, qs):
        from .agent.tools import _resolve
        path = self._project((qs.get("project") or [""])[0])
        target = _resolve(path, (qs.get("path") or [""])[0])
        if not os.path.isfile(target):
            return 404, {"error": "no such file"}
        if os.path.splitext(target)[1] in (".pt", ".bin"):
            return 200, {"path": target, "binary": True,
                         "size": os.path.getsize(target)}
        with open(target, encoding="utf-8", errors="replace") as f:
            return 200, {"path": target, "content": f.read(4096)}

    BLOCK_KINDS = ("attention", "swiglu_ffn", "gelu_ffn", "conv1d", "rmsnorm")

    def spec_edit(self, body):
        """Structured spec edits from the architecture canvas — the server is
        the YAML engine, the canvas is just a view."""
        path = self._spec_path(body["project"])
        doc = yaml.safe_load(open(path, encoding="utf-8"))
        op = body.get("op")
        model = doc.setdefault("model", {})
        layers = model.get("layers")
        if op != "add_pair" and layers is None:
            # materialize the shorthand into an explicit stack on first edit
            n = int(model.get("n_layers", 3) or 3)
            layers = []
            for _ in range(n):
                layers += [{"kind": "attention"}, {"kind": "swiglu_ffn"}]
            model.pop("n_layers", None)
            model["layers"] = layers
        if op == "add_pair":
            model["layers"] = (model.get("layers") or []) + [{"kind": "attention"},
                                                             {"kind": "swiglu_ffn"}]
        elif op == "add_block":
            kind = body.get("kind")
            if kind not in self.BLOCK_KINDS:
                return 400, {"error": f"unknown block kind {kind}"}
            model["layers"].append({"kind": kind} if kind in ("attention",) else {"kind": kind})
        elif op == "remove_block":
            i = int(body["index"])
            if not 0 <= i < len(model["layers"]):
                return 400, {"error": "bad index"}
            model["layers"].pop(i)
        elif op == "move_block":
            i, d = int(body["index"]), 1 if body.get("dir") == "down" else -1
            j = i + d
            if not (0 <= i < len(model["layers"]) and 0 <= j < len(model["layers"])):
                return 400, {"error": "can't move there"}
            model["layers"][i], model["layers"][j] = model["layers"][j], model["layers"][i]
        elif op == "set_block_param":
            i = int(body["index"])
            layer = model["layers"][i]
            key, value = body["key"], body["value"]
            if value in ("", None):
                layer.get("params", {}).pop(key, None)
            else:
                v = value
                if isinstance(value, str) and value.replace(".", "", 1).isdigit():
                    v = float(value) if "." in value else int(value)
                layer.setdefault("params", {})[key] = v
        elif op == "set_train":
            key, value = body["key"], body["value"]
            v = value
            if isinstance(value, str) and value.replace(".", "", 1).isdigit():
                v = float(value) if "." in value else int(value)
            doc.setdefault("train", {})[key] = v
        else:
            return 400, {"error": f"unknown op {op}"}
        if not model.get("layers"):
            return 400, {"error": "the layer stack can't be empty"}
        content = yaml.safe_dump(doc, sort_keys=False)
        tmp = path + ".pending"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        validation = {"ok": True, "error": None}
        try:
            load_spec(tmp)
        except ValueError as e:
            validation = {"ok": False, "error": str(e)}
        os.replace(tmp, path)
        return 200, {"saved": True, "validation": validation}

    def save_spec(self, body):
        path = self._spec_path(body["project"])
        content = body["spec_yaml"]
        tmp = path + ".pending"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        validation = {"ok": True, "error": None}
        try:
            load_spec(tmp)
        except ValueError as e:
            validation = {"ok": False, "error": str(e)}
        os.replace(tmp, path)  # save regardless — the editor is the source of truth
        return 200, {"saved": True, "validation": validation}

    def train(self, body):
        project = self._project(body["project"])
        overrides = {k: body[k] for k in ("steps", "batch", "lr") if body.get(k) is not None}
        try:
            run_dir = MANAGER.start(project, os.path.join(project, "model.nanospec"),
                                    overrides=overrides or None, device=body.get("device"))
        except (RuntimeError, ValueError) as e:
            return 400, {"error": str(e)}
        return 200, {"run_dir": run_dir, "id": os.path.basename(run_dir)}

    def stop(self, body):
        return 200, {"stopping": MANAGER.stop()}

    def active(self, qs):
        st = MANAGER.status()
        if st.get("run_dir"):
            st["metrics"] = registry.read_metrics(st["run_dir"])[-400:]
        return 200, st

    def runs(self, qs):
        path = self._project((qs.get("project") or [""])[0])
        return 200, {"runs": registry.list_runs(path)}

    def run(self, qs):
        path = self._project((qs.get("project") or [""])[0])
        rid = (qs.get("id") or [""])[0]
        _, run = registry.read_run(rid, path)
        run["metrics_log"] = registry.read_metrics(rid, path)
        samples = os.path.join(registry.runs_dir(path), rid, "samples.txt")
        if os.path.exists(samples):
            with open(samples, encoding="utf-8") as f:
                run["samples"] = f.read()[-4000:]
        return 200, run

    def generate(self, body):
        from .trainer import sample_text
        project = self._project(body["project"])
        run, run_dir, spec, model, _data, tok = self._model_for(project, body.get("run_id"))
        if spec.model.head != "lm":
            return 400, {"error": "generate is for language models"}
        device = "cuda" if body.get("device", "auto") == "cuda" and torch_cuda() else "cpu"
        model.to(device)
        text = sample_text(model, body.get("prompt", "Once upon a time"),
                           int(body.get("tokens", 160)), float(body.get("temperature", 0.8)))
        return 200, {"text": text, "run": run["id"]}

    def export(self, body):
        from .exporter import export_run
        project = self._project(body["project"])
        try:
            res = export_run(project, body["run_id"])
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            return 400, {"error": str(e)}
        self._model_cache.clear()
        return 200, {"files": list(res["files"]), "report": res["report"]}

    def report(self, qs):
        path = self._project((qs.get("project") or [""])[0])
        rid = (qs.get("id") or [""])[0]
        p = os.path.join(path, "runs", rid, "export", "quality_report.md")
        if not os.path.exists(p):
            return 404, {"error": "no quality report yet — export this run first"}
        with open(p, encoding="utf-8") as f:
            return 200, {"report": f.read()}

    def agent(self, body):
        project = self._project(body["project"])
        try:
            return 200, self.session.step(project, body.get("message", ""),
                                          confirm=bool(body.get("confirm")))
        except Exception as e:  # noqa: BLE001
            return 500, {"error": f"{type(e).__name__}: {e}"}

    def init_project(self, body):
        name = (body.get("name") or "").strip()
        template = body.get("template", "tiny-char-lm")
        src = os.path.join(TEMPLATES, template)
        if not name or os.path.basename(name) != name:
            return 400, {"error": "project name must be a plain folder name"}
        if template not in os.listdir(TEMPLATES):
            return 400, {"error": f"unknown template {template}"}
        dst = os.path.join(self.workspace, name)
        if os.path.exists(dst):
            return 400, {"error": f"project '{name}' already exists"}
        shutil.copytree(src, dst)
        return 200, {"created": name}


def torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet — the console belongs to training
            pass

        def _json(self, code: int, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _static(self, rel: str):
            path = os.path.normpath(os.path.join(STUDIO, rel.lstrip("/")))
            if not path.startswith(STUDIO) or not os.path.isfile(path):
                self._json(404, {"error": "not found"})
                return
            ctype = {".html": "text/html", ".js": "text/javascript",
                     ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml",
                     ".ico": "image/x-icon"}.get(os.path.splitext(path)[1], "text/plain")
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path.startswith("/api/"):
                fn = getattr(api, u.path[len("/api/"):].strip("/"), None)
                if fn is None:
                    return self._json(404, {"error": f"no endpoint {u.path}"})
                try:
                    code, payload = fn(parse_qs(u.query))
                except (FileNotFoundError, ValueError) as e:
                    return self._json(400, {"error": str(e)})
                except Exception as e:  # noqa: BLE001
                    return self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return self._json(code, payload)
            return self._static("index.html" if u.path in ("/", "") else u.path)

        def do_PUT(self):
            if not urlparse(self.path).path.startswith("/api/"):
                return self._json(404, {"error": "not found"})
            length = min(int(self.headers.get("Content-Length", 0)), MAX_BODY)
            body = json.loads(self.rfile.read(length) or b"{}")
            fn = getattr(api, urlparse(self.path).path[len("/api/"):].strip("/"), None)
            if fn is None:
                return self._json(404, {"error": "no endpoint"})
            try:
                code, payload = fn(body)
            except (FileNotFoundError, ValueError) as e:
                return self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return self._json(code, payload)

        def do_POST(self):
            if not urlparse(self.path).path.startswith("/api/"):
                return self._json(404, {"error": "not found"})
            length = min(int(self.headers.get("Content-Length", 0)), MAX_BODY)
            body = json.loads(self.rfile.read(length) or b"{}")
            fn = getattr(api, urlparse(self.path).path[len("/api/"):].strip("/"), None)
            if fn is None:
                return self._json(404, {"error": "no endpoint"})
            try:
                code, payload = fn(body)
            except (FileNotFoundError, ValueError) as e:
                return self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return self._json(code, payload)

    return Handler


def serve(workspace: str, port: int):
    api = Api(workspace)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(api))
    print(f"nanoforge studio → http://127.0.0.1:{port}")
    print(f"workspace: {api.workspace}   (offline: bound to localhost, no telemetry)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping studio…")
