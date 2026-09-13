"""The run manager: one training job at a time, on a thread, cancellable.

The studio and the CLI both go through this. A laptop is the target hardware —
a single concurrent run with a hard stop is the honest feature set, and the
stop flag is checked between steps so cancellation is clean.
"""

from __future__ import annotations

import os
import threading

import torch

from . import registry
from .compiler import compile_spec
from .data import make_data, resolve_data_path
from .spec import load_spec
from .trainer import train_run


class RunManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.active_run_dir: str | None = None
        self.active_project: str | None = None
        self.last_error: str | None = None

    def start(self, project: str, spec_path: str, overrides: dict | None = None,
              device: str | None = None) -> str:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError(
                    f"a run is already active ({self.active_run_dir}); stop it first"
                )
            spec = load_spec(spec_path)
            data_path = resolve_data_path(spec_path, spec)
            run_dir = registry.new_run(project, spec.to_dict(), config_note=str(overrides or ""))

            resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            data, tok = make_data(spec, data_path)
            vocab = tok.vocab_size if tok is not None else spec.vocab_size
            n_features = getattr(data, "n_features", None)

            # persist what export needs later (classification feature count, real vocab)
            _, run = registry.read_run(run_dir)
            run["n_features"] = n_features
            run["vocab_size"] = vocab
            registry.update_run(run_dir, run)

            model = compile_spec(spec, vocab, n_features)

            self._stop.clear()
            self.active_run_dir = run_dir
            self.active_project = project
            self.last_error = None

            def worker():
                try:
                    train_run(run_dir, spec, model, data, resolved_device,
                              overrides=overrides, stop_flag=self._stop)
                except Exception as e:  # defensive: the trainer has its own handler
                    self.last_error = f"{type(e).__name__}: {e}"
                finally:
                    self.active_run_dir = None
                    self.active_project = None

            self._thread = threading.Thread(target=worker, daemon=True,
                                            name=f"forge-run-{os.path.basename(run_dir)}")
            self._thread.start()
            return run_dir

    def stop(self) -> bool:
        if self._thread and self._thread.is_alive():
            self._stop.set()
            return True
        return False

    def status(self) -> dict:
        running = bool(self._thread and self._thread.is_alive())
        out = {"running": running, "run_dir": self.active_run_dir,
               "project": self.active_project, "last_error": self.last_error}
        if running and self.active_run_dir:
            try:
                _, run = registry.read_run(self.active_run_dir)
                out["run"] = run
            except (FileNotFoundError, OSError):
                pass
        return out


MANAGER = RunManager()
