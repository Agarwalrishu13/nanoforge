"""nanoforge's zero-dependency test runner.

    python tests/run_all.py            # everything torch is installed
    python tests/run_all.py --fast     # spec-level only (no torch needed)
    python -m nanoforge selftest       # same thing via the CLI
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MODULES = ["test_spec", "test_compiler", "test_export", "test_e2e_smoke"]


def run_all(fast: bool = False, verbose: bool = False) -> list[str]:
    failures = []
    if HERE not in sys.path:
        sys.path.insert(0, HERE)  # test_* modules import as top-level names
    for mod_name in MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            if fast and "torch" in str(e):
                if verbose:
                    print(f"skip {mod_name} (--fast, needs torch)")
                continue
            raise
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                if verbose:
                    print(f"PASS {mod_name}.{name}")
            except Exception:  # noqa: BLE001
                failures.append(f"{mod_name}.{name}")
                print(f"FAIL {mod_name}.{name}")
                traceback.print_exc()
    print(f"\n{len(failures)} failure(s)" if failures else "\nall tests passed")
    return failures


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    sys.exit(1 if run_all(fast=fast, verbose=True) else 0)
