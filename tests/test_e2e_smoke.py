"""End-to-end smoke: spec -> train -> registry -> export -> quality report.

Runs on CPU with a tiny model so it finishes in seconds — this is the same
path the studio and the CLI take, just scripted.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORIES = os.path.join(ROOT, "templates", "tiny-char-lm", "data", "stories.txt")


def _make_project(td):
    project = os.path.join(td, "smoke")
    os.makedirs(os.path.join(project, "data"))
    shutil.copy(STORIES, os.path.join(project, "data", "stories.txt"))
    doc = yaml.safe_load(open(os.path.join(ROOT, "templates", "tiny-char-lm", "model.nanospec")))
    doc["model"].update({"dim": 32, "hidden": 88, "seq_len": 64})
    doc["train"].update({"steps": 40, "batch": 16, "warmup": 5, "eval_every": 20,
                         "sample_every": 20, "eval_batches": 3})
    with open(os.path.join(project, "model.nanospec"), "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f)
    return project


def test_full_pipeline_cpu():
    from nanoforge.data import make_data, resolve_data_path
    from nanoforge.compiler import compile_spec
    from nanoforge.exporter import export_run
    from nanoforge.spec import load_spec
    from nanoforge.trainer import train_run
    from nanoforge import registry

    with tempfile.TemporaryDirectory() as td:
        project = _make_project(td)
        spec_path = os.path.join(project, "model.nanospec")
        spec = load_spec(spec_path)
        data, tok = make_data(spec, resolve_data_path(spec_path, spec))
        model = compile_spec(spec, spec.vocab_size)
        run_dir = registry.new_run(project, spec.to_dict())
        run = train_run(run_dir, spec, model, data, "cpu")

        assert run["status"] == "done", run.get("error")
        assert run["metrics"]["steps_done"] == 40
        metrics = registry.read_metrics(run_dir)
        first = next(m["loss"] for m in metrics if "loss" in m)
        last = [m["loss"] for m in metrics if "loss" in m][-1]
        assert last < first, f"loss did not drop: {first} -> {last}"

        rid = os.path.basename(run_dir)
        res = export_run(project, rid)
        assert os.path.exists(res["files"]["fp32"])
        assert res["report"] and "fp32" in res["report"]

        # the agent can read the project and explain it — fully offline
        from nanoforge.agent.loop import AgentSession
        session = AgentSession(td)
        out = session.step(project, "explain my spec")
        assert out["reply"] and "tiny-char-lm" in out["reply"]
