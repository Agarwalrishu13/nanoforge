"""forge — the command line. Same operations the studio exposes, headless.

    python -m nanoforge init demo --template tiny-char-lm
    python -m nanoforge train workspace/demo --steps 300
    python -m nanoforge generate workspace/demo --run <id> -p "Once upon a time"
    python -m nanoforge export workspace/demo --run <id>
    python -m nanoforge serve --port 8501
    python -m nanoforge selftest
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import torch

from . import __version__, registry
from .compiler import compile_spec
from .data import make_data, resolve_data_path
from .runner import MANAGER
from .spec import load_spec
from .trainer import evaluate, sample_text, train_run

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
VALID_TEMPLATES = ("tiny-char-lm", "tiny-classifier")


def resolve_project(p: str) -> str:
    path = os.path.abspath(p)
    if not os.path.isdir(path):
        sys.exit(f"error: project folder not found: {p}")
    if not os.path.exists(os.path.join(path, "model.nanospec")):
        sys.exit(f"error: {p} has no model.nanospec — is it a nanoforge project?")
    return path


def load_run_model(project: str, run_id: str | None):
    """Rebuild the model from a run's snapshot spec + best checkpoint."""
    run_dir, run = registry.read_run(run_id, project) if run_id else (None, None)
    if run is None:
        run = registry.latest_run(project, "done") or registry.latest_run(project)
        if not run:
            sys.exit("error: no runs in this project's registry yet — train first")
        run_dir = os.path.join(project, "runs", run["id"])
    from .spec import NanoSpec, ModelSpec, TrainSpec, DataSpec, ExportSpec, Block
    sd = run["spec"]
    m = dict(sd["model"])
    layers = [Block(b["kind"], b.get("params", {})) for b in (m.get("layers") or [])]
    m.pop("layers", None)
    spec = NanoSpec(
        name=sd["name"], tokenizer=sd.get("tokenizer", "byte"), bpe_model=sd.get("bpe_model"),
        model=ModelSpec(**m, layers=layers or None),
        train=TrainSpec(**sd.get("train", {})),
        data=DataSpec(**sd.get("data", {})),
        export=ExportSpec(**sd.get("export", {})),
    )
    ckpt = os.path.join(run_dir, "ckpt_best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "ckpt_last.pt")
    if not os.path.exists(ckpt):
        sys.exit(f"error: no checkpoint in {run_dir}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    data_path = os.path.normpath(os.path.join(project, spec.data.path))
    data, tok = make_data(spec, data_path)
    model = compile_spec(spec, run.get("vocab_size") or (tok.vocab_size if tok else spec.vocab_size),
                         getattr(data, "n_features", None))
    model.load_state_dict(ck["model"])
    model.eval()
    return run, run_dir, spec, model, data, tok


# ---------------------------------------------------------------- commands ----
def cmd_init(args):
    if args.template not in VALID_TEMPLATES:
        sys.exit(f"error: template must be one of {VALID_TEMPLATES}")
    src = os.path.join(TEMPLATES_DIR, args.template)
    dst = os.path.abspath(args.name)
    if os.path.exists(dst) and not args.force:
        sys.exit(f"error: {dst} already exists (use --force to overwrite)")
    shutil.copytree(src, dst, dirs_exist_ok=args.force)
    print(f"created project {dst}")
    print(f"  spec:     {os.path.join(dst, 'model.nanospec')}")
    print(f"next:")
    print(f"  python -m nanoforge train {dst}")
    print(f"  python -m nanoforge serve          # then open the studio")
    return dst


def cmd_train(args):
    project = resolve_project(args.project)
    overrides = {k: v for k, v in
                 (("steps", args.steps), ("batch", args.batch), ("lr", args.lr)) if v}
    
    # Show friendly pre-flight check
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Pre-flight check...")
    print(f"  - Device: {device}")
    if device == "cpu":
        print("  [Warning] Training on CPU will be slower. For faster training, use a GPU with CUDA support.")
    print(f"  - Project: {os.path.basename(project)}")
    
    # Load spec to show what we're training
    try:
        from .spec import load_spec
        spec = load_spec(os.path.join(project, "model.nanospec"))
        n_params = sum(p.numel() for p in compile_spec(spec, spec.vocab_size).parameters())
        print(f"  - Model: {spec.name} ({n_params:,} parameters)")
        print(f"  - Architecture: {spec.stack_summary()}")
        print(f"  - Training: {spec.train.steps} steps, batch {spec.train.batch}, lr {spec.train.lr}")
    except Exception as e:
        print(f"  [Warning] Could not load spec: {e}")
        spec = None
    
    print("\nStarting training...")
    run_dir = MANAGER.start(project, os.path.join(project, "model.nanospec"),
                            overrides=overrides or None, device=device)
    print(f"  Run folder: {os.path.basename(run_dir)}")
    print("  (Press Ctrl+C to stop early - your checkpoint will be saved)\n")
    
    seen = 0
    total_steps = spec.train.steps if spec else 0
    try:
        while MANAGER.status()["running"] or seen == 0:
            metrics = registry.read_metrics(run_dir)
            for rec in metrics[seen:]:
                step = rec.get("step", "?")
                progress = f"[{step}/{total_steps}]" if total_steps else f"[step {step}]"
                
                # Format metrics nicely
                if "loss" in rec:
                    loss = rec["loss"]
                    status = "[OK]" if loss < 2.0 else "[Warn]" if loss < 3.0 else "[High]"
                    print(f"  {status} {progress} loss={loss:.4f}")
                elif "val_loss" in rec:
                    print(f"  [Eval] {progress} val_loss={rec['val_loss']:.4f}")
                elif "val_accuracy" in rec:
                    print(f"  [Acc] {progress} accuracy={rec['val_accuracy']:.2%}")
            seen = len(metrics)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopped by user. Saving checkpoint...")
        time.sleep(2)  # Give manager time to save
    
    _, run = registry.read_run(run_dir)
    if run["status"] == "failed":
        print(f"\nTraining FAILED: {run['error']}")
        print("\nTry asking the agent for help:")
        print(f"   python -m nanoforge agent \"why did it fail?\" --project {project}")
        print("\nCommon fixes:")
        print("   - Reduce batch size if you see 'out of memory'")
        print("   - Lower learning rate if you see 'NaN' or 'loss exploded'")
        print("   - Check data file path in model.nanospec")
        sys.exit(1)
    m = run["metrics"]
    best_val = m.get('best_val', 'N/A')
    if isinstance(best_val, float):
        best_val = f"{best_val:.4f}"
    print(f"\nTraining complete!")
    print(f"  Time: {m.get('wall_seconds', 0):.1f}s")
    print(f"  Parameters: {m.get('n_params', 0):,}")
    print(f"  Best validation: {best_val}")
    print(f"\nNext steps:")
    print(f"   - Test it:     python -m nanoforge generate {project} -p \"Once upon a time\"")
    print(f"   - Export it:   python -m nanoforge export {project} --run {run['id']}")
    print(f"   - View logs:   cat {os.path.join(run_dir, 'metrics.jsonl')}")
    return run


def cmd_eval(args):
    project = resolve_project(args.project)
    run, run_dir, spec, model, data, _ = load_run_model(project, args.run)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ev = evaluate(model, data, spec, device, spec.train.eval_batches)
    print(f"Evaluating run {run['id']} (step {run['metrics'].get('steps_done')})...")
    print(f"  Device: {device}")
    print("-" * 50)
    for k, v in ev.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("-" * 50)
    return ev


def cmd_generate(args):
    project = resolve_project(args.project)
    run, _, spec, model, _, tok = load_run_model(project, args.run)
    if spec.model.head != "lm":
        sys.exit("error: generate is for language models - try `eval` for classifiers")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Generating from run {run['id']}...")
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"  Device: {device}")
    print("-" * 50)
    ids = tok.encode(args.prompt)
    out = model.generate(ids, max_new=args.tokens, temperature=args.temperature, top_k=40)
    text = tok.decode(out)
    print(text)
    print("-" * 50)
    print(f"\nGenerated {len(out) - len(ids)} new tokens")


def cmd_export(args):
    from .exporter import export_run
    project = resolve_project(args.project)
    run_id = args.run
    if not run_id:
        latest = registry.latest_run(project)
        if not latest:
            sys.exit("error: no runs in this project's registry yet - train first")
        run_id = latest["id"]
        print(f"(no --run given; exporting latest: {run_id})")
    
    print("Exporting model to nanollama.c format...")
    res = export_run(project, run_id)
    print("\nExported files:")
    for fmt, f in res["files"].items():
        size_mb = os.path.getsize(f) / 1e6
        print(f"  [{fmt.upper()}] {os.path.basename(f)} ({size_mb:.2f} MB)")
    print("\nTo run on nanollama.c:")
    print(f"  nanollama run -m {res['files']['fp32']} -i \"Once upon a time\"")
    if "int8" in res["files"]:
        print(f"  nanollama run -m {res['files']['int8']} -i \"Once upon a time\" -q  # quantized")
    if res["report"]:
        print(f"\nQuality Report:\n{res['report']}")
    return res


def cmd_serve(args):
    from .server import serve
    serve(args.workspace, args.port)


def cmd_agent(args):
    from .agent.loop import AgentSession
    project = resolve_project(args.project)
    session = AgentSession(os.path.dirname(project))
    res = session.step(project, args.message, confirm=args.confirm)
    print(res["reply"])
    for r in res["results"]:
        if r.get("kind") == "write_spec" and r.get("status") == "preview":
            print("\n--- proposed diff (re-run with --confirm to apply) ---")
            print(r.get("diff", "(empty)"))
    return res


def cmd_selftest(args):
    import importlib.util
    path = os.path.join(TEMPLATES_DIR, "..", "tests", "run_all.py")
    spec = importlib.util.spec_from_file_location(
        "forge_tests_runner", os.path.abspath(path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failures = module.run_all(fast=args.fast, verbose=True)
    sys.exit(1 if failures else 0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nanoforge",
                                 description="build, train, and ship tiny AI models — offline")
    ap.add_argument("--version", action="version", version=f"nanoforge {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a project from a template")
    p.add_argument("name", help="project folder to create")
    p.add_argument("--template", default="tiny-char-lm", choices=VALID_TEMPLATES)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("train", help="train a project (one run = one folder in runs/)")
    p.add_argument("project")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval", help="evaluate a run's best checkpoint")
    p.add_argument("project")
    p.add_argument("--run", default=None)
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("generate", help="sample from a run's best checkpoint")
    p.add_argument("project")
    p.add_argument("--run", default=None)
    p.add_argument("-p", "--prompt", default="Once upon a time")
    p.add_argument("-n", "--tokens", type=int, default=160)
    p.add_argument("-t", "--temperature", type=float, default=0.8)
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("export", help="export a run to the nanollama.c .bin format")
    p.add_argument("project")
    p.add_argument("--run", default=None)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("serve", help="run the studio (browser dashboard, localhost only)")
    p.add_argument("--workspace", default=os.path.join(os.getcwd(), "workspace"))
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("agent", help="ask the offline agent about a project")
    p.add_argument("message")
    p.add_argument("--project", required=True)
    p.add_argument("--confirm", action="store_true",
                   help="apply proposed spec writes (without this: preview only)")
    p.set_defaults(fn=cmd_agent)

    p = sub.add_parser("selftest", help="run the built-in test suite")
    p.add_argument("--fast", action="store_true", help="skip tests that need torch")
    p.set_defaults(fn=cmd_selftest)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
