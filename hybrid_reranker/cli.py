import argparse
import os
import platform
import shutil
import tomllib
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from .checks import VOLATILE_KEYS, comparable, needs_api, prerequisites
from .io import write_json


def main():
    parser = argparse.ArgumentParser(description="Selective LLM reranking on BEIR HotpotQA")
    parser.add_argument("--config", default="config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "retrieve", "tune"]:
        commands.add_parser(name)
    command = commands.add_parser("controls")
    command.add_argument("--split", choices=["validation", "test"], default="test")
    command.add_argument("--repeats", type=int, default=1000)
    command.add_argument("--mock", action="store_true")
    command = commands.add_parser("resplit")
    command.add_argument("--repeats", type=int, default=100)
    command.add_argument("--calibration-size", type=int)
    command.add_argument("--source", choices=["both", "validation", "test"], default="both")
    command.add_argument("--mock", action="store_true")
    command = commands.add_parser("gate")
    command.add_argument("--min-gain", type=float, default=0.0)
    command.add_argument("--mock", action="store_true")
    methods = ["dense", "bert", "llm", "cascade_gap", "cascade_cosine", "cascade_gate"]
    for name in ["run", "evaluate"]:
        command = commands.add_parser(name)
        command.add_argument("--split", choices=["validation", "test"], default="test")
        command.add_argument("--method", choices=[*methods, "all"], default="all")
        command.add_argument("--answers", action="store_true")
        if name == "run":
            command.add_argument(
                "--allow-api", action="store_true",
                help="confirm that the Groq account is on the free plan; required for anything that calls the API",
            )
    args = parser.parse_args()
    load_dotenv()
    config_path = Path(args.config)
    config = tomllib.loads(config_path.read_text())
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "config.toml"
    if snapshot.exists():
        saved = tomllib.loads(snapshot.read_text())
        if comparable(saved) != comparable(config):
            print("Config changed. Set a new run_dir to keep experiment results separate.")
            return
        changed = [key for key in VOLATILE_KEYS if saved.get(key) != config.get(key)]
        if changed:
            print(f"Note: {', '.join(changed)} differs from the saved config; it affects speed only, so the run continues.")
    else:
        shutil.copyfile(config_path, snapshot)
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(),
            "packages": {name: version(name) for name in [
                "numpy", "pyarrow", "huggingface-hub", "sentence-transformers", "transformers",
                "torch", "groq", "tiktoken",
            ]},
        })
    selected = methods if getattr(args, "method", None) == "all" else [getattr(args, "method", None)]
    if getattr(args, "method", None) == "all" and not (run_dir / "gate.json").exists():
        selected = [method for method in selected if method != "cascade_gate"]
    message = prerequisites(args.command, run_dir, getattr(args, "split", None), getattr(args, "method", None), selected)
    if message:
        print(message)
        return
    if needs_api(args.command, args.method if args.command == "run" else "", getattr(args, "answers", False)) and not (
        args.allow_api or os.environ.get("RERANK_ALLOW_API") == "1"
    ):
        print(
            "This command can send requests to the Groq API. Check in the Groq console that the account is on the "
            "Free plan with no payment method, then repeat with --allow-api (or set RERANK_ALLOW_API=1)."
        )
        return
    if args.command == "prepare":
        from .data import prepare
        prepare(config)
    elif args.command == "retrieve":
        from .search import retrieve
        retrieve(config)
    elif args.command == "run":
        from .run import run
        for method in selected:
            if not run(config, args.split, method, args.answers):
                break
    elif args.command == "tune":
        from .evaluate import tune
        tune(config)
    elif args.command == "controls":
        from .controls import controls
        controls(config, args.split, args.repeats, args.mock)
    elif args.command == "resplit":
        from .splits import resplit
        resplit(config, args.repeats, args.calibration_size, args.source, args.mock)
    elif args.command == "gate":
        from .gate import train_gate
        train_gate(config, args.min_gain, args.mock)
    else:
        from .evaluate import evaluate
        evaluate(config, args.split, selected, args.answers)
