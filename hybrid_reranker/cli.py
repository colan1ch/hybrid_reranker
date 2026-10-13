import argparse
import platform
import shutil
import tomllib
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from .io import write_json


def main():
    parser = argparse.ArgumentParser(description="Selective LLM reranking on BEIR HotpotQA")
    parser.add_argument("--config", default="config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "retrieve", "tune"]:
        commands.add_parser(name)
    methods = ["dense", "bert", "llm", "cascade_gap", "cascade_cosine"]
    for name in ["run", "evaluate"]:
        command = commands.add_parser(name)
        command.add_argument("--split", choices=["validation", "test"], default="test")
        command.add_argument("--method", choices=[*methods, "all"], default="all")
        command.add_argument("--answers", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    config_path = Path(args.config)
    config = tomllib.loads(config_path.read_text())
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "config.toml"
    if snapshot.exists() and tomllib.loads(snapshot.read_text()) != config:
        print("Config changed. Set a new run_dir to keep experiment results separate.")
        return
    if not snapshot.exists():
        shutil.copyfile(config_path, snapshot)
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(),
            "packages": {name: version(name) for name in [
                "numpy", "pyarrow", "huggingface-hub", "sentence-transformers", "transformers",
                "torch", "groq", "tiktoken",
            ]},
        })
    if args.command == "prepare":
        from .data import prepare
        prepare(config)
    elif args.command == "retrieve":
        from .search import retrieve
        retrieve(config)
    elif args.command == "run":
        from .run import run
        for method in methods if args.method == "all" else [args.method]:
            if not run(config, args.split, method, args.answers):
                break
    elif args.command == "tune":
        from .evaluate import tune
        tune(config)
    else:
        from .evaluate import evaluate
        evaluate(config, args.split, methods if args.method == "all" else [args.method], args.answers)
