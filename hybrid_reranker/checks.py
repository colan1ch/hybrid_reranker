from pathlib import Path

# Settings that change speed or hardware but not the results; changing them must not block a run.
VOLATILE_KEYS = ("device", "batch_size")


def comparable(config):
    return {key: value for key, value in config.items() if key not in VOLATILE_KEYS}


def needs_api(command, method, answers):
    """True when the command may send requests to the Groq API."""
    return command == "run" and (answers or method in ("llm", "all") or method.startswith("cascade"))


def require(run_dir, names, step):
    missing = [name for name in names if not (Path(run_dir) / name).exists()]
    if missing:
        return f"Missing {', '.join(missing)} in {run_dir}. Run `rerank {step}` first."
    return None


def prerequisites(command, run_dir, split=None, method=None, methods=()):
    """Message about the first input a command is missing, or None when it can start."""
    if command == "retrieve":
        return require(run_dir, ["validation.jsonl", "test.jsonl"], "prepare")
    if command == "tune":
        return (
            require(run_dir, ["validation.jsonl"], "prepare")
            or require(run_dir, ["candidates.jsonl"], "retrieve")
            or require(run_dir, ["validation.bert.jsonl"], "run --split validation --method bert")
            or require(run_dir, ["validation.llm.jsonl"], "run --split validation --method llm --allow-api")
        )
    if command == "run":
        message = require(run_dir, [f"{split}.jsonl"], "prepare") or require(run_dir, ["candidates.jsonl"], "retrieve")
        if message is None and (method == "all" or method.startswith("cascade")):
            message = require(run_dir, ["thresholds.json"], "tune")
        if message is None and method.startswith("cascade"):
            message = require(run_dir, [f"{split}.bert.jsonl"], f"run --split {split} --method bert")
        return message
    if command == "evaluate":
        return require(run_dir, [f"{split}.jsonl"], "prepare") or require(
            run_dir, [f"{split}.{name}.jsonl" for name in methods], f"run --split {split}")
    return None
