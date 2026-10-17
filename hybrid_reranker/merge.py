import json
import tomllib
from pathlib import Path

from .checks import comparable
from .io import append_jsonl, load_results, read_jsonl

METHODS = ("dense", "bert", "llm", "cascade_gap", "cascade_cosine", "cascade_gate")
FROZEN = ("thresholds.json", "gate.json")


def merge(config, source_run_dir, source_api=None):
    """Add another machine's results for the same questions: rows this run lacks, and answers it lacks."""
    run_dir, source = Path(config["run_dir"]), Path(source_run_dir)
    snapshot = source / "config.toml"
    if not snapshot.exists() or comparable(tomllib.loads(snapshot.read_text())) != comparable(config):
        print(f"{snapshot} is missing or differs from this config; only runs with the same settings can be merged.")
        return False
    for split in ("validation", "test"):
        ours = [row["id"] for row in read_jsonl(run_dir / f"{split}.jsonl")]
        if ours != [row["id"] for row in read_jsonl(source / f"{split}.jsonl")]:
            print(f"{split}.jsonl differs between the two runs; merge needs the same questions.")
            return False
    for split in ("validation", "test"):
        for method in METHODS:
            name = f"{split}.{method}.jsonl"
            theirs = load_results(source / name)
            if not theirs:
                continue
            ours = load_results(run_dir / name)
            added = [row for query_id, row in theirs.items() if query_id not in ours
                     or ("answer" in row and "answer" not in ours[query_id])]
            for row in added:
                append_jsonl(run_dir / name, row)
            print(f"{name}: added {len(added)} rows")
    for name in FROZEN:
        if (source / name).exists():
            if not (run_dir / name).exists():
                (run_dir / name).write_text((source / name).read_text())
                print(f"{name}: copied")
            elif json.loads((source / name).read_text()) != json.loads((run_dir / name).read_text()):
                print(f"WARNING: {name} differs between the runs; kept this run's file.")
    if source_api:
        api = Path(config["data_dir"]) / "api.jsonl"
        known = {row["key"] for row in read_jsonl(api)}
        added = [row for row in read_jsonl(Path(source_api)) if row["key"] not in known]
        for row in added:
            append_jsonl(api, row)
        print(f"api.jsonl: added {len(added)} cached responses")
    return True
