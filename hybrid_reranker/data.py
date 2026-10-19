import csv
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from pyarrow.parquet import ParquetFile

from .io import write_json, write_jsonl


def download(config, repo, filename, revision):
    return hf_hub_download(
        repo, filename, repo_type="dataset", revision=config[revision],
        cache_dir=str(Path(config["data_dir"]) / "downloads"),
    )


def parquet_rows(path, batch_size=1024):
    for batch in ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def load_qrels(config, split, repo="BeIR/scifact-qrels"):
    path = download(config, repo, f"{split}.tsv", "qrels_revision")
    qrels = defaultdict(dict)
    with open(path) as file:
        for row in csv.DictReader(file, delimiter="\t"):
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return qrels


# Claims of the seeded BEIR test shuffle that were sampled in the first 200 positions; their 120B answers are cached.
CACHED_CLAIMS = 200


def labelled_claims(config, split):
    """Ids and qrels of the BEIR SciFact claims in `split` that have at least one relevant abstract, sorted by id."""
    qrels = load_qrels(config, split)
    return {query_id: qrels[query_id] for query_id in sorted(qrels) if any(score > 0 for score in qrels[query_id].values())}


def prepare(config):
    """Write validation.jsonl and test.jsonl for SciFact.

    validation: `validation_size` claims drawn from BEIR `train`, used only to choose the threshold and train the gate.
    test: the claims of BEIR `test` (`test_size` of them, all 300 by default); the two samples cannot overlap
    because they come from different BEIR splits.

    A claim has no answer, only the abstracts that support or refute it (`qrels`), so there is no generator step.

    Test order: the claims after the first CACHED_CLAIMS of the seeded shuffle come first, then those 200 (for 300
    claims: 100 new ones, then 200). The 200 are the ones whose 120B answers are already in the shared API cache, so `--part i/2` splits the requests that still cost
    quota evenly between two people (each gets half of the first 100), while the cached ones are free for both.
    """
    run_dir = Path(config["run_dir"])
    if any((run_dir / f"{split}.jsonl").exists() for split in ("validation", "test")):
        print("Existing question files were kept.")
        return
    path = download(config, "BeIR/scifact", "queries/queries-00000-of-00001.parquet", "beir_revision")
    queries = {row["_id"]: row["text"] for row in parquet_rows(path)}
    train, test = (
        {query_id: qrels for query_id, qrels in labelled_claims(config, split).items() if query_id in queries}
        for split in ("train", "test")
    )
    validation_size, test_size = config["validation_size"], config["test_size"]
    if len(train) < validation_size or len(test) < test_size:
        print(f"Only {len(train)} train and {len(test)} test SciFact claims; reduce validation_size and test_size.")
        return
    validation_ids = list(train)
    random.Random(config["seed"] + 1).shuffle(validation_ids)
    shuffled = list(test)
    random.Random(config["seed"]).shuffle(shuffled)
    test_ids = (shuffled[CACHED_CLAIMS:] + shuffled[:CACHED_CLAIMS])[:test_size]
    chosen = {
        "validation": (validation_ids[:validation_size], train, "train"),
        "test": (test_ids, test, "test"),
    }
    for split, (ids, qrels, source) in chosen.items():
        write_jsonl(run_dir / f"{split}.jsonl", [
            {"id": query_id, "question": queries[query_id], "qrels": qrels[query_id], "beir_split": source}
            for query_id in ids
        ])
        print(f"{split}: {len(ids)} SciFact claims from BEIR {source}")
    write_json(run_dir / "sample_counts.json", {split: {"claim": len(ids)} for split, (ids, _, _) in chosen.items()})
    write_json(run_dir / "sampling.json", {"dataset": "scifact", "seed": config["seed"],
                                         "validation_size": validation_size, "test_size": test_size})
