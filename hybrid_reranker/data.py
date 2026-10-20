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


def labelled_claims(config, split):
    """Ids and qrels of the BEIR SciFact claims in `split` that have at least one relevant abstract, sorted by id."""
    qrels = load_qrels(config, split)
    return {query_id: qrels[query_id] for query_id in sorted(qrels) if any(score > 0 for score in qrels[query_id].values())}


def prepare(config):
    """Write validation.jsonl and test.jsonl for SciFact.

    Both samples are disjoint random subsets of the 300 labelled claims of BEIR `test` (seeded shuffle): the first
    `validation_size` claims are for choosing the threshold and training the gate, the next `test_size` are the test.
    Validation and test therefore come from the same distribution, and a smaller validation set is a prefix of a larger one.

    A claim has no answer, only the abstracts that support or refute it (`qrels`), so there is no generator step.
    """
    run_dir = Path(config["run_dir"])
    if any((run_dir / f"{split}.jsonl").exists() for split in ("validation", "test")):
        print("Existing question files were kept.")
        return
    path = download(config, "BeIR/scifact", "queries/queries-00000-of-00001.parquet", "beir_revision")
    queries = {row["_id"]: row["text"] for row in parquet_rows(path)}
    qrels = {query_id: labels for query_id, labels in labelled_claims(config, "test").items() if query_id in queries}
    validation_size, test_size = config["validation_size"], config["test_size"]
    if len(qrels) < validation_size + test_size:
        print(f"Only {len(qrels)} SciFact test claims; reduce validation_size and test_size.")
        return
    ids = list(qrels)
    random.Random(config["seed"]).shuffle(ids)
    chosen = {"validation": ids[:validation_size], "test": ids[validation_size:validation_size + test_size]}
    for split, selected in chosen.items():
        write_jsonl(run_dir / f"{split}.jsonl", [
            {"id": query_id, "question": queries[query_id], "qrels": qrels[query_id], "beir_split": "test"}
            for query_id in selected
        ])
        print(f"{split}: {len(selected)} disjoint SciFact test claims")
    write_json(run_dir / "sample_counts.json", {split: {"claim": len(selected)} for split, selected in chosen.items()})
    write_json(run_dir / "sampling.json", {"dataset": "scifact", "seed": config["seed"],
                                         "validation_size": validation_size, "test_size": test_size})
