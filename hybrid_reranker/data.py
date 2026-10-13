import csv
import random
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from huggingface_hub import hf_hub_download
from pyarrow.parquet import ParquetFile

from .io import read_jsonl, write_json, write_jsonl


def download(config, repo, filename, revision):
    return hf_hub_download(
        repo, filename, repo_type="dataset", revision=config[revision],
        cache_dir=str(Path(config["data_dir"]) / "downloads"),
    )


def parquet_rows(path, batch_size=1024):
    for batch in ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def load_qrels(config, split):
    path = download(config, "BeIR/hotpotqa-qrels", f"{split}.tsv", "qrels_revision")
    qrels = defaultdict(dict)
    with open(path) as file:
        for row in csv.DictReader(file, delimiter="\t"):
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return qrels


def sample_queries(rows, size, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["level"], row["type"]].append(row)
    rng = random.Random(seed)
    for key in sorted(groups):
        groups[key].sort(key=lambda row: row["id"])
        rng.shuffle(groups[key])
    selected = [
        row for layer in zip_longest(*(groups[key] for key in sorted(groups)))
        for row in layer if row is not None
    ][:size]
    rng.shuffle(selected)
    return selected


def prepare(config):
    run_dir = Path(config["run_dir"])
    queries_path = download(
        config, "BeIR/hotpotqa", "queries/queries-00000-of-00001.parquet", "beir_revision",
    )
    queries = {row["_id"]: row["text"] for row in parquet_rows(queries_path)}
    counts = {}
    for split, source in [("validation", "train"), ("test", "validation")]:
        path = run_dir / f"{split}.jsonl"
        if path.exists():
            selected = read_jsonl(path)
        else:
            qrels = load_qrels(config, "dev" if split == "validation" else "test")
            filenames = (
                [f"distractor/train-{i:05d}-of-00002.parquet" for i in range(2)]
                if source == "train" else ["distractor/validation-00000-of-00001.parquet"]
            )
            rows = []
            for filename in filenames:
                source_path = download(config, "hotpotqa/hotpot_qa", filename, "hotpot_revision")
                for row in parquet_rows(source_path):
                    if row["id"] not in qrels:
                        continue
                    context = dict(zip(row["context"]["title"], row["context"]["sentences"]))
                    facts = [
                        {"title": title, "sent_id": index, "text": context[title][index]}
                        for title, index in zip(
                            row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"],
                        )
                    ]
                    rows.append({
                        "id": row["id"], "question": queries[row["id"]],
                        "answer": row["answer"], "level": row["level"], "type": row["type"],
                        "qrels": qrels[row["id"]], "supporting_facts": facts,
                    })
            selected = sample_queries(rows, config[f"{split}_size"], config["seed"])
            write_jsonl(path, selected)
        counts[split] = dict(Counter(f"{row['level']}/{row['type']}" for row in selected))
        print(f"{split}: {len(selected)} queries, {counts[split]}")
    write_json(run_dir / "sample_counts.json", counts)
