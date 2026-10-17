import csv
import json
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


def usable_facts(row, context):
    """Supporting facts of a HotpotQA row, or None if one of them points outside its paragraph.

    A few rows in the source data name a sentence that does not exist (for example sentence 20 of a
    two-sentence paragraph). Such questions cannot be scored on supporting facts, so they are skipped.
    """
    facts = []
    for title, index in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        sentences = context.get(title)
        if sentences is None or not 0 <= index < len(sentences):
            return None
        facts.append({"title": title, "sent_id": index, "text": sentences[index]})
    return facts


def question_rows(config, source, qrels, queries):
    filenames = (
        [f"distractor/train-{i:05d}-of-00002.parquet" for i in range(2)]
        if source == "train" else ["distractor/validation-00000-of-00001.parquet"]
    )
    rows, skipped = [], []
    for filename in filenames:
        source_path = download(config, "hotpotqa/hotpot_qa", filename, "hotpot_revision")
        for row in parquet_rows(source_path):
            if row["id"] not in qrels:
                continue
            context = dict(zip(row["context"]["title"], row["context"]["sentences"]))
            facts = usable_facts(row, context)
            if facts is None:
                skipped.append(row["id"])
                continue
            rows.append({
                "id": row["id"], "question": queries[row["id"]],
                "answer": row["answer"], "level": row["level"], "type": row["type"],
                "qrels": qrels[row["id"]], "supporting_facts": facts,
            })
    return rows, skipped


def prepare_shared(config):
    run_dir = Path(config["run_dir"])
    if any((run_dir / f"{split}.jsonl").exists() for split in ("validation", "test")):
        print("Shared BEIR test sampling needs an empty run_dir; existing question files were kept.")
        return
    path = download(config, "BeIR/hotpotqa", "queries/queries-00000-of-00001.parquet", "beir_revision")
    queries = {row["_id"]: row["text"] for row in parquet_rows(path)}
    rows, skipped = question_rows(config, "validation", load_qrels(config, "test"), queries)
    if len(rows) < config["validation_size"] + config["test_size"]:
        print(f"Only {len(rows)} eligible BEIR test queries; reduce sample sizes or use the full source.")
        return
    validation = sample_queries(rows, config["validation_size"], config["seed"])
    used = {row["id"] for row in validation}
    test = sample_queries([row for row in rows if row["id"] not in used], config["test_size"], config["seed"] + 1)
    counts = {}
    for split, selected in (("validation", validation), ("test", test)):
        write_jsonl(run_dir / f"{split}.jsonl", [{**row, "beir_split": "test"} for row in selected])
        counts[split] = dict(Counter(f"{row['level']}/{row['type']}" for row in selected))
        print(f"{split}: {len(selected)} disjoint queries from BEIR test, {counts[split]}")
    write_json(run_dir / "sample_counts.json", counts)
    write_json(run_dir / "sampling.json", {"sample_source": "beir_test", "seed": config["seed"],
                                         "validation_size": len(validation), "test_size": len(test)})
    if skipped:
        write_json(run_dir / "skipped_queries.json", {"beir_test": skipped})
        print(f"BEIR test: skipped {len(skipped)} queries with broken supporting facts: {', '.join(skipped)}")


def prepare(config):
    if config.get("sample_source", "original") == "beir_test":
        return prepare_shared(config)
    run_dir = Path(config["run_dir"])
    queries_path = download(
        config, "BeIR/hotpotqa", "queries/queries-00000-of-00001.parquet", "beir_revision",
    )
    queries = {row["_id"]: row["text"] for row in parquet_rows(queries_path)}
    counts, skipped = {}, {}
    for split, source in [("validation", "train"), ("test", "validation")]:
        path = run_dir / f"{split}.jsonl"
        if path.exists():
            selected = read_jsonl(path)
        else:
            qrels = load_qrels(config, "dev" if split == "validation" else "test")
            rows, broken = question_rows(config, source, qrels, queries)
            if broken:
                skipped[split] = broken
            selected = sample_queries(rows, config[f"{split}_size"], config["seed"])
            write_jsonl(path, selected)
        counts[split] = dict(Counter(f"{row['level']}/{row['type']}" for row in selected))
        print(f"{split}: {len(selected)} queries, {counts[split]}")
    write_json(run_dir / "sample_counts.json", counts)
    if skipped:
        path = run_dir / "skipped_queries.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        write_json(path, {**previous, **skipped})
        for split, ids in skipped.items():
            print(f"{split}: skipped {len(ids)} queries with broken supporting facts: {', '.join(ids)}")
