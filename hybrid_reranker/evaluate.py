import csv
import math
from pathlib import Path

import numpy as np

from .io import load_results, read_jsonl, write_json, write_jsonl
from .metrics import query_metrics, ranking_metrics


def complete_results(path, queries, answers=False):
    results = load_results(path)
    missing = [query["id"] for query in queries if query["id"] not in results
               or (answers and "answer" not in results[query["id"]])]
    if missing:
        print(f"{path}: {len(missing)} queries missing. Finish the run first.")
        return None
    return results


def write_csv(path, rows):
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tune(config):
    run_dir = Path(config["run_dir"])
    output = run_dir / "thresholds.json"
    if output.exists():
        print(f"Thresholds already frozen: {output}")
        return
    queries = read_jsonl(run_dir / "validation.jsonl")
    if not queries:
        print(f"No validation queries in {run_dir}. Run `rerank prepare` first.")
        return
    bert = complete_results(run_dir / "validation.bert.jsonl", queries)
    llm = complete_results(run_dir / "validation.llm.jsonl", queries)
    if bert is None or llm is None:
        return
    candidates = load_results(run_dir / "candidates.jsonl")
    bert_scores = np.array([ranking_metrics(bert[row["id"]]["ranking"], row["qrels"])["ndcg@10"] for row in queries])
    llm_scores = np.array([ranking_metrics(llm[row["id"]]["ranking"], row["qrels"])["ndcg@10"] for row in queries])
    tokens = np.array([llm[row["id"]]["rerank_usage"]["total_tokens"] for row in queries])
    thresholds = {}
    for rule in ["gap", "cosine"]:
        signals = np.array([
            bert[row["id"]]["gap"] if rule == "gap" else candidates[row["id"]]["documents"][0]["score"]
            for row in queries
        ])
        values = [math.nextafter(float(signals.min()), -math.inf), *sorted(set(signals.tolist()))]
        curve = []
        for threshold in values:
            selected = signals <= threshold
            curve.append({
                "threshold": threshold, "ndcg@10": float(np.where(selected, llm_scores, bert_scores).mean()),
                "llm_fraction": float(selected.mean()), "tokens_per_query": float((tokens * selected).mean()),
            })
        acceptable = [row for row in curve if row["ndcg@10"] >= llm_scores.mean() - config["ndcg_tolerance"]]
        best = min(acceptable, key=lambda row: (row["tokens_per_query"], row["llm_fraction"], -row["ndcg@10"]))
        thresholds[rule] = best
        write_csv(run_dir / f"validation.{rule}.csv", curve)
        print(f"{rule}: threshold={best['threshold']:.6f}, nDCG={best['ndcg@10']:.4f}, LLM={best['llm_fraction']:.1%}")
    write_json(output, {
        **thresholds, "validation_queries": len(queries), "llm_ndcg@10": float(llm_scores.mean()),
        "ndcg_tolerance": config["ndcg_tolerance"],
    })


def evaluate(config, split, methods, answers=False):
    run_dir = Path(config["run_dir"])
    queries = read_jsonl(run_dir / f"{split}.jsonl")
    if not queries:
        print(f"No {split} queries in {run_dir}. Run `rerank prepare` first.")
        return
    kind = "MOCK" if config.get("mock_data", False) or any(row.get("mock", False) for row in queries) else "RESULTS"
    results = {method: complete_results(run_dir / f"{split}.{method}.jsonl", queries, answers) for method in methods}
    if any(rows is None for rows in results.values()):
        return
    measurements = {
        method: [query_metrics(query, results[method][query["id"]], answers) for query in queries]
        for method in methods
    }
    keys = list(measurements[methods[0]][0])
    arrays = {method: np.array([[row[key] for key in keys] for row in rows]) for method, rows in measurements.items()}
    rng = np.random.default_rng(config["seed"])
    indices = rng.integers(len(queries), size=(config["bootstrap_samples"], len(queries)))
    means, differences = [], []
    references = [name for name in ("llm", "bert") if name in arrays]
    for method, values in arrays.items():
        means.append({"data_kind": kind, "method": method, "queries": len(queries), **dict(zip(keys, values.mean(axis=0).tolist()))})
        for reference in references:
            if method == reference:
                continue
            delta = values - arrays[reference]
            interval = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975], axis=0)
            for index, key in enumerate(keys):
                differences.append({
                    "data_kind": kind, "method": method, "reference": reference, "metric": key,
                    "difference": float(delta[:, index].mean()),
                    "ci95_low": float(interval[0, index]), "ci95_high": float(interval[1, index]),
                })
    write_jsonl(run_dir / f"{split}.metrics.jsonl", [
        {"data_kind": kind, "id": query["id"], "method": method, **metrics}
        for method, rows in measurements.items() for query, metrics in zip(queries, rows)
    ])
    write_csv(run_dir / f"{split}.summary.csv", means)
    if differences:
        write_csv(run_dir / f"{split}.bootstrap.csv", differences)
    write_json(run_dir / f"{split}.summary.json", {"data_kind": kind, "means": means, "paired_bootstrap": differences})
    for row in means:
        print(f"{row['method']:16} nDCG={row['ndcg@10']:.4f} Recall={row['recall@20']:.4f} "
              f"LLM={row['llm_fraction']:.1%} failed={row['llm_failed']:.1%} tokens/query={row['rerank_total_tokens']:.1f}")
