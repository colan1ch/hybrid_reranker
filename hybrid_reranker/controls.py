import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .evaluate import complete_results, write_csv
from .io import read_jsonl, write_json
from .metrics import ranking_metrics

BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.0)


def load_pairs(config, split):
    run_dir = Path(config["run_dir"])
    queries = read_jsonl(run_dir / f"{split}.jsonl")
    if not queries:
        print(f"No {split} queries. Run `rerank prepare` first.")
        return None
    bert = complete_results(run_dir / f"{split}.bert.jsonl", queries)
    llm = complete_results(run_dir / f"{split}.llm.jsonl", queries)
    if bert is None or llm is None:
        return None
    rows = []
    for query in queries:
        query_id = query["id"]
        rows.append({
            "id": query_id, "query": query, "bert": bert[query_id], "llm": llm[query_id],
            "bert_ndcg": ranking_metrics(bert[query_id]["ranking"], query["qrels"])["ndcg@10"],
            "llm_ndcg": ranking_metrics(llm[query_id]["ranking"], query["qrels"])["ndcg@10"],
        })
    return rows


def arrays(rows):
    return tuple(np.array([row[key] for row in rows]) for key in ("bert_ndcg", "llm_ndcg"))


def data_kind(config, rows, mock=False):
    return "MOCK" if mock or config.get("mock_data", False) or any(row["query"].get("mock", False) for row in rows) else "RESULTS"


def auc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None


def threshold_curve(signals, bert, llm, tokens):
    values = [math.nextafter(float(min(signals)), -math.inf), *sorted(set(map(float, signals)))]
    curve = []
    for threshold in values:
        selected = signals <= threshold
        curve.append({
            "threshold": threshold, "ndcg@10": float(np.where(selected, llm, bert).mean()),
            "llm_fraction": float(selected.mean()), "tokens_per_query": float((tokens * selected).mean()),
        })
    return curve


def choose_threshold(signals, bert, llm, tokens, tolerance):
    curve = threshold_curve(signals, bert, llm, tokens)
    acceptable = [row for row in curve if row["ndcg@10"] >= llm.mean() - tolerance]
    return min(acceptable, key=lambda row: (row["tokens_per_query"], row["llm_fraction"], -row["ndcg@10"]))


def budget_masks(scores, count):
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(-np.asarray(scores), kind="stable")[:count]] = True
    return selected


def random_control(bert, llm, tokens, count, repeats, seed):
    rng = np.random.default_rng(seed)
    quality, costs = [], []
    inclusion = np.zeros(len(bert))
    for _ in range(repeats):
        selected = np.zeros(len(bert), dtype=bool)
        selected[rng.choice(len(bert), count, replace=False)] = True
        inclusion += selected
        quality.append(float(np.where(selected, llm, bert).mean()))
        costs.append(float((tokens * selected).mean()))
    return {
        "ndcg@10": float(np.mean(quality)), "tokens_per_query": float(np.mean(costs)),
        "selection_p05": float(np.quantile(quality, 0.05)), "selection_p95": float(np.quantile(quality, 0.95)),
        "inclusion": (inclusion / repeats).tolist(),
    }


def decision_table(selected, gains):
    return [
        {"called": called, "effect": effect, "queries": int(np.sum((selected == called) & condition))}
        for called in (True, False)
        for effect, condition in [("helped", gains > 0), ("unchanged", gains == 0), ("harmed", gains < 0)]
    ]


def controls(config, split="test", repeats=1000, mock=False):
    rows = load_pairs(config, split)
    if rows is None:
        return
    run_dir = Path(config["run_dir"])
    threshold_path = run_dir / "thresholds.json"
    if not threshold_path.exists():
        print("Missing thresholds.json. Run `rerank tune` first.")
        return
    bert, llm = arrays(rows)
    gains = llm - bert
    tokens = np.array([row["llm"]["rerank_usage"]["total_tokens"] for row in rows])
    gaps = np.array([row["bert"]["gap"] for row in rows])
    scores = {"gap": -gaps, "oracle": gains}
    gate_path = run_dir / "gate.json"
    if gate_path.exists():
        from .gate import gate_scores
        signal = gate_scores(config, split, rows, oof=split == "validation")
        if signal is None:
            return
        scores["gate"] = -signal
    kind = data_kind(config, rows, mock or (gate_path.exists() and json.loads(gate_path.read_text())["data_kind"] == "MOCK"))
    curves = []
    for budget in BUDGETS:
        count = int(len(rows) * budget)
        for method, signal in scores.items():
            selected = budget_masks(signal, count)
            curves.append({
                "data_kind": kind, "method": method, "budget": budget, "calls": count,
                "llm_fraction": count / len(rows), "ndcg@10": float(np.where(selected, llm, bert).mean()),
                "gain_over_bert": float((gains * selected).mean()),
                "tokens_per_query": float((tokens * selected).mean()),
                "selection_p05": None, "selection_p95": None,
            })
        random = random_control(bert, llm, tokens, count, repeats, config["seed"])
        curves.append({
            "data_kind": kind, "method": "random", "budget": budget, "calls": count,
            "llm_fraction": count / len(rows), **{key: value for key, value in random.items() if key != "inclusion"},
            "gain_over_bert": random["ndcg@10"] - float(bert.mean()),
        })
    threshold = json.loads(threshold_path.read_text())["gap"]["threshold"]
    selected = gaps <= threshold
    count = int(selected.sum())
    random = random_control(bert, llm, tokens, count, repeats, config["seed"])
    oracle = budget_masks(gains, count)
    quality = float(np.where(selected, llm, bert).mean())
    oracle_quality = float(np.where(oracle, llm, bert).mean())
    improvement = float(llm.mean() - bert.mean())
    oracle_gain = oracle_quality - random["ndcg@10"]
    summary = {
        "data_kind": kind, "split": split, "queries": len(rows), "seed": config["seed"], "random_repeats": repeats,
        "bert_ndcg": float(bert.mean()), "llm_ndcg": float(llm.mean()), "auc_gap": auc(gains > 0, -gaps),
        "gate_available": "gate" in scores, "threshold": threshold, "llm_fraction": count / len(rows),
        "rule_ndcg": quality, "random_ndcg": random["ndcg@10"], "oracle_ndcg": oracle_quality,
        "random_inclusion": random["inclusion"],
        "retained_gain": (quality - bert.mean()) / improvement if improvement else None,
        "gain_over_random": quality - random["ndcg@10"],
        "oracle_fraction": (quality - random["ndcg@10"]) / oracle_gain if oracle_gain else None,
        "budget_policy": "exact floor(n * budget) calls; stable input order breaks score ties",
    }
    matrix = [{"data_kind": kind, "method": "gap", **row} for row in decision_table(selected, gains)]
    if "gate" in scores:
        gate = json.loads(gate_path.read_text())
        gate_selected = -scores["gate"] <= gate["threshold"]
        summary["auc_gate"] = auc(gains > gate["min_gain"], scores["gate"])
        summary["gate_min_gain"] = gate["min_gain"]
        summary["gate_fraction"] = float(gate_selected.mean())
        matrix.extend({"data_kind": kind, "method": "gate", **row} for row in decision_table(gate_selected, gains))
    output = run_dir / "controls"
    output.mkdir(exist_ok=True)
    write_csv(output / f"{split}.curves.csv", curves)
    write_csv(output / f"{split}.decisions.csv", matrix)
    write_json(output / f"{split}.json", {**summary, "curves": curves, "decisions": matrix})
    print(f"{kind}: {len(rows)} queries, gap AUC={summary['auc_gap']}, saved {output}")
