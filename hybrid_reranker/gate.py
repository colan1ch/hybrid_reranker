import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .controls import arrays, auc, choose_threshold, data_kind, load_pairs
from .io import append_jsonl, load_results, write_json

FEATURES = ("gap_1_2", "gap_1_5", "std_top10")


def features(result):
    scores = np.asarray(result["scores"], dtype=float)
    return [float(scores[0] - scores[1]), float(scores[0] - scores[4]), float(scores[:10].std())]


def new_model(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))


def train_gate(config, min_gain=0.0, mock=False):
    output = Path(config["run_dir"]) / "gate.json"
    if output.exists():
        print(f"Gate already frozen: {output}")
        return
    rows = load_pairs(config, "validation")
    if rows is None:
        return
    bert, llm = arrays(rows)
    labels = (llm - bert) > min_gain
    positives, negatives = int(labels.sum()), int((~labels).sum())
    print(f"Gate labels: {positives}/{len(rows)} positive ({positives / len(rows):.1%}); min_gain={min_gain:g}")
    if positives < 10 or negatives < 5:
        print("WARNING: not training gate; need at least 10 positive and 5 negative validation labels for 5 folds.")
        return
    values = np.array([features(row["bert"]) for row in rows])
    cv = StratifiedKFold(5, shuffle=True, random_state=config["seed"])
    probabilities = np.zeros(len(rows))
    folds = []
    for train, held_out in cv.split(values, labels):
        model = new_model(config["seed"])
        model.fit(values[train], labels[train])
        probabilities[held_out] = model.predict_proba(values[held_out])[:, 1]
        folds.append({"train_ids": [rows[index]["id"] for index in train],
                      "held_out_ids": [rows[index]["id"] for index in held_out]})
    tokens = np.array([row["llm"]["rerank_usage"]["total_tokens"] for row in rows])
    best = choose_threshold(-probabilities, bert, llm, tokens, config["ndcg_tolerance"])
    model = new_model(config["seed"])
    model.fit(values, labels)
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    write_json(output, {
        "data_kind": data_kind(config, rows, mock), "features": list(FEATURES), "min_gain": min_gain,
        "seed": config["seed"], "positive_fraction": positives / len(rows), "positive_count": positives,
        "oof_auc": auc(labels, probabilities), "selection": best, "threshold": best["threshold"],
        "probability_threshold": -best["threshold"], "ndcg_tolerance": config["ndcg_tolerance"],
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
        "training_ids": [row["id"] for row in rows], "folds": folds,
        "oof_probabilities": {row["id"]: float(probability) for row, probability in zip(rows, probabilities)},
    })
    print(f"Saved validation-only gate: OOF AUC={auc(labels, probabilities):.4f}, threshold={best['threshold']:.6f}")


def gate_scores(config, split, rows, oof=False):
    model = json.loads((Path(config["run_dir"]) / "gate.json").read_text())
    if oof:
        return -np.array([model["oof_probabilities"][row["id"]] for row in rows])
    if split == "test" and set(model["training_ids"]) & {row["id"] for row in rows}:
        print("Gate training ids overlap test ids; use disjoint samples.")
        return None
    values = np.array([features(row["bert"]) for row in rows])
    logits = ((values - model["mean"]) / model["scale"]) @ model["coef"] + model["intercept"]
    return -1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def run_gate(config, split, answers=False):
    run_dir = Path(config["run_dir"])
    if not (run_dir / "gate.json").exists():
        print("Missing gate.json. Run `rerank gate` first.")
        return False
    rows = load_pairs(config, split)
    if rows is None:
        return False
    if answers and any("answer" not in row[method] for row in rows for method in ("bert", "llm")):
        print("Saved BERT and LLM answers are required; cascade_gate never calls the API.")
        return False
    scores = gate_scores(config, split, rows, oof=split == "validation")
    if scores is None:
        return False
    model = json.loads((run_dir / "gate.json").read_text())
    output = run_dir / f"{split}.cascade_gate.jsonl"
    saved = load_results(output)
    for row, score in zip(rows, scores):
        if row["id"] in saved and (not answers or "answer" in saved[row["id"]]):
            continue
        called = bool(score <= model["threshold"])
        source = row["llm"] if called else row["bert"]
        result = {key: value for key, value in source.items() if answers or not key.startswith("answer")}
        result.update(llm_called=called, llm_failed=source.get("llm_failed", False) if called else False,
                      rerank_usage=source["rerank_usage"] if called else {}, gate_signal=float(score))
        append_jsonl(output, result)
    print(f"Saved offline {split}/cascade_gate: {output}")
    return True
