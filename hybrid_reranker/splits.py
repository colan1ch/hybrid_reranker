from pathlib import Path

import numpy as np

from .controls import arrays, choose_threshold, data_kind, load_pairs
from .evaluate import write_csv
from .io import write_json


def resplit(config, repeats=100, calibration_size=None, source="both", mock=False):
    rows = []
    for split in ("validation", "test") if source == "both" else (source,):
        part = load_pairs(config, split)
        if part is None:
            return
        rows.extend(part)
    if calibration_size is not None:
        size = calibration_size
    else:  # the frozen protocol's size when pooling both splits, half of a single split otherwise
        size = config["validation_size"] if source == "both" else len(rows) // 2
    if not 0 < size < len(rows):
        print(f"calibration_size must be between 1 and {len(rows) - 1}; got {size}.")
        return
    if len({row["id"] for row in rows}) != len(rows):
        print("Repeated query ids in the split pool; use disjoint source samples.")
        return
    bert, llm = arrays(rows)
    gaps = np.array([row["bert"]["gap"] for row in rows])
    tokens = np.array([row["llm"]["rerank_usage"]["total_tokens"] for row in rows])
    rng = np.random.default_rng(config["seed"])
    kind = data_kind(config, rows, mock)
    results = []
    for repeat in range(repeats):
        order = rng.permutation(len(rows))
        calibration, check = order[:size], order[size:]
        best = choose_threshold(gaps[calibration], bert[calibration], llm[calibration],
                                tokens[calibration], config["ndcg_tolerance"])
        selected = gaps[check] <= best["threshold"]
        quality = float(np.where(selected, llm[check], bert[check]).mean())
        loss = float(llm[check].mean() - quality)
        results.append({
            "data_kind": kind, "repeat": repeat, "threshold": best["threshold"],
            "calibration_fraction": best["llm_fraction"], "check_fraction": float(selected.mean()),
            "ndcg@10": quality, "difference_vs_llm": -loss,
            "tolerance_violated": loss > config["ndcg_tolerance"] + 1e-12,
            "calibration_ids": [rows[index]["id"] for index in calibration],
            "check_ids": [rows[index]["id"] for index in check],
        })
    fields = ("threshold", "calibration_fraction", "check_fraction", "difference_vs_llm")
    summary = {
        key: dict(zip(("p05", "median", "p95"), np.quantile([row[key] for row in results], [.05, .5, .95]).tolist()))
        for key in fields
    }
    output = Path(config["run_dir"]) / "controls"
    output.mkdir(exist_ok=True)
    write_csv(output / "resplits.csv", [
        {key: value for key, value in row.items() if not key.endswith("_ids")} for row in results
    ])
    write_json(output / "resplits.json", {
        "data_kind": kind, "source": source, "seed": config["seed"], "repeats": repeats,
        "calibration_size": size, "check_size": len(rows) - size, "summary": summary,
        "violation_fraction": float(np.mean([row["tolerance_violated"] for row in results])),
        "note": "Overlapping repeated splits; p05-p95 are empirical spread, not a confidence interval.",
        "splits": results,
    })
    print(f"{kind}: {repeats} repeated splits ({size}/{len(rows) - size}); saved {output / 'resplits.json'}")
