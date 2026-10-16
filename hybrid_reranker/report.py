import json
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from .controls import BUDGETS, arrays, budget_masks, controls, data_kind, load_pairs
from .evaluate import complete_results, write_csv
from .gate import gate_scores
from .io import load_results, write_json
from .metrics import answer_metrics, ranking_metrics

LABELS = {"dense": "Dense", "bert": "BERT", "llm": "Always LLM", "gap": "Gap rule",
          "gate": "Logistic gate", "random": "Random", "oracle": "Oracle"}
FIELDS = ("ndcg@10", "both_in_top5", "em", "f1", "llm_fraction", "rerank_tokens", "answer_tokens", "llm_failed", "answer_failed")


def measurements(query, result):
    relevant = {doc_id for doc_id, score in query["qrels"].items() if score > 0}
    answer = answer_metrics(result["answer"], query["answer"]) if "answer" in result else {"em": np.nan, "f1": np.nan}
    return [ranking_metrics(result["ranking"], query["qrels"])["ndcg@10"],
            float(bool(relevant) and relevant <= set(result["ranking"][:5])), answer["em"], answer["f1"],
            float(result["llm_called"]), result["rerank_usage"].get("total_tokens", 0),
            result.get("answer_usage", {}).get("total_tokens", np.nan), float(result.get("llm_failed", False)),
            float(result.get("answer_failed", False)) if "answer" in result else np.nan]


def format_number(value):
    return "--" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def latex_table(path, caption, headers, rows):
    def escape(value):
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace(
            ">", r"\textgreater{}").replace("<", r"\textless{}")

    lines = [r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\caption{" + escape(caption) + "}",
             r"\begin{tabular}{" + "l" * len(headers) + "}", r"\hline",
             " & ".join(map(escape, headers)) + r" \\", r"\hline"]
    lines.extend(" & ".join(escape(value) for value in row) + r" \\" for row in rows)
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n")


def figure_2(summary, output, kind, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    styles = {"gap": ("-", "o"), "gate": ("--", "s"), "random": (":", "^"), "oracle": ("-.", "D")}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(7.1, 3.8), layout="constrained")
        for method, (style, marker) in styles.items():
            rows = [row for row in summary["plot_curves"] if row["method"] == method]
            if rows:
                marks = [index for index, row in enumerate(rows)
                         if any(abs(row["llm_fraction"] - budget) < 1e-9 for budget in BUDGETS)]
                ax.plot([row["llm_fraction"] for row in rows], [row["gain_over_bert"] for row in rows],
                        color="0.15", linestyle=style, marker=marker, markevery=marks, markerfacecolor="white",
                        linewidth=1.4, markersize=5, label=LABELS[method] + (" (expected)" if method == "random" else ""))
        ax.axhline(0, color="0.65", linewidth=0.7)
        points = [("gap", summary["llm_fraction"], summary["rule_ndcg"], "o")]
        if summary.get("gate_ndcg") is not None:
            points.append(("gate", summary["gate_fraction"], summary["gate_ndcg"], "s"))
        for method, fraction, quality, marker in points:
            ax.scatter([fraction], [quality - summary["bert_ndcg"]], marker=marker, color="black", s=45, zorder=6,
                       label=f"{LABELS[method]}, frozen threshold")
        full_gain = summary["llm_ndcg"] - summary["bert_ndcg"]
        ax.scatter([0, 1], [0, full_gain], marker="*", color="black", s=85, zorder=5)
        ax.annotate("BERT", (0, 0), xytext=(6, -14), textcoords="offset points", fontsize=9)
        ax.annotate("Always LLM", (1, full_gain), xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=9)
        ax.set(xlim=(-.03, 1.03), xlabel="Fraction of queries sent to LLM", ylabel="Change in nDCG@10 vs BERT")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=.15)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncols=3, frameon=False, fontsize=8)
        fig.suptitle(f"{'МАКЕТ / MOCK — ' if kind == 'MOCK' else ''}{split}, n={summary['queries']}")
        for extension in ("png", "pdf"):
            fig.savefig(output / f"figure_2.{extension}", dpi=220)
        plt.close(fig)


def live_gap_mismatches(run_dir, split, rows, gap_selected):
    """Queries where a saved live cascade_gap run disagrees with the offline rebuild, or None without such a run."""
    live = load_results(run_dir / f"{split}.cascade_gap.jsonl")
    if not live:
        return None
    mismatches = [row["id"] for row, called in zip(rows, gap_selected)
                  if row["id"] not in live or live[row["id"]]["ranking"] != row["llm" if called else "bert"]["ranking"]]
    if mismatches:
        print(f"WARNING: saved {split}.cascade_gap differs from the offline rebuild on {len(mismatches)} queries "
              f"(e.g. {mismatches[0]}); the tables use the rebuild from {split}.llm.jsonl.")
    return len(mismatches)


def report(config, split="test", repeats=1000, mock=False):
    if find_spec("matplotlib") is None:
        print('Report plots need matplotlib. Install with `pip install -e ".[plots]"`.')
        return
    run_dir = Path(config["run_dir"])
    rows = load_pairs(config, split)
    if rows is None:
        return
    dense = complete_results(run_dir / f"{split}.dense.jsonl", [row["query"] for row in rows])
    if dense is None:
        return
    controls(config, split, repeats, mock)
    control_path = run_dir / "controls" / f"{split}.json"
    if not control_path.exists():
        return
    summary = json.loads(control_path.read_text())
    kind = data_kind(config, rows, mock)
    bert, llm = arrays(rows)
    gains = llm - bert
    gaps = np.array([row["bert"]["gap"] for row in rows])
    gap_selected = gaps <= summary["threshold"]
    gate_model, signals = None, None
    if (run_dir / "gate.json").exists():
        gate_model = json.loads((run_dir / "gate.json").read_text())
        signals = gate_scores(config, split, rows, oof=split == "validation")
        if signals is None:
            return
    else:
        print("No gate.json (the gate was not trained); gate rows and the gate curve are left out.")
    if "MOCK" in (gate_model and gate_model["data_kind"], summary["data_kind"]):
        kind = "MOCK"
    live = live_gap_mismatches(run_dir, split, rows, gap_selected)
    weights = {"gap": gap_selected, "random": np.array(summary["random_inclusion"]),
               "oracle": budget_masks(gains, int(gap_selected.sum()))}
    if gate_model:
        weights["gate"] = signals <= gate_model["threshold"]
    values = {
        method: np.array([measurements(row["query"], row[method]) for row in rows]) for method in ("bert", "llm")
    }
    values["dense"] = np.array([measurements(row["query"], dense[row["id"]]) for row in rows])
    for method, weight in weights.items():
        values[method] = np.where(weight[:, None] == 0, values["bert"], np.where(
            weight[:, None] == 1, values["llm"], values["bert"] * (1 - weight[:, None]) + values["llm"] * weight[:, None]))
    rng = np.random.default_rng(config["seed"])
    indices = rng.integers(len(rows), size=(config["bootstrap_samples"], len(rows)))
    table = []
    for method in [name for name in ("dense", "bert", "llm", "gap", "gate", "random", "oracle") if name in values]:
        measured = values[method]
        means = measured.mean(axis=0)
        quality = measured[:, 0]
        delta = quality - values["llm"][:, 0]
        low, high = np.quantile(quality[indices].mean(axis=1), [.025, .975])
        delta_low, delta_high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        table.append({
            "data_kind": kind, "method": method, "queries": len(rows),
            **{key: float(value) if np.isfinite(value) else None for key, value in zip(FIELDS, means)},
            "ndcg_ci95_low": float(low), "ndcg_ci95_high": float(high), "difference_vs_llm": float(delta.mean()),
            "difference_ci95_low": float(delta_low), "difference_ci95_high": float(delta_high),
        })
    diagnostics = [
        {"data_kind": kind, "block": "decisions", "item": f"{'called' if called else 'skipped'} / {effect}", **{
            method: next((row["queries"] for row in summary["decisions"]
                          if row["method"] == method and row["called"] == called and row["effect"] == effect), None)
            for method in ("gap", "gate")}}
        for called in (True, False) for effect in ("helped", "unchanged", "harmed")
    ]
    resplits_path = run_dir / "controls/resplits.json"
    if resplits_path.exists():
        resplits = json.loads(resplits_path.read_text())
        if resplits["data_kind"] == "MOCK":
            kind = "MOCK"
        for name, statistics in resplits["summary"].items():
            diagnostics.append({"data_kind": kind, "block": "stability", "item": f"{name}: median [p05, p95]",
                                "gap": f"{statistics['median']:.4f} [{statistics['p05']:.4f}, {statistics['p95']:.4f}]", "gate": None})
    else:
        diagnostics.append({"data_kind": kind, "block": "stability", "item": "Not computed; run rerank resplit",
                            "gap": None, "gate": None})
    budgets = [("gap", int(gap_selected.sum()))] + ([("gate", int(weights["gate"].sum()))] if gate_model else [])
    for name, count in budgets:
        diagnostics.append({
            "data_kind": kind, "block": "equal budget", "item": f"nDCG@10 with {count} calls ({name} budget)",
            "gap": float(np.where(budget_masks(-gaps, count), llm, bert).mean()),
            "gate": float(np.where(budget_masks(-signals, count), llm, bert).mean()) if gate_model else None,
        })
    min_gain = gate_model["min_gain"] if gate_model else 0.0
    diagnostics.extend([
        {"data_kind": kind, "block": "signal", "item": f"AUC: gap gain>0, gate gain>{min_gain:g}",
         "gap": summary["auc_gap"], "gate": summary.get("auc_gate")},
        {"data_kind": kind, "block": "signal", "item": "Positive label fraction",
         "gap": float((gains > 0).mean()), "gate": float((gains > min_gain).mean()) if gate_model else None},
        {"data_kind": kind, "block": "signal", "item": "OOF AUC on validation", "gap": None,
         "gate": gate_model["oof_auc"] if gate_model else None},
    ])
    output = run_dir / "report" / split
    output.mkdir(parents=True, exist_ok=True)
    for row in table + diagnostics:
        row["data_kind"] = kind
    write_csv(output / "table_1.csv", table)
    write_csv(output / "table_2.csv", diagnostics)
    marker = "MOCK. " if kind == "MOCK" else ""
    latex_table(output / "table_1.tex", marker + "Saved ranking results; intervals: paired query bootstrap, 95%. Gap and gate use their own "
                "frozen thresholds, so their call fractions differ; Table 2 compares them at equal budgets.",
                ["Method", "nDCG@10 [95%]", "Delta vs LLM [95%]", "Both@5", "EM", "F1", "Call frac.", "120B tok/q", "20B tok/q", "120B fail", "20B fail"], [
                    [LABELS[row["method"]], f"{row['ndcg@10']:.3f} [{row['ndcg_ci95_low']:.3f}, {row['ndcg_ci95_high']:.3f}]",
                     f"{row['difference_vs_llm']:.3f} [{row['difference_ci95_low']:.3f}, {row['difference_ci95_high']:.3f}]",
                     *[format_number(row[key]) for key in FIELDS[1:]]] for row in table])
    latex_table(output / "table_2.tex", marker + "Routing diagnostics. Stability is empirical p05-p95, not a confidence interval.",
                ["Block", "Item", "Gap rule", "Logistic gate"], [
                    [row["block"], row["item"], format_number(row["gap"]), format_number(row["gate"])] for row in diagnostics])
    write_json(output / "report.json", {
        "data_kind": kind, "split": split, "gate_min_gain": gate_model["min_gain"] if gate_model else None, "table_1": table, "table_2": diagnostics,
        "live_cascade_gap_mismatches": live,
        "notes": ["Missing saved answers are left blank, never generated or estimated.",
                  ("Gap and gate rows are rebuilt from saved bert/llm results; live_cascade_gap_mismatches counts "
                   "queries where a saved cascade_gap run ranked differently (null when there is no such run)."),
                  "Random uses the empirical inclusion weights of the fixed-seed draws at the frozen gap budget.",
                  "Oracle uses exactly the same number of calls as the frozen gap rule.",
                  "Stability describes the saved resplit pool, not independent trials or gate refits.",
                  "Figure 2 shows budget curves; repeated fixed-threshold splits do not define curve confidence bands."],
    })
    figure_2(summary, output, kind, split)
    print(f"{kind}: figure 2 and tables 1-2 saved to {output}")
