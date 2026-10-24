import csv
import json
import re
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from .controls import BUDGETS, arrays, budget_masks, controls, data_kind, load_pairs
from .evaluate import complete_results, write_csv
from .gate import gate_scores
from .io import load_results, write_json
from .metrics import ranking_metrics

LABELS = {"dense": "Dense", "bert": "BERT", "llm": "Always LLM", "gap": "Gap rule",
          "gate": "Logistic gate", "random": "Random", "oracle": "Oracle"}
FIELDS = ("ndcg@10", "all_relevant_in_top5", "llm_fraction", "rerank_tokens", "llm_failed")


def measurements(query, result):
    relevant = {doc_id for doc_id, score in query["qrels"].items() if score > 0}
    return [ranking_metrics(result["ranking"], query["qrels"])["ndcg@10"],
            float(bool(relevant) and relevant <= set(result["ranking"][:5])),
            float(result["llm_called"]), result["rerank_usage"].get("total_tokens", 0),
            float(result.get("llm_failed", False))]


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


TEXT = {
    "en": {
        "labels": LABELS, "random_suffix": " (expected)", "frozen": "{}, frozen threshold",
        "bert": "BERT", "always": "Always LLM", "x": "Fraction of queries sent to LLM",
        "y": "Change in nDCG@10 vs BERT", "panels": ("(a)", "(b)"), "helped": "LLM helped", "unchanged": "LLM unchanged",
        "harmed": "LLM harmed", "gap_x": "BERT score gap", "gap_y": "Share of queries in the group",
        "threshold": "threshold", "title": "{split}, n={n}",
    },
    "ru": {
        "labels": {"dense": "Только поиск", "bert": "BERT", "llm": "Всегда LLM", "gap": "Правило по разрыву",
                   "gate": "Мини-модель", "random": "Монетка", "oracle": "Оракул"},
        "random_suffix": " (ожидание)", "frozen": "{}, порог",
        "bert": "BERT", "always": "Всегда LLM", "x": "Доля утверждений, переданных LLM",
        "y": "Прирост nDCG@10 к BERT", "panels": ("а)", "б)"), "helped": "LLM помогла",
        "unchanged": "LLM не изменила", "harmed": "LLM навредила", "gap_x": "Разрыв оценок BERT",
        "gap_y": "Доля утверждений в группе", "threshold": "порог", "title": None,
    },
}


def figure_2(summary, output, kind, split, lang="en", name="figure_2"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    text = TEXT[lang]
    styles = {"gap": ("-", "o"), "gate": ("--", "s"), "random": (":", "^"), "oracle": ("-.", "D")}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        panels = bool(summary.get("gaps")) and len(summary["gaps"]) == len(summary.get("gains", ()))
        if panels:
            fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 3.5), layout="constrained", gridspec_kw={"width_ratios": [2.1, 1]})
        else:
            fig, ax = plt.subplots(figsize=(7.1, 3.8), layout="constrained")
        for method, (style, marker) in styles.items():
            rows = [row for row in summary["plot_curves"] if row["method"] == method]
            if rows:
                marks = [index for index, row in enumerate(rows)
                         if any(abs(row["llm_fraction"] - budget) < 1e-9 for budget in BUDGETS)]
                ax.plot([row["llm_fraction"] for row in rows], [row["gain_over_bert"] for row in rows],
                        color="0.15", linestyle=style, marker=marker, markevery=marks, markerfacecolor="white",
                        linewidth=1.4, markersize=5, label=text["labels"][method] + (text["random_suffix"] if method == "random" else ""))
        ax.axhline(0, color="0.65", linewidth=0.7)
        points = [("gap", summary["llm_fraction"], summary["rule_ndcg"], "o")]
        if summary.get("gate_ndcg") is not None:
            points.append(("gate", summary["gate_fraction"], summary["gate_ndcg"], "s"))
        for method, fraction, quality, marker in points:
            ax.scatter([fraction], [quality - summary["bert_ndcg"]], marker=marker, color="black", s=45, zorder=6,
                       label=text["frozen"].format(text["labels"][method]))
        full_gain = summary["llm_ndcg"] - summary["bert_ndcg"]
        ax.scatter([0, 1], [0, full_gain], marker="*", color="black", s=85, zorder=5)
        ax.annotate(text["bert"], (0, 0), xytext=(6, -14), textcoords="offset points", fontsize=9)
        ax.annotate(text["always"], (1, full_gain), xytext=(-6, 16), textcoords="offset points", ha="right", fontsize=9)
        ax.set(xlim=(-.03, 1.03), xlabel=text["x"], ylabel=text["y"])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=.15)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncols=3 if not panels else 2, frameon=False, fontsize=7.5)
        if panels:
            gaps, gains = np.asarray(summary["gaps"]), np.asarray(summary["gains"])
            top = 8.0  # gaps of 8 or more share the last bin
            edges = np.linspace(0, top, 13)
            groups = (("helped", gains > 0, "-", "0.1"), ("unchanged", gains == 0, "--", "0.45"), ("harmed", gains < 0, ":", "0.1"))
            for name_, mask, style, color in groups:
                if mask.any():
                    weights = np.full(int(mask.sum()), 1 / mask.sum())
                    bx.hist(np.clip(gaps[mask], 0, top - 1e-9), bins=edges, weights=weights, histtype="step", linestyle=style,
                            color=color, linewidth=1.5, label=f"{text[name_]} (n={int(mask.sum())})")
            bx.axvline(summary["threshold"], color="0.5", linewidth=1, linestyle="-.")
            bx.annotate(text["threshold"], (summary["threshold"], bx.get_ylim()[1]), xytext=(3, -9), textcoords="offset points", fontsize=7.5, color="0.35")
            bx.set(xlim=(0, top), xlabel=text["gap_x"], ylabel=text["gap_y"])
            bx.set_xticks([0, 2, 4, 6, 8])
            bx.grid(axis="y", color="0.9", linewidth=0.6)
            bx.spines[["top", "right"]].set_visible(False)
            bx.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncols=1, frameon=False, fontsize=7.5)
            ax.set_title(text["panels"][0], loc="left", fontsize=9, pad=40)
            bx.set_title(text["panels"][1], loc="left", fontsize=9, pad=40)
        title = text["title"]
        if title or kind == "MOCK":
            fig.suptitle(f"{'МАКЕТ / MOCK — ' if kind == 'MOCK' else ''}{title.format(split=split, n=summary['queries']) if title else ''}")
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=220)
        plt.close(fig)


CRITERIA = (
    ("loss", "nDCG@10 minus always-LLM [95%] >= -tolerance"),
    ("calls", "LLM call fraction < 0.5"),
    ("random", "nDCG@10 minus random at the same calls [95%] > 0"),
)


def success_criteria(values, method, indices, tolerance):
    """The success criterion fixed before the runs (RESULTS_PLAN.md, section 0), checked on saved results."""
    quality = values[method][:, 0]
    fraction = float(values[method][:, FIELDS.index("llm_fraction")].mean())
    # Random k-of-n calls include every query with probability k/n, so its expected per-query quality is exact.
    random = values["bert"][:, 0] + fraction * (values["llm"][:, 0] - values["bert"][:, 0])
    result = {}
    for key, delta in (("loss", quality - values["llm"][:, 0]), ("random", quality - random)):
        low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        passed = bool(delta.mean() >= -tolerance) if key == "loss" else bool(low > 0)
        result[key] = {"value": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high), "passed": passed,
                       "text": f"{delta.mean():.3f} [{low:.3f}, {high:.3f}] {'yes' if passed else 'no'}"}
    result["calls"] = {"value": fraction, "passed": fraction < .5, "text": f"{fraction:.3f} {'yes' if fraction < .5 else 'no'}"}
    result["passed"] = all(result[key]["passed"] for key, _ in CRITERIA)
    return result


def ru_number(value, digits=3, sign=False):
    """A number in Russian typography: decimal comma, real minus sign."""
    return (f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}").replace("-", "−").replace(".", ",")


def ru_interval(low, high):
    return f"[{ru_number(low, sign=True)}; {ru_number(high, sign=True)}]"


def ru_text(value):
    """A number-and-interval string written by the English report, rewritten with decimal commas."""
    return re.sub(r"-(?=\d)", "−", str(value).replace(", ", "; ")).replace(".", ",")


RU_METHODS = {"dense": "Только поиск (dense)", "bert": "Только BERT", "llm": "Всегда LLM (120B)",
              "gap": "Каскад: правило по разрыву", "gate": "Каскад: мини-модель"}
RU_DECISIONS = {"called / helped": "Позвали, 120B помогла", "called / unchanged": "Позвали, 120B не изменила",
                "called / harmed": "Позвали, 120B навредила", "skipped / helped": "Не позвали, 120B помогла бы",
                "skipped / unchanged": "Не позвали, 120B не изменила бы", "skipped / harmed": "Не позвали, 120B навредила бы"}
RU_STABILITY = {"threshold": "Порог разрыва: медиана [5%; 95%]", "calibration_fraction": "Доля вызовов на калибровочной части",
                "check_fraction": "Доля вызовов на проверочной части", "difference_vs_llm": "Отставание от «всегда LLM»"}
RU_CRITERIA = {"loss": "Потеря nDCG@10 к «всегда LLM» не больше допуска: разность [95% ДИ]",
               "calls": "Доля вызовов 120B меньше 0,5",
               "random": "Выигрыш у монетки при той же доле вызовов: разность [95% ДИ]"}


def write_ru_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)


def paper_outputs(directory, table, diagnostics, criteria, summary, kind, split):
    """Figure 2 and the two tables as they go into the Russian article: no repeat of what the figure shows.

    Table 1 keeps what Figure 2 cannot show (call share, tokens, confidence intervals of differences, dense); the random
    and oracle rows are lines of the figure. Files use ';' and a decimal comma so they paste into a Word table.
    """
    directory.mkdir(parents=True, exist_ok=True)
    figure_2(summary, directory, kind, split, lang="ru")
    by_method = {row["method"]: row for row in table}
    rows = []
    for method in RU_METHODS:
        if method not in by_method:
            continue
        row = by_method[method]
        versus_llm = "—" if method == "llm" else (
            f"{ru_number(row['difference_vs_llm'], sign=True)} {ru_interval(row['difference_ci95_low'], row['difference_ci95_high'])}")
        versus_random = "—"
        if method in criteria:
            random = criteria[method]["random"]
            versus_random = f"{ru_number(random['value'], sign=True)} {ru_interval(random['ci95_low'], random['ci95_high'])}"
        rows.append([RU_METHODS[method], ru_number(100 * row["llm_fraction"], 1), f"{row['rerank_tokens']:.0f}", versus_llm, versus_random])
    write_ru_csv(directory / "table_1.csv", [
        "Метод", "Доля вызовов 120B, %", "Токенов на утверждение", "Разность nDCG@10 с «всегда LLM» [95% ДИ]",
        "Разность nDCG@10 с монеткой при той же доле вызовов [95% ДИ]"], rows)
    rows = []
    for row in diagnostics:
        block, item, gap, gate = row["block"], row["item"], row["gap"], row["gate"]
        if block == "decisions":
            rows.append(["Решения", RU_DECISIONS.get(item, item), gap, gate])
        elif block == "stability":
            name = item.split(":")[0]
            rows.append(["Устойчивость", RU_STABILITY.get(name, item), ru_text(gap) if gap is not None else None, None])
        elif block == "signal":
            label = ("AUC: польза 120B (разность больше 0)" if item.startswith("AUC") else
                     "Доля утверждений, где 120B помогла" if item.startswith("Positive") else
                     "AUC мини-модели на отложенных частях проверки")
            rows.append(["Сигнал", label, *(ru_number(v) if v is not None else "—" for v in (gap, gate))])
    for key in ("loss", "calls", "random"):
        cells = []
        for method in ("gap", "gate"):
            result = criteria.get(method, {}).get(key)
            if result is None:
                cells.append("—")
                continue
            verdict = "да" if result["passed"] else "нет"
            cells.append(f"{ru_number(100 * result['value'], 1)} %, {verdict}" if key == "calls" else
                         f"{ru_number(result['value'], sign=True)} {ru_interval(result['ci95_low'], result['ci95_high'])}, {verdict}")
        rows.append(["Критерий успеха", RU_CRITERIA[key], *cells])
    write_ru_csv(directory / "table_2.csv", ["Блок", "Показатель", "Правило по разрыву", "Мини-модель"], rows)


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
    min_gain = gate_model["min_gain"] if gate_model else 0.0
    diagnostics.extend([
        {"data_kind": kind, "block": "signal", "item": f"AUC: gap gain>0, gate gain>{min_gain:g}",
         "gap": summary["auc_gap"], "gate": summary.get("auc_gate")},
        {"data_kind": kind, "block": "signal", "item": "Positive label fraction",
         "gap": float((gains > 0).mean()), "gate": float((gains > min_gain).mean()) if gate_model else None},
        {"data_kind": kind, "block": "signal", "item": "OOF AUC on validation", "gap": None,
         "gate": gate_model["oof_auc"] if gate_model else None},
    ])
    criteria = {method: success_criteria(values, method, indices, config["ndcg_tolerance"])
                for method in ("gap", "gate") if method in values}
    for key, item in CRITERIA:
        diagnostics.append({"data_kind": kind, "block": "criterion", "item": item, **{
            method: criteria[method][key]["text"] if method in criteria else None for method in ("gap", "gate")}})
    output = run_dir / "report" / split
    output.mkdir(parents=True, exist_ok=True)
    for row in table + diagnostics:
        row["data_kind"] = kind
    write_csv(output / "table_1.csv", table)
    write_csv(output / "table_2.csv", diagnostics)
    marker = "MOCK. " if kind == "MOCK" else ""
    latex_table(output / "table_1.tex", marker + "Saved ranking results; intervals: paired query bootstrap, 95%. Gap and gate use their own "
                "frozen thresholds, so their call fractions differ.",
                ["Method", "nDCG@10 [95%]", "Delta vs LLM [95%]", "All rel. in top 5", "Call frac.", "120B tok/q", "120B fail"], [
                    [LABELS[row["method"]], f"{row['ndcg@10']:.3f} [{row['ndcg_ci95_low']:.3f}, {row['ndcg_ci95_high']:.3f}]",
                     f"{row['difference_vs_llm']:.3f} [{row['difference_ci95_low']:.3f}, {row['difference_ci95_high']:.3f}]",
                     *[format_number(row[key]) for key in FIELDS[1:]]] for row in table])
    latex_table(output / "table_2.tex", marker + "Routing diagnostics. Stability is empirical p05-p95, not a confidence interval.",
                ["Block", "Item", "Gap rule", "Logistic gate"], [
                    [row["block"], row["item"], format_number(row["gap"]), format_number(row["gate"])] for row in diagnostics])
    write_json(output / "report.json", {
        "data_kind": kind, "split": split, "gate_min_gain": gate_model["min_gain"] if gate_model else None, "table_1": table, "table_2": diagnostics,
        "live_cascade_gap_mismatches": live, "success_criteria": criteria,
        "notes": [("Gap and gate rows are rebuilt from saved bert/llm results; live_cascade_gap_mismatches counts "
                   "queries where a saved cascade_gap run ranked differently (null when there is no such run)."),
                  "Random uses the empirical inclusion weights of the fixed-seed draws at the frozen gap budget.",
                  "Oracle uses exactly the same number of calls as the frozen gap rule.",
                  "Stability describes the saved resplit pool, not independent trials or gate refits.",
                  "Figure 2 shows budget curves; repeated fixed-threshold splits do not define curve confidence bands."],
    })
    figure_2(summary, output, kind, split)
    paper_outputs(output / "ru", table, diagnostics, criteria, summary, kind, split)
    print(f"{kind}: figure 2 and tables 1-2 saved to {output}")
