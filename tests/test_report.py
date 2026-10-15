import csv
import json

import pytest

from hybrid_reranker.gate import train_gate
from hybrid_reranker.io import load_results, write_jsonl
from hybrid_reranker.report import report
from hybrid_reranker.splits import resplit

pytest.importorskip("matplotlib")


def prepare_report(config, run_dir, offline_run):
    offline_run("validation")
    rows = offline_run("test")
    write_jsonl(run_dir / "test.dense.jsonl", [row["bert"] for row in rows])
    train_gate(config)
    return rows


def test_report_exports_marked_mock_figure_and_both_tables(config, run_dir, offline_run):
    prepare_report(config, run_dir, offline_run)
    resplit(config, repeats=10, calibration_size=25)
    protected = {path: path.read_bytes() for path in run_dir.glob("*.json*")}
    report(config, repeats=100)
    output = run_dir / "report/test"
    saved = json.loads((output / "report.json").read_text())
    assert saved["data_kind"] == "MOCK" and len(saved["table_1"]) == 7
    assert {row["block"] for row in saved["table_2"]} == {"decisions", "stability", "signal"}
    assert (output / "figure_2.png").read_bytes().startswith(b"\x89PNG")
    assert (output / "figure_2.pdf").read_bytes().startswith(b"%PDF")
    for index in (1, 2):
        assert "MOCK" in (output / f"table_{index}.tex").read_text()
        rows = list(csv.DictReader((output / f"table_{index}.csv").open()))
        assert all(row["data_kind"] == "MOCK" for row in rows)
    table = {row["method"]: row for row in saved["table_1"]}
    assert table["gap"]["ndcg@10"] == 1
    assert table["gap"]["difference_vs_llm"] == pytest.approx(.2)
    assert table["oracle"]["llm_fraction"] == table["gap"]["llm_fraction"] == .2
    assert table["gap"]["rerank_tokens"] == 480
    assert table["llm"]["difference_ci95_low"] == table["llm"]["difference_ci95_high"] == 0
    assert all(path.read_bytes() == content for path, content in protected.items())


def test_report_leaves_missing_answers_blank(config, run_dir, offline_run):
    prepare_report(config, run_dir, offline_run)
    for method in ("dense", "bert", "llm"):
        path = run_dir / f"test.{method}.jsonl"
        rows = load_results(path)
        for row in rows.values():
            row.pop("answer")
            row.pop("answer_usage")
        write_jsonl(path, rows.values())
    report(config, repeats=10)
    result = json.loads((run_dir / "report/test/report.json").read_text())
    assert all(row["em"] is None and row["f1"] is None and row["answer_tokens"] is None for row in result["table_1"])


def test_report_requires_fitted_gate_and_dense_results(config, run_dir, offline_run, capsys):
    offline_run("validation")
    offline_run("test")
    report(config)
    assert "rerank gate" in capsys.readouterr().out
    train_gate(config)
    report(config)
    assert "test.dense.jsonl" in capsys.readouterr().out
    assert not (run_dir / "report").exists()


def test_report_retains_generator_failure_counts(config, run_dir, offline_run):
    prepare_report(config, run_dir, offline_run)
    path = run_dir / "test.llm.jsonl"
    rows = load_results(path)
    next(iter(rows.values()))["answer_failed"] = True
    write_jsonl(path, rows.values())
    report(config, repeats=10)
    saved = json.loads((run_dir / "report/test/report.json").read_text())
    llm = next(row for row in saved["table_1"] if row["method"] == "llm")
    assert llm["answer_failed"] == .02 and llm["llm_failed"] == 0
    assert saved["gate_min_gain"] == 0
