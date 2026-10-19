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
    assert {row["block"] for row in saved["table_2"]} == {"decisions", "stability", "signal", "equal budget", "criterion"}
    gap = saved["success_criteria"]["gap"]
    assert gap["loss"]["value"] == pytest.approx(.2) and gap["calls"]["value"] == pytest.approx(.2)
    assert gap["random"]["ci95_low"] > 0 and gap["passed"]
    equal = [row for row in saved["table_2"] if row["block"] == "equal budget"]
    assert equal[0]["item"] == "nDCG@10 with 10 calls (gap budget)" and equal[0]["gap"] == 1
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
    with (run_dir / "controls/test.cases.csv").open() as file:
        probabilities = [float(row["gate_probability"]) for row in csv.DictReader(file)]
    assert all(0 <= value <= 1 for value in probabilities) and max(probabilities) > .5


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


def test_report_requires_dense_results(config, run_dir, offline_run, capsys):
    offline_run("validation")
    offline_run("test")
    report(config)
    assert "test.dense.jsonl" in capsys.readouterr().out
    assert not (run_dir / "report").exists()


def test_report_without_a_trained_gate_leaves_gate_rows_out(config, run_dir, offline_run, capsys):
    rows = offline_run("test")
    write_jsonl(run_dir / "test.dense.jsonl", [row["bert"] for row in rows])
    report(config, repeats=10)
    assert "No gate.json" in capsys.readouterr().out
    saved = json.loads((run_dir / "report/test/report.json").read_text())
    assert [row["method"] for row in saved["table_1"]] == ["dense", "bert", "llm", "gap", "random", "oracle"]
    assert all(row["gate"] is None for row in saved["table_2"])
    assert (run_dir / "report/test/figure_2.pdf").exists()


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


def unmark(config, run_dir):
    """Real-looking inputs: no MOCK flag in the config or the questions."""
    config["mock_data"] = False
    for split in ("validation", "test"):
        path = run_dir / f"{split}.jsonl"
        write_jsonl(path, [{key: value for key, value in row.items() if key != "mock"} for row in load_results(path).values()])


def test_a_mock_resplit_marks_every_report_file(config, run_dir, offline_run):
    rows = offline_run("validation"), offline_run("test")
    write_jsonl(run_dir / "test.dense.jsonl", [row["bert"] for row in rows[1]])
    unmark(config, run_dir)
    train_gate(config)
    resplit(config, repeats=5, calibration_size=25, mock=True)
    report(config, repeats=10)
    output = run_dir / "report/test"
    for index in (1, 2):
        assert {row["data_kind"] for row in csv.DictReader((output / f"table_{index}.csv").open())} == {"MOCK"}


def test_report_warns_when_a_live_cascade_gap_run_disagrees(config, run_dir, offline_run, capsys):
    rows = prepare_report(config, run_dir, offline_run)
    live = [{**row["bert"], "id": row["id"]} for row in rows]  # gap threshold 1.0 calls the first 10 queries
    write_jsonl(run_dir / "test.cascade_gap.jsonl", live)
    report(config, repeats=10)
    assert json.loads((run_dir / "report/test/report.json").read_text())["live_cascade_gap_mismatches"] == 10
    assert "differs from the offline rebuild on 10 queries" in capsys.readouterr().out


def test_success_criterion_fails_when_the_rule_calls_too_often(config, run_dir, offline_run):
    prepare_report(config, run_dir, offline_run)
    (run_dir / "thresholds.json").write_text(json.dumps({"gap": {"threshold": 3.0}}))  # 30 of 50 queries
    report(config, repeats=10)
    gap = json.loads((run_dir / "report/test/report.json").read_text())["success_criteria"]["gap"]
    assert gap["calls"]["value"] == pytest.approx(.6) and not gap["calls"]["passed"] and not gap["passed"]
    rows = list(csv.DictReader((run_dir / "report/test/table_2.csv").open()))
    assert next(row for row in rows if row["item"] == "LLM call fraction < 0.5")["gap"] == "0.600 no"
