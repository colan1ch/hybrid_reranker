import csv
import json

import numpy as np
import pytest

from hybrid_reranker.controls import (
    arrays,
    auc,
    budget_masks,
    choose_threshold,
    controls,
    decision_table,
    random_control,
)


def test_oracle_selects_all_ten_helpful_questions(offline_run):
    rows = offline_run()
    bert, llm = arrays(rows)
    selected = budget_masks(llm - bert, 10)
    assert np.flatnonzero(selected).tolist() == list(range(10))
    assert np.where(selected, llm, bert).mean() == pytest.approx(1)


def test_auc_orientation_ties_and_missing_class():
    labels = np.array([True] * 10 + [False] * 40)
    assert auc(labels, labels.astype(float)) == 1
    assert auc(labels, np.ones(50)) == 0.5
    assert auc(np.zeros(50), np.arange(50)) is None
    rng = np.random.default_rng(42)
    assert auc(rng.integers(2, size=10000), rng.normal(size=10000)) == pytest.approx(0.5, abs=0.02)


def test_random_control_has_exact_budget_and_fixed_seed():
    bert, llm, tokens = np.zeros(50), np.ones(50), np.full(50, 2400)
    first = random_control(bert, llm, tokens, 10, 100, 42)
    assert first == random_control(bert, llm, tokens, 10, 100, 42)
    assert first["ndcg@10"] == pytest.approx(0.2)
    assert first["tokens_per_query"] == 480


def test_decisions_keep_all_six_cells():
    rows = decision_table(np.array([1, 1, 0, 0], dtype=bool), np.array([1, -1, 0, 1]))
    counts = {(row["called"], row["effect"]): row["queries"] for row in rows}
    assert counts == {(True, "helped"): 1, (True, "unchanged"): 0, (True, "harmed"): 1,
                      (False, "helped"): 1, (False, "unchanged"): 1, (False, "harmed"): 0}


def test_controls_write_mock_results_without_touching_inputs(config, run_dir, offline_run):
    offline_run()
    protected = {path: path.read_bytes() for path in run_dir.glob("*.json*")}
    controls(config, repeats=100)
    result = json.loads((run_dir / "controls/test.json").read_text())
    assert result["data_kind"] == "MOCK" and result["auc_gap"] == 1
    assert result["llm_fraction"] == 0.2
    assert len(result["curves"]) == 15 and len(result["decisions"]) == 6
    for budget in (0.10, 0.25, 0.50, 0.75, 1):
        rows = [row for row in result["curves"] if row["budget"] == budget]
        assert len({row["calls"] for row in rows}) == 1
    full = [row["ndcg@10"] for row in result["curves"] if row["budget"] == 1]
    assert full == pytest.approx([result["llm_ndcg"]] * 3)
    assert all(path.read_bytes() == content for path, content in protected.items())


def test_threshold_chooses_tokens_then_fraction_then_quality():
    selected = choose_threshold(np.array([1, 2, 3]), np.array([0., 0., 1.]),
                                np.array([1., 1., 0.]), np.array([100, 200, 300]), 0)
    assert selected["threshold"] == 1 and selected["tokens_per_query"] == pytest.approx(100 / 3)


def test_case_file_puts_decisions_that_lost_quality_first(config, run_dir, offline_run):
    offline_run()
    (run_dir / "thresholds.json").write_text(json.dumps({"gap": {"threshold": 0.5}}))
    controls(config, repeats=10)
    with (run_dir / "controls/test.cases.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 50 and all(row["data_kind"] == "MOCK" for row in rows)
    lost = [row for row in rows if row["gap_lost_quality"] == "True"]
    assert [row["id"] for row in lost] == [f"test-{index:03d}" for index in range(5, 10)]
    assert rows[:5] == lost and all(row["gap_called"] == "False" and row["llm_effect"] == "helped" for row in lost)
