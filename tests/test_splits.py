import json

import numpy as np
import pytest

from hybrid_reranker.controls import arrays, choose_threshold
from hybrid_reranker.splits import resplit


def test_resplits_are_disjoint_reproducible_and_do_not_touch_thresholds(config, run_dir, offline_run):
    offline_run("validation")
    offline_run("test")
    frozen = (run_dir / "thresholds.json").read_bytes()
    resplit(config, repeats=20, calibration_size=25)
    path = run_dir / "controls/resplits.json"
    first = path.read_bytes()
    result = json.loads(first)
    assert result["data_kind"] == "MOCK" and result["check_size"] == 75
    for row in result["splits"]:
        assert len(row["calibration_ids"]) == 25 and len(row["check_ids"]) == 75
        assert not set(row["calibration_ids"]) & set(row["check_ids"])
        assert row["tolerance_violated"] == (-row["difference_vs_llm"] > config["ndcg_tolerance"] + 1e-12)
    resplit(config, repeats=20, calibration_size=25)
    assert first == path.read_bytes() and frozen == (run_dir / "thresholds.json").read_bytes()


def test_resplit_threshold_uses_only_the_calibration_side(config, run_dir, offline_run):
    rows = offline_run()
    resplit(config, repeats=1, calibration_size=20, source="test")
    result = json.loads((run_dir / "controls/resplits.json").read_text())
    row = result["splits"][0]
    chosen = [item for item in rows if item["id"] in row["calibration_ids"]]
    bert, llm = arrays(chosen)
    expected = choose_threshold(np.array([item["bert"]["gap"] for item in chosen]), bert, llm,
                                np.full(len(chosen), 2400), config["ndcg_tolerance"])
    assert row["threshold"] == expected["threshold"]
    values = [item["check_fraction"] for item in result["splits"]]
    assert result["summary"]["check_fraction"]["median"] == pytest.approx(np.median(values))


def test_invalid_calibration_size_does_not_write_results(config, run_dir, offline_run, capsys):
    offline_run()
    resplit(config, calibration_size=50, source="test")
    assert "calibration_size must" in capsys.readouterr().out
    assert not (run_dir / "controls/resplits.json").exists()


def test_single_source_resplit_defaults_to_half_of_the_split(config, run_dir, offline_run):
    offline_run("validation")
    resplit(config, repeats=3, source="validation")
    result = json.loads((run_dir / "controls/resplits.json").read_text())
    assert result["calibration_size"] == 25 and result["check_size"] == 25
