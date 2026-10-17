import json

import numpy as np
import pytest

from hybrid_reranker import gate
from hybrid_reranker.controls import load_pairs
from hybrid_reranker.io import load_results, write_jsonl


def test_features_use_only_three_bert_score_statistics():
    scores = list(range(20, 0, -1))
    assert gate.FEATURES == ("gap_1_2", "gap_1_5", "std_top10")
    assert gate.features({"scores": scores}) == pytest.approx([1, 4, np.std(scores[:10])])


def test_gate_never_reads_test_during_training(config, run_dir, offline_run, monkeypatch):
    offline_run("validation")
    offline_run("test")
    (run_dir / "test.jsonl").write_text("poisoned test data: not JSON")
    (run_dir / "test.bert.jsonl").unlink()
    original = gate.load_pairs
    seen = []

    def validation_only(settings, split):
        assert split == "validation"
        seen.append(split)
        return original(settings, split)

    monkeypatch.setattr(gate, "load_pairs", validation_only)
    gate.train_gate(config)
    model = json.loads((run_dir / "gate.json").read_text())
    assert seen == ["validation"] and model["data_kind"] == "MOCK"
    assert model["oof_auc"] > 0.9
    assert all(query_id.startswith("validation-") for query_id in model["training_ids"])
    assert set(model["oof_probabilities"]) == set(model["training_ids"])
    held_out = []
    for fold in model["folds"]:
        assert not set(fold["train_ids"]) & set(fold["held_out_ids"])
        assert len(fold["train_ids"]) == 40
        held_out.extend(fold["held_out_ids"])
    assert sorted(held_out) == sorted(model["training_ids"])


def test_rare_positive_labels_warn_and_do_not_train(config, run_dir, offline_run, capsys):
    offline_run("validation")
    gate.train_gate(config, min_gain=1)
    output = capsys.readouterr().out
    assert "0/50 positive (0.0%)" in output and "WARNING" in output
    assert not (run_dir / "gate.json").exists()


def test_offline_gate_reuses_answers_and_usage_and_makes_no_api_calls(config, run_dir, offline_run, monkeypatch):
    offline_run("validation")
    source = offline_run("test")
    gate.train_gate(config)
    from hybrid_reranker import run

    def forbidden(*args, **kwargs):
        pytest.fail("cascade_gate must not construct an API wrapper")
    monkeypatch.setattr(run, "LLM", forbidden)
    assert run.run(config, "test", "cascade_gate", answers=True)
    rows = load_results(run_dir / "test.cascade_gate.jsonl")
    for item in source:
        result = rows[item["id"]]
        expected = item["llm"] if result["llm_called"] else item["bert"]
        assert result["ranking"] == expected["ranking"] and result["answer"] == expected["answer"]
        assert result["rerank_usage"] == expected["rerank_usage"]
    before = (run_dir / "test.cascade_gate.jsonl").read_bytes()
    assert run.run(config, "test", "cascade_gate", answers=True)
    assert before == (run_dir / "test.cascade_gate.jsonl").read_bytes()


def test_validation_gate_uses_oof_predictions(config, run_dir, offline_run):
    rows = offline_run("validation")
    gate.train_gate(config)
    model = json.loads((run_dir / "gate.json").read_text())
    assert gate.run_gate(config, "validation")
    saved = load_results(run_dir / "validation.cascade_gate.jsonl")
    assert [saved[row["id"]]["gate_signal"] for row in rows] == pytest.approx(
        [-model["oof_probabilities"][row["id"]] for row in rows])


def test_training_test_id_overlap_is_rejected(config, run_dir, offline_run, capsys):
    offline_run("validation")
    gate.train_gate(config)
    for name in ("", ".bert", ".llm"):
        (run_dir / f"test{name}.jsonl").write_bytes((run_dir / f"validation{name}.jsonl").read_bytes())
    assert not gate.run_gate(config, "test")
    assert "overlap test ids" in capsys.readouterr().out


def test_missing_saved_answers_do_not_trigger_generation(config, run_dir, offline_run, capsys):
    offline_run("validation")
    offline_run("test")
    gate.train_gate(config)
    rows = load_results(run_dir / "test.llm.jsonl")
    for row in rows.values():
        row.pop("answer")
    write_jsonl(run_dir / "test.llm.jsonl", rows.values())
    assert not gate.run_gate(config, "test", answers=True)
    assert "never calls the API" in capsys.readouterr().out


def test_controls_include_gate_curve_and_auc(config, run_dir, offline_run):
    from hybrid_reranker.controls import controls
    offline_run("validation")
    offline_run("test")
    gate.train_gate(config)
    controls(config, repeats=10)
    result = json.loads((run_dir / "controls/test.json").read_text())
    assert result["gate_available"]
    assert len([row for row in result["curves"] if row["method"] == "gate"]) == 5
    assert all(row["query"]["mock"] for row in load_pairs(config, "test"))
