import json
import sys
import tomllib
from pathlib import Path

import pytest

from hybrid_reranker import cli
from hybrid_reranker.io import write_jsonl


@pytest.fixture
def invoke(config, tmp_path, monkeypatch):
    """Run `rerank ...` in-process with the temporary config and return what it printed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RERANK_ALLOW_API", raising=False)
    path = tmp_path / "config.toml"

    def write(**changes):
        values = {**config, **changes}
        path.write_text("".join(f"{key} = {json.dumps(value)}\n" for key, value in values.items()))
        assert tomllib.loads(path.read_text()) == values  # the file must round-trip exactly

    def call(*args, capsys, **changes):
        write(**changes)
        monkeypatch.setattr(sys, "argv", ["rerank", "--config", str(path), *args])
        cli.main()
        return capsys.readouterr().out

    return call


def test_changing_the_device_does_not_block_the_run(invoke, capsys):
    invoke("evaluate", "--split", "test", capsys=capsys)
    output = invoke("evaluate", "--split", "test", capsys=capsys, device="cuda", batch_size=8)
    assert "Config changed" not in output and "differs from the saved config" in output


def test_changing_a_setting_that_affects_results_still_blocks_the_run(invoke, capsys):
    invoke("evaluate", "--split", "test", capsys=capsys)
    assert "Config changed" in invoke("evaluate", "--split", "test", capsys=capsys, seed=7)


@pytest.mark.parametrize("command, step", [
    (("tune",), "prepare"),
    (("evaluate", "--split", "test"), "prepare"),
    (("retrieve",), "prepare"),
    (("run", "--split", "test", "--method", "bert"), "prepare"),
])
def test_commands_run_out_of_order_say_which_step_comes_first(invoke, capsys, command, step):
    output = invoke(*command, capsys=capsys)
    assert f"Run `rerank {step}` first" in output and "Traceback" not in output


def test_run_asks_for_the_next_missing_step(invoke, config, capsys):
    run_dir = Path(config["run_dir"])
    invoke("evaluate", "--split", "test", capsys=capsys)  # creates the run folder
    write_jsonl(run_dir / "test.jsonl", [])
    assert "candidates.jsonl" in invoke("run", "--split", "test", "--method", "dense", capsys=capsys)
    write_jsonl(run_dir / "candidates.jsonl", [])
    assert "thresholds.json" in invoke("run", "--split", "test", "--method", "all", capsys=capsys)
    write_jsonl(run_dir / "validation.jsonl", [])
    assert "validation.bert.jsonl" in invoke("tune", capsys=capsys)


@pytest.mark.parametrize("args", [
    ("--method", "llm"), ("--method", "all"), ("--method", "cascade_gap"), ("--method", "bert", "--answers"),
])
def test_anything_that_may_call_the_api_needs_the_confirmation_flag(invoke, config, capsys, args, monkeypatch):
    run_dir = Path(config["run_dir"])
    invoke("evaluate", "--split", "test", capsys=capsys)
    for name in ("test.jsonl", "candidates.jsonl", "test.bert.jsonl"):
        write_jsonl(run_dir / name, [])
    (run_dir / "thresholds.json").write_text("{}")
    output = invoke("run", "--split", "test", *args, capsys=capsys)
    assert "--allow-api" in output and not (Path(config["data_dir"]) / "api.jsonl").exists()


def test_the_local_models_do_not_need_the_flag(invoke, config, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    run_dir = Path(config["run_dir"])
    invoke("evaluate", "--split", "test", capsys=capsys)
    for name in ("test.jsonl", "candidates.jsonl"):
        write_jsonl(run_dir / name, [])
    output = invoke("run", "--split", "test", "--method", "dense", capsys=capsys)
    assert "--allow-api" not in output and "Saved test/dense" in output


def test_offline_analysis_commands_and_gate_do_not_need_api_confirmation(invoke, config, offline_run, capsys):
    offline_run("validation")
    offline_run("test")
    assert "validation-only gate" in invoke("gate", "--mock", capsys=capsys)
    assert "MOCK" in invoke("controls", "--repeats", "10", capsys=capsys)
    assert "repeated splits" in invoke("resplit", "--repeats", "5", "--calibration-size", "25", capsys=capsys)
    assert "offline test/cascade_gate" in invoke("run", "--method", "cascade_gate", "--answers", capsys=capsys)
    assert not (Path(config["data_dir"]) / "api.jsonl").exists()


def test_estimate_does_not_create_or_change_a_run(invoke, config, capsys):
    output = invoke("estimate", "--answer-queries", "0", capsys=capsys)
    assert '"estimate_only": true' in output
    assert not Path(config["run_dir"]).exists()
