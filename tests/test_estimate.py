from hybrid_reranker.estimate import estimate
from hybrid_reranker.io import write_jsonl


def test_estimate_accounts_for_separate_daily_and_minute_limits(config, capsys):
    config.update(validation_size=100, test_size=300)
    result = estimate(config, answer_tokens=1000)
    first, second = result["models"]
    assert first["calls"] == 400 and first["estimated_tokens"] == 960000 and first["quota_days"] == 5
    assert second["calls"] == 900 and second["estimated_tokens"] == 900000 and second["quota_days"] == 5
    assert result["parallel_quota_days"] == 5 and result["estimate_only"]
    assert "not measured" in capsys.readouterr().out


def test_request_limit_and_configured_token_estimate(config):
    config.update(requests_per_day=2, requests_per_minute=30, tokens_per_minute=8000)
    result = estimate(config, queries=20, fraction=.25, answer_queries=0)
    first, second = result["models"]
    assert first["calls"] == 5 and first["quota_days"] == 3
    assert first["minimum_minutes_by_rate"] == 1.5
    assert second["calls"] == 0 and second["quota_days"] == 0
    assert result["answer_tokens_source"].startswith("upper bound")


def test_answer_tokens_are_measured_on_saved_contexts(config, run_dir, tokenizer):
    write_jsonl(run_dir / "test.jsonl", [{"id": "q1", "question": "Who?"}])
    passages = [{"id": "d1", "title": "A", "text": "short text"}]
    write_jsonl(run_dir / "test.bert.jsonl", [{"id": "q1", "context": passages}])
    result = estimate(config, answer_queries=10)
    bound = config["context_count"] * config["context_doc_tokens"] + config["answer_output_tokens"] + 256
    assert result["answer_tokens_source"].startswith("measured")
    assert 256 < result["models"][1]["estimated_tokens_per_call"] < bound
