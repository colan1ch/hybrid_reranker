from hybrid_reranker.estimate import estimate


def test_estimate_accounts_for_separate_daily_and_minute_limits(config, capsys):
    result = estimate(config, answer_tokens=1000)
    first, second = result["models"]
    assert first["calls"] == 400 and first["estimated_tokens"] == 960000 and first["quota_days"] == 5
    assert second["calls"] == 1500 and second["estimated_tokens"] == 1500000 and second["quota_days"] == 8
    assert result["parallel_quota_days"] == 8 and result["estimate_only"]
    assert "not measured" in capsys.readouterr().out


def test_request_limit_and_configured_token_estimate(config):
    config.update(requests_per_day=2, requests_per_minute=30, tokens_per_minute=8000)
    result = estimate(config, queries=20, fraction=.25, answer_queries=0)
    first, second = result["models"]
    assert first["calls"] == 5 and first["quota_days"] == 3
    assert first["minimum_minutes_by_rate"] == 1.5
    assert second["calls"] == 0 and second["quota_days"] == 0
