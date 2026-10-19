import json

import pytest
from conftest import documents, permutation

from hybrid_reranker.io import read_jsonl
from hybrid_reranker.llm import LLM, parse_ranking, sum_usage


def test_valid_ranking_needs_one_request(config, groq):
    client = groq(lambda request: (permutation(), "stop"))
    ranking, usage, error = LLM(config).rerank("q?", documents())
    assert ranking[:3] == ["d20", "d19", "d18"] and error is None
    assert len(client.requests) == 1 and usage["total_tokens"] == 1050


def test_a_bad_ranking_is_retried_once_with_a_different_request(config, groq):
    answers = iter([(json.dumps({"ranking": [1, 1, 2]}), "stop"), (permutation(), "stop")])
    client = groq(lambda request: next(answers))
    ranking, usage, error = LLM(config).rerank("q?", documents())
    assert ranking is not None and error is None
    first, second = client.requests
    assert second["max_completion_tokens"] > first["max_completion_tokens"]
    assert second["messages"][0]["content"] != first["messages"][0]["content"]
    assert usage["total_tokens"] == 2100  # both attempts are paid for and counted


def test_a_cut_off_ranking_is_retried_with_a_longer_limit(config, groq):
    answers = iter([('{"ranking": [1,2', "length"), (permutation(), "stop")])
    client = groq(lambda request: next(answers))
    ranking, _, error = LLM(config).rerank("q?", documents())
    assert ranking is not None and error is None and len(client.requests) == 2


def test_two_bad_answers_give_a_failure_instead_of_stopping_the_run(config, groq):
    client = groq(lambda request: (json.dumps({"ranking": [1, 1, 2]}), "stop"))
    llm = LLM(config)
    ranking, usage, error = llm.rerank("q?", documents())
    assert ranking is None and error == "not a permutation" and usage["total_tokens"] == 2100
    assert len(client.requests) == 2
    # Asking again is answered from the cache: no new requests and the same failure, not a stall.
    assert llm.rerank("q?", documents()) == (None, usage, "not a permutation")
    assert len(client.requests) == 2


def test_exhausted_local_quota_returns_none_and_sends_nothing(config, groq):
    client = groq(lambda request: (permutation(), "stop"))
    config["requests_per_day"] = 0
    assert LLM(config).rerank("q?", documents()) is None
    assert client.requests == []


def test_answer_retries_when_cut_off_and_gives_empty_answer_when_it_keeps_failing(config, groq):
    answers = iter([("Par", "length"), ("  Paris ", "stop")])
    client = groq(lambda request: next(answers))
    assert LLM(config).answer("Capital?", [{"id": "d1", "title": "T", "text": "x"}])[0] == "Paris"
    assert len(client.requests) == 2
    client = groq(lambda request: ("Par", "length"))
    config["data_dir"] += "-other"
    text, usage, error = LLM(config).answer("Capital?", [{"id": "d1", "title": "T", "text": "x"}])
    assert text == "" and error == "finish_reason=length" and usage["total_tokens"] == 2100


def test_parse_ranking_reasons():
    assert parse_ranking({"finish_reason": "length", "content": ""}, 3) == (None, "finish_reason=length")
    assert parse_ranking({"finish_reason": "stop", "content": "not json"}, 3) == (None, "unreadable response")
    assert parse_ranking({"finish_reason": "stop", "content": json.dumps({"ranking": [1, 2]})}, 3) == (None, "not a permutation")
    assert parse_ranking({"finish_reason": "stop", "content": json.dumps({"ranking": [3, 1, 2]})}, 3) == ([3, 1, 2], None)


def test_sum_usage_ignores_missing_values():
    assert sum_usage({"prompt_tokens": 5, "total_tokens": 7}, {"completion_tokens": None, "total_tokens": 1}) == {
        "prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 8}


def test_quota_counts_only_this_keys_requests(config, groq, monkeypatch, capsys):
    import time

    from hybrid_reranker.io import write_jsonl

    monkeypatch.setenv("GROQ_API_KEY", "mine")
    mine = LLM(config).account
    full = {"model": config["llm_reranker"], "timestamp": time.time(), "reserved_tokens": 0,
            "usage": {"total_tokens": config["tokens_per_day"]}, "content": "", "finish_reason": "stop"}
    write_jsonl(LLM(config).path, [{**full, "key": "merged", "account": "someoneelse"}])
    client = groq(lambda request: (permutation(), "stop"))
    assert LLM(config).rerank("q?", documents())[0] is not None and len(client.requests) == 1
    write_jsonl(LLM(config).path, [{**full, "key": "own", "account": mine}])
    assert LLM(config).rerank("other question?", documents()) is None
    assert "Local quota reached" in capsys.readouterr().out
    assert len(client.requests) == 1


def http_error(kind, status=400, body=None, headers=None):
    import groq
    import httpx

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    if kind == "timeout":
        return groq.APITimeoutError(request=request)
    response = httpx.Response(status, request=request, headers=headers or {})
    return {400: groq.BadRequestError, 429: groq.RateLimitError, 500: groq.InternalServerError,
            401: groq.AuthenticationError}[status]("error", response=response, body=body)


JSON_FAILED = {"error": {"message": "Failed to validate JSON.", "type": "invalid_request_error",
                         "code": "json_validate_failed", "failed_generation": ""}}


def script(steps):
    """A behaviour that plays one step per request: an exception to raise or a (content, finish_reason) pair."""
    steps = iter(steps)

    def behave(request):
        step = next(steps)
        if isinstance(step, Exception):
            raise step
        return step
    return behave


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr("hybrid_reranker.llm.time.sleep", lambda seconds: waited.append(seconds))
    return waited


def test_invalid_json_from_groq_is_a_failed_answer_that_is_retried_once(config, groq, sleeps):
    client = groq(script([http_error("400", 400, JSON_FAILED), (permutation(), "stop")]))
    ranking, usage, error = LLM(config).rerank("q?", documents())
    assert ranking is not None and error is None and len(client.requests) == 2
    assert usage["total_tokens"] > 1050  # the failed attempt is counted as well


def test_invalid_json_twice_falls_back_instead_of_crashing_and_is_remembered(config, groq, sleeps):
    client = groq(script([http_error("400", 400, JSON_FAILED)] * 2))
    llm = LLM(config)
    ranking, _, error = llm.rerank("q?", documents())
    assert ranking is None and error == "finish_reason=error:json_validate_failed"
    assert llm.rerank("q?", documents())[2] == error and len(client.requests) == 2  # served from the cache


def test_output_parse_failed_is_treated_like_invalid_json(config, groq, sleeps):
    body = {"error": {"message": "Failed to parse output", "type": "invalid_request_error",
                      "code": "output_parse_failed", "failed_generation": "We need to rank..."}}
    client = groq(script([http_error("400", 400, body)] * 2))
    ranking, _, error = LLM(config).rerank("q?", documents())
    assert ranking is None and error == "finish_reason=error:output_parse_failed" and len(client.requests) == 2


def test_network_trouble_is_retried_with_pauses_and_not_remembered(config, groq, sleeps):
    client = groq(script([http_error("timeout"), http_error("500", 500), (permutation(), "stop")]))
    llm = LLM(config)
    assert llm.rerank("q?", documents())[0] is not None and len(client.requests) == 3
    assert [wait for wait in sleeps if wait >= 5] == [5, 15]
    assert len(read_jsonl(llm.path)) == 1  # only the answer, not the failed attempts


def test_a_network_that_stays_down_stops_the_run_for_later(config, groq, sleeps, capsys):
    client = groq(script([http_error("timeout")] * 10))
    assert LLM(config).rerank("q?", documents()) is None
    assert len(client.requests) == 6 and "Resume later" in capsys.readouterr().out


def test_a_short_rate_limit_is_waited_out_and_a_long_one_stops_the_run(config, groq, sleeps, capsys):
    short = http_error("429", 429, headers={"retry-after": "7"})
    groq(script([short, (permutation(), "stop")]))
    assert LLM(config).rerank("q?", documents())[0] is not None and 8 in sleeps
    long = http_error("429", 429, headers={"retry-after": "7200"})
    groq(script([long]))
    assert LLM(config).rerank("another question?", documents()) is None
    assert "Resume later" in capsys.readouterr().out


def test_a_setup_problem_is_not_hidden(config, groq, sleeps):
    import groq as groq_module

    groq(script([http_error("401", 401)]))
    with pytest.raises(groq_module.AuthenticationError):
        LLM(config).rerank("q?", documents())
    groq(script([http_error("400", 400, {"error": {"code": "invalid_request", "message": "bad"}})]))
    with pytest.raises(groq_module.BadRequestError):
        LLM(config).rerank("other?", documents())


def test_the_local_daily_guard_can_be_left_to_groq(config, groq, monkeypatch, sleeps):
    config["tokens_per_day"] = 100
    client = groq(lambda request: (permutation(), "stop"))
    assert LLM(config).rerank("q?", documents()) is None and not client.requests
    monkeypatch.setenv("RERANK_LOCAL_QUOTA", "off")
    assert LLM(config).rerank("q?", documents())[0] is not None and len(client.requests) == 1
    groq(script([http_error("429", 429, headers={"retry-after": "7200"})]))
    assert LLM(config).rerank("another?", documents()) is None  # Groq says no: pause, as before


def test_a_configured_instruction_replaces_the_default_but_the_default_is_unchanged(config, groq):
    config.pop("rerank_instruction", None)
    client = groq(lambda request: (permutation(), "stop"))
    LLM(config).rerank("q?", documents())
    assert "multi-hop" in client.requests[0]["messages"][0]["content"]
    config.update(rerank_instruction="Judge the claim.", data_dir=config["data_dir"] + "-other")
    LLM(config).rerank("q?", documents())
    assert client.requests[1]["messages"][0]["content"] == "Judge the claim."
