import json

from conftest import documents, permutation
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
