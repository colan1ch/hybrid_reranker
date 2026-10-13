import json

import pytest
from conftest import documents, permutation

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from hybrid_reranker.io import load_results, write_jsonl  # noqa: E402
from hybrid_reranker.run import run  # noqa: E402


def prepare_run_dir(run_dir):
    queries = [
        {"id": f"q{i}", "question": f"question {i}?", "answer": "Paris", "qrels": {"d1": 1, "d2": 1},
         "supporting_facts": [{"title": "Title 1", "sent_id": 0, "text": "some words"}]}
        for i in (1, 2)
    ]
    write_jsonl(run_dir / "validation.jsonl", queries)
    write_jsonl(run_dir / "candidates.jsonl", [{"id": q["id"], "documents": documents()} for q in queries])


def behaviour(request):
    if request["model"] == "openai/gpt-oss-20b":  # not endswith("20b"): "gpt-oss-120b" ends that way too
        return "Paris", "stop"
    if "question 1?" in request["messages"][1]["content"]:
        return json.dumps({"ranking": [1, 1, 2]}), "stop"  # never a valid permutation
    return permutation(), "stop"


def test_one_failed_question_no_longer_stops_the_run(config, run_dir, groq):
    prepare_run_dir(run_dir)
    client = groq(behaviour)
    assert run(config, "validation", "llm", answers=True) is True
    rows = load_results(run_dir / "validation.llm.jsonl")
    failed, fine = rows["q1"], rows["q2"]
    assert failed["llm_failed"] is True and failed["llm_error"] == "not a permutation"
    assert failed["ranking"][:3] == ["d1", "d2", "d3"]  # no BERT result here, so the search order is kept
    assert failed["rerank_usage"]["total_tokens"] == 2100
    assert fine["llm_failed"] is False and fine["ranking"][0] == "d20"
    assert failed["answer"] == "Paris" and failed["answer_failed"] is False
    sent = len(client.requests)
    assert run(config, "validation", "llm", answers=True) is True  # finished work is not repeated
    assert len(client.requests) == sent


def test_cascade_without_thresholds_explains_what_to_do(config, run_dir, groq, capsys):
    prepare_run_dir(run_dir)
    groq(behaviour)
    assert run(config, "validation", "cascade_gap") is False
    assert "rerank tune" in capsys.readouterr().out
