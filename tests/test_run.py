import json

import pytest
from conftest import documents, permutation

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from hybrid_reranker.io import load_results, write_jsonl
from hybrid_reranker.run import run


def prepare_run_dir(run_dir):
    queries = [
        {"id": f"q{i}", "question": f"question {i}?", "answer": "Paris", "qrels": {"d1": 1, "d2": 1},
         "supporting_facts": [{"title": "Title 1", "sent_id": 0, "text": "some words"}]}
        for i in (1, 2)
    ]
    write_jsonl(run_dir / "validation.jsonl", queries)
    write_jsonl(run_dir / "candidates.jsonl", [{"id": q["id"], "documents": documents()} for q in queries])
    bert = [f"d{i}" for i in (2, 1, *range(3, 21))]
    write_jsonl(run_dir / "validation.bert.jsonl", [{"id": q["id"], "ranking": bert} for q in queries])


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
    assert failed["ranking"][:3] == ["d2", "d1", "d3"]  # the BERT order, as in the cascades
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


def test_llm_can_see_the_candidates_in_bert_order(config, run_dir, groq):
    prepare_run_dir(run_dir)
    order = [f"d{i}" for i in range(20, 0, -1)]
    write_jsonl(run_dir / "validation.bert.jsonl", [{"id": f"q{i}", "ranking": order} for i in (1, 2)])
    config["llm_input_order"] = "bert"
    client = groq(lambda request: (permutation(), "stop"))
    assert run(config, "validation", "llm") is True
    passages = json.loads(client.requests[0]["messages"][1]["content"])["passages"]
    assert passages[0]["text"].startswith("Title 20") and passages[-1]["text"].startswith("Title 1\n")
    assert load_results(run_dir / "validation.llm.jsonl")["q1"]["ranking"][0] == "d1"


def test_llm_run_needs_bert_results_first(config, run_dir, groq, capsys):
    prepare_run_dir(run_dir)
    (run_dir / "validation.bert.jsonl").unlink()
    client = groq(lambda request: (permutation(), "stop"))
    assert run(config, "validation", "llm") is False
    assert "--method bert" in capsys.readouterr().out and not client.requests


def test_a_part_processes_only_its_share_of_the_questions(config, run_dir, tokenizer):
    prepare_run_dir(run_dir)
    assert run(config, "validation", "dense", part=(2, 2)) is True
    assert list(load_results(run_dir / "validation.dense.jsonl")) == ["q2"]
