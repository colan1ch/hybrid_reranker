import csv
import json

from hybrid_reranker.evaluate import evaluate, tune
from hybrid_reranker.io import write_jsonl


def ranking_with(first, second):
    ranking = [f"x{i}" for i in range(1, 21)]
    ranking[first - 1], ranking[second - 1] = "gold1", "gold2"
    return ranking


def result(query_id, ranks, called=False, failed=False, tokens=0):
    ranking = ranking_with(*ranks)
    return {
        "id": query_id, "ranking": ranking, "llm_called": called, "llm_failed": failed,
        "rerank_usage": {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens},
        "context": [{"title": "T", "text": "T\nfact"}], "gap": 0.5,
    }


def make_run(run_dir, split="test", count=6):
    queries = [{"id": f"q{i}", "qrels": {"gold1": 1, "gold2": 1}, "supporting_facts": [], "answer": "a", "question": "q"} for i in range(count)]
    write_jsonl(run_dir / f"{split}.jsonl", queries)
    write_jsonl(run_dir / f"{split}.dense.jsonl", [result(q["id"], (5, 9)) for q in queries])
    write_jsonl(run_dir / f"{split}.bert.jsonl", [result(q["id"], (2, 5) if i % 2 else (1, 2)) for i, q in enumerate(queries)])
    write_jsonl(run_dir / f"{split}.llm.jsonl", [result(q["id"], (1, 2), called=True, failed=i == 0, tokens=2400) for i, q in enumerate(queries)])
    return queries


def test_bootstrap_compares_with_both_the_full_llm_and_bert(config, run_dir):
    make_run(run_dir)
    config["bootstrap_samples"] = 200
    evaluate(config, "test", ["dense", "bert", "llm"])
    rows = list(csv.DictReader((run_dir / "test.bootstrap.csv").open()))
    pairs = {(row["method"], row["reference"]) for row in rows}
    assert pairs == {("dense", "llm"), ("dense", "bert"), ("bert", "llm"), ("llm", "bert")}
    ndcg = next(row for row in rows if row["method"] == "bert" and row["reference"] == "llm" and row["metric"] == "ndcg@10")
    assert float(ndcg["difference"]) < 0 and float(ndcg["ci95_low"]) <= float(ndcg["difference"]) <= float(ndcg["ci95_high"])


def test_failures_and_tokens_show_up_in_the_summary(config, run_dir):
    make_run(run_dir)
    config["bootstrap_samples"] = 50
    evaluate(config, "test", ["bert", "llm"])
    means = {row["method"]: row for row in json.loads((run_dir / "test.summary.json").read_text())["means"]}
    assert means["llm"]["llm_failed"] == 1 / 6 and means["bert"]["llm_failed"] == 0
    assert means["llm"]["rerank_total_tokens"] == 2400 and means["llm"]["llm_fraction"] == 1


def test_evaluate_and_tune_without_prepared_questions_explain_instead_of_crashing(config, run_dir, capsys):
    evaluate(config, "test", ["bert"])
    tune(config)
    output = capsys.readouterr().out
    assert output.count("rerank prepare") == 2


def test_tune_freezes_thresholds(config, run_dir):
    queries = make_run(run_dir, split="validation", count=8)
    write_jsonl(run_dir / "candidates.jsonl", [{"id": q["id"], "documents": [{"id": "gold1", "score": 0.5 + i / 100}]} for i, q in enumerate(queries)])
    tune(config)
    saved = json.loads((run_dir / "thresholds.json").read_text())
    assert {"gap", "cosine", "validation_queries"} <= saved.keys() and saved["validation_queries"] == 8
