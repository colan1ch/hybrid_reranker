import math

from hybrid_reranker.metrics import (
    answer_metrics,
    query_metrics,
    ranking_metrics,
    support_recall,
)


def ranking_with(first, second, size=20):
    ranking = [f"x{i}" for i in range(1, size + 1)]
    ranking[first - 1], ranking[second - 1] = "gold1", "gold2"
    return ranking


QRELS = {"gold1": 1, "gold2": 1}


def test_ndcg_when_gold_texts_are_first_and_third():
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert math.isclose(ranking_metrics(ranking_with(1, 3), QRELS)["ndcg@10"], expected)


def test_perfect_order_scores_one():
    metrics = ranking_metrics(ranking_with(1, 2), QRELS)
    assert metrics["ndcg@10"] == 1 and metrics["mrr@10"] == 1 and metrics["recall@20"] == 1


def test_gold_outside_top_ten_scores_zero_ndcg_but_counts_for_recall_at_twenty():
    metrics = ranking_metrics(ranking_with(11, 14), QRELS)
    assert metrics["ndcg@10"] == 0 and metrics["mrr@10"] == 0 and metrics["recall@20"] == 1


def test_f1_and_em_follow_the_official_rules():
    assert answer_metrics("The Eiffel Tower", "eiffel tower") == {"em": 1.0, "f1": 1.0}
    assert answer_metrics("yes", "no") == {"em": 0.0, "f1": 0.0}
    partial = answer_metrics("Christmas Album", "Elvis' Christmas Album")
    assert partial["em"] == 0.0 and math.isclose(partial["f1"], 0.8)


def test_support_recall_counts_facts_found_in_the_passages_handed_to_the_generator():
    passages = [{"title": "A", "text": "A\nFirst sentence. Second sentence."}]
    facts = [
        {"title": "A", "sent_id": 0, "text": "First sentence."},
        {"title": "A", "sent_id": 1, "text": "Second sentence."},
        {"title": "B", "sent_id": 0, "text": "Other."},
    ]
    assert math.isclose(support_recall(passages, facts), 2 / 3)


def test_failure_flags_default_to_zero_for_results_saved_before_they_existed():
    query = {"qrels": QRELS, "supporting_facts": [], "answer": "x"}
    result = {"ranking": ranking_with(1, 2), "context": [], "llm_called": False, "rerank_usage": {}}
    metrics = query_metrics(query, result)
    assert metrics["llm_failed"] == 0.0
    result.update(llm_called=True, llm_failed=True, answer="x", answer_failed=True,
                  answer_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    metrics = query_metrics(query, result, answers=True)
    assert metrics["llm_failed"] == 1.0 and metrics["answer_failed"] == 1.0


def test_support_recall_without_supporting_facts_is_the_share_of_relevant_documents_in_the_context():
    query = {"qrels": {"d1": 1, "d2": 1, "d3": 0}}
    result = {"ranking": ["d1", "d2"], "context": [{"id": "d1"}, {"id": "d3"}], "llm_called": False, "rerank_usage": {}}
    assert query_metrics(query, result)["support_recall@5"] == 0.5
