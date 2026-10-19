import html
import math
import re
import string
from collections import Counter


def ranking_metrics(ranking, qrels):
    relevant = {doc_id: score for doc_id, score in qrels.items() if score > 0}
    dcg = sum(relevant.get(doc_id, 0) / math.log2(index + 2) for index, doc_id in enumerate(ranking[:10]))
    ideal = sum(score / math.log2(index + 2) for index, score in enumerate(sorted(relevant.values(), reverse=True)[:10]))
    reciprocal = next((1 / index for index, doc_id in enumerate(ranking[:10], 1) if doc_id in relevant), 0)
    return {
        "ndcg@10": dcg / ideal if ideal else 0,
        "recall@20": len(set(ranking[:20]) & relevant.keys()) / len(relevant) if relevant else 0,
        "mrr@10": reciprocal,
    }


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_metrics(prediction, answer):
    prediction, answer = normalize_answer(prediction), normalize_answer(answer)
    em = float(prediction == answer)
    special = {"yes", "no", "noanswer"}
    if prediction != answer and (prediction in special or answer in special):
        return {"em": em, "f1": 0.0}
    predicted, gold = prediction.split(), answer.split()
    common = sum((Counter(predicted) & Counter(gold)).values())
    f1 = 2 * common / (len(predicted) + len(gold)) if common else 0.0
    return {"em": em, "f1": f1}


def support_recall(passages, facts):
    def normalize(text):
        return " ".join(html.unescape(text).split())

    found = sum(any(
        normalize(fact["title"]) == normalize(passage["title"])
        and normalize(fact["text"]) in normalize(passage["text"])
        for passage in passages
    ) for fact in facts)
    return found / len(facts) if facts else 0.0


def query_metrics(query, result, answers=False):
    metrics = ranking_metrics(result["ranking"], query["qrels"])
    if "supporting_facts" in query:
        metrics["support_recall@5"] = support_recall(result["context"], query["supporting_facts"])
    else:  # datasets whose labels are whole documents (SciFact): the share of relevant documents in the context
        relevant = {doc_id for doc_id, score in query["qrels"].items() if score > 0}
        metrics["support_recall@5"] = len(relevant & {passage["id"] for passage in result["context"]}) / len(relevant)
    metrics["llm_fraction"] = float(result["llm_called"])
    metrics["llm_failed"] = float(result.get("llm_failed", False))
    usage = result["rerank_usage"]
    metrics["rerank_input_tokens"] = usage.get("prompt_tokens", 0)
    metrics["rerank_output_tokens"] = usage.get("completion_tokens", 0)
    metrics["rerank_total_tokens"] = usage.get("total_tokens", 0)
    if answers:
        metrics.update(answer_metrics(result["answer"], query["answer"]))
        metrics["answer_failed"] = float(result.get("answer_failed", False))
        usage = result["answer_usage"]
        metrics["answer_input_tokens"] = usage["prompt_tokens"]
        metrics["answer_output_tokens"] = usage["completion_tokens"]
        metrics["answer_total_tokens"] = usage["total_tokens"]
    return metrics
