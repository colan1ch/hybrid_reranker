import json
import math


def estimate(config, queries=None, answer_queries=None, methods=5, fraction=1.0, rerank_tokens=2400, answer_tokens=None):
    queries = queries if queries is not None else config["validation_size"] + config["test_size"]
    answer_queries = answer_queries if answer_queries is not None else config["test_size"]
    if answer_tokens is None:
        answer_tokens = config["context_count"] * config["context_doc_tokens"] + config["answer_output_tokens"] + 256
    models = []
    for model, calls, tokens_per_call in [
        (config["llm_reranker"], int(queries * fraction), rerank_tokens),
        (config["generator"], answer_queries * methods, answer_tokens),
    ]:
        tokens = calls * tokens_per_call
        days = max(math.ceil(calls / config["requests_per_day"]), math.ceil(tokens / config["tokens_per_day"]))
        models.append({
            "model": model, "calls": calls, "estimated_tokens_per_call": tokens_per_call, "estimated_tokens": tokens,
            "quota_days": days, "minimum_minutes_by_rate": max(calls * 60 / config["requests_per_minute"],
                                                                tokens * 60 / config["tokens_per_minute"]) / 60,
        })
    result = {
        "estimate_only": True, "queries": queries, "answer_queries": answer_queries, "answer_methods": methods,
        "rerank_fraction": fraction, "models": models, "parallel_quota_days": max(row["quota_days"] for row in models),
        "notes": ["Estimate, not measured API usage. No cache hits or retries included.",
                  "120B defaults to 2400 tokens/call; generator defaults to a configured context/output estimate.",
                  "Model quotas are separate; parallel_quota_days assumes both models can be scheduled in parallel."],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
