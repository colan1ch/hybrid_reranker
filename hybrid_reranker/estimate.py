import json
import math
from pathlib import Path

from .io import load_results, read_jsonl

# Cascades send the same requests as `bert` and `llm`, so their answers come from the API cache.
ANSWER_METHODS = ("dense", "bert", "llm")
ASSUMED_ANSWER_OUTPUT = 256


def measured_answer_tokens(config):
    """Mean prompt tokens of saved generator contexts plus an assumed output, or None when nothing is saved."""
    import tiktoken

    from .llm import ANSWER_SYSTEM

    run_dir = Path(config["run_dir"])
    questions = {row["id"]: row["question"] for split in ("validation", "test") for row in read_jsonl(run_dir / f"{split}.jsonl")}
    prompts = [
        ANSWER_SYSTEM + json.dumps({"question": questions[query_id], "passages": row["context"]}, ensure_ascii=False)
        for split in ("validation", "test") for method in ANSWER_METHODS
        for query_id, row in load_results(run_dir / f"{split}.{method}.jsonl").items()
        if query_id in questions and "context" in row
    ]
    if not prompts:
        return None
    try:
        encoding = tiktoken.get_encoding("o200k_harmony")
    except Exception as error:  # noqa: BLE001 - the encoding file is downloaded on first use
        print(f"Cannot load the o200k_harmony tokenizer ({error}); using the configured upper bound.")
        return None
    mean = sum(len(encoding.encode(prompt, disallowed_special=())) for prompt in prompts) / len(prompts)
    return math.ceil(mean) + ASSUMED_ANSWER_OUTPUT


def estimate(config, queries=None, answer_queries=None, methods=3, fraction=1.0, rerank_tokens=2400,
             answer_tokens=None):
    queries = queries if queries is not None else config["validation_size"] + config["test_size"]
    answer_queries = answer_queries if answer_queries is not None else config["test_size"]
    if answer_tokens is not None:
        answer_source = "given"
    elif (answer_tokens := measured_answer_tokens(config)) is not None:
        answer_source = f"measured on saved contexts + {ASSUMED_ANSWER_OUTPUT} assumed output tokens"
    else:
        answer_tokens = config["context_count"] * config["context_doc_tokens"] + config["answer_output_tokens"] + 256
        answer_source = "upper bound: every passage at context_doc_tokens; run `rerank run --method bert` to measure"
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
        "rerank_fraction": fraction, "answer_tokens_source": answer_source, "models": models,
        "parallel_quota_days": max(row["quota_days"] for row in models),
        "notes": ["Estimate, not measured API usage. Retries are not included.",
                  "120B defaults to 2400 tokens/call; cascades reuse the cached `llm` requests.",
                  f"Generator calls count only {', '.join(ANSWER_METHODS)}; cascade answers reuse their cached requests.",
                  "Model quotas are separate; parallel_quota_days assumes both models can be scheduled in parallel."],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
