import json
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from .io import append_jsonl, load_results, read_jsonl
from .llm import LLM, context

# A long unbroken run of failed answers points at a systematic problem; stop before spending the day's quota on it.
MAX_FAILURES_IN_A_ROW = 8


def run(config, split, method, answers=False, part=None):
    """Returns True when finished, None when it stopped to resume later (quota or connection), False on a problem
    that needs a person.

    `part=(index, count)` processes every count-th question starting at index (1-based), so two machines
    with their own API keys can share a split; `rerank merge` joins the results afterwards."""
    if method == "cascade_gate":
        from .gate import run_gate
        return run_gate(config, split, answers)
    run_dir = Path(config["run_dir"])
    queries = read_jsonl(run_dir / f"{split}.jsonl")
    if part:
        index, count = part
        queries = [query for position, query in enumerate(queries) if position % count == index - 1]
        print(f"Part {index}/{count}: {len(queries)} {split} queries.")
    candidates = load_results(run_dir / "candidates.jsonl")
    bert = load_results(run_dir / f"{split}.bert.jsonl")
    output = run_dir / f"{split}.{method}.jsonl"
    saved = load_results(output)
    threshold = None
    input_order = config.get("llm_input_order", "dense")
    if input_order not in ("dense", "bert"):
        print(f"llm_input_order must be \"dense\" or \"bert\"; got {input_order!r}.")
        return False
    if method == "llm":
        # A failed LLM call falls back to the BERT order, as in the cascades, so BERT must come first.
        absent = [query["id"] for query in queries if query["id"] not in bert]
        if absent:
            print(f"{len(absent)} {split} queries have no BERT result (the fallback order for failed LLM calls"
                  f"{' and the input order' if input_order == 'bert' else ''}). "
                  f"Run `rerank run --split {split} --method bert` first.")
            return False
    if method.startswith("cascade"):
        rule = method.removeprefix("cascade_")
        if not (run_dir / "thresholds.json").exists():
            print("Missing thresholds.json. Run `rerank tune` first.")
            return False
        absent = [query["id"] for query in queries if query["id"] not in bert]
        if absent:
            print(f"{len(absent)} {split} queries have no BERT result. Run `rerank run --split {split} --method bert` first.")
            return False
        threshold = json.loads((run_dir / "thresholds.json").read_text())[rule]["threshold"]
    llm = LLM(config)
    model = None
    streak = 0
    for query in tqdm(queries, desc=f"{split}/{method}"):
        query_id = query["id"]
        if query_id in saved and (not answers or "answer" in saved[query_id]):
            continue
        documents = candidates[query_id]["documents"]
        if query_id in saved:
            row = saved[query_id]
        else:
            row = {"id": query_id, "llm_called": False, "llm_failed": False, "rerank_usage": {}}
            if method == "dense":
                row["ranking"] = [doc["id"] for doc in documents]
            elif method == "bert":
                if model is None:
                    model = CrossEncoder(
                        config["reranker"], device=config["device"],
                        max_length=config["encoder_max_length"], activation_fn=torch.nn.Identity(),
                        revision=config["reranker_revision"],
                    )
                scores = model.predict(
                    [(query["question"], doc["title"] + "\n" + doc["text"]) for doc in documents],
                    batch_size=config["batch_size"], show_progress_bar=False,
                )
                order = sorted(range(len(documents)), key=lambda index: -float(scores[index]))
                row["ranking"] = [documents[index]["id"] for index in order]
                row["scores"] = [float(scores[index]) for index in order]
                row["gap"] = row["scores"][0] - row["scores"][1]
            else:
                call_llm = method == "llm"
                if threshold is not None:
                    signal = bert[query_id]["gap"] if rule == "gap" else documents[0]["score"]
                    call_llm = signal <= threshold
                if call_llm:
                    shown = documents
                    if input_order == "bert":
                        by_id = {doc["id"]: doc for doc in documents}
                        shown = [by_id[doc_id] for doc_id in bert[query_id]["ranking"]]
                    result = llm.rerank(query["question"], shown)
                    if result is None:
                        return None  # quota or connection: not finished, resume later
                    ranking, row["rerank_usage"], error = result
                    row["llm_called"] = True
                    if ranking is None:
                        # The LLM gave no usable ranking even after one retry: keep the cheaper order and record it.
                        row["llm_failed"], row["llm_error"] = True, error
                        ranking = bert.get(query_id, {}).get("ranking") or [doc["id"] for doc in documents]
                    row["ranking"] = ranking
                else:
                    row["ranking"] = bert[query_id]["ranking"]
            row["context"] = context(config, documents, row["ranking"])
            append_jsonl(output, row)
        if answers:
            result = llm.answer(query["question"], row["context"])
            if result is None:
                return None  # quota or connection: not finished, resume later
            row["answer"], row["answer_usage"], error = result
            row["answer_failed"] = error is not None
            if error:
                row["answer_error"] = error
            append_jsonl(output, row)
        streak = streak + 1 if row.get("llm_failed") or row.get("answer_failed") else 0
        if streak >= MAX_FAILURES_IN_A_ROW:
            print(f"{streak} failed answers in a row ({row.get('llm_error') or row.get('answer_error')}); stopping. "
                  "Check the error, then run the same command again.")
            return False
    print(f"Saved {split}/{method}: {output}")
    return True
