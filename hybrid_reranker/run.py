import json
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from .io import append_jsonl, load_results, read_jsonl
from .llm import LLM, context


def run(config, split, method, answers=False):
    run_dir = Path(config["run_dir"])
    queries = read_jsonl(run_dir / f"{split}.jsonl")
    candidates = load_results(run_dir / "candidates.jsonl")
    bert = load_results(run_dir / f"{split}.bert.jsonl")
    output = run_dir / f"{split}.{method}.jsonl"
    saved = load_results(output)
    threshold = None
    if method.startswith("cascade"):
        rule = method.removeprefix("cascade_")
        threshold = json.loads((run_dir / "thresholds.json").read_text())[rule]["threshold"]
    llm = LLM(config)
    model = None
    for query in tqdm(queries, desc=f"{split}/{method}"):
        query_id = query["id"]
        if query_id in saved and (not answers or "answer" in saved[query_id]):
            continue
        documents = candidates[query_id]["documents"]
        if query_id in saved:
            row = saved[query_id]
        else:
            row = {"id": query_id, "llm_called": False, "rerank_usage": {}}
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
                    result = llm.rerank(query["question"], documents)
                    if result is None:
                        return False
                    row["ranking"], row["rerank_usage"] = result
                    row["llm_called"] = True
                else:
                    row["ranking"] = bert[query_id]["ranking"]
            row["context"] = context(config, documents, row["ranking"])
            append_jsonl(output, row)
        if answers:
            result = llm.answer(query["question"], row["context"])
            if result is None:
                return False
            if result["finish_reason"] != "stop":
                print("Incomplete answer; inspect data/api.jsonl and increase the output limit.")
                return False
            row["answer"] = result["content"].strip()
            row["answer_usage"] = result["usage"]
            append_jsonl(output, row)
    print(f"Saved {split}/{method}: {output}")
    return True
