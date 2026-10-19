import hashlib
import json
from pathlib import Path

import numpy as np
from pyarrow.parquet import ParquetFile
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .data import download
from .io import read_jsonl, write_jsonl


def retrieve(config):
    run_dir = Path(config["run_dir"])
    output = run_dir / "candidates.jsonl"
    if output.exists():
        print(f"Candidates already saved: {output}")
        return
    queries = read_jsonl(run_dir / "validation.jsonl") + read_jsonl(run_dir / "test.jsonl")
    if not queries:
        print(f"No queries in {run_dir}. Run `rerank prepare` first.")
        return
    corpus_path = download(
        config, f"BeIR/{config.get('dataset', 'hotpotqa')}", "corpus/corpus-00000-of-00001.parquet", "beir_revision",
    )
    model = SentenceTransformer(
        config["retriever"], device=config["device"], revision=config["retriever_revision"],
    )
    model.max_seq_length = config["encoder_max_length"]
    query_vectors = model.encode(
        [config["query_instruction"] + row["question"] for row in queries],
        batch_size=config["batch_size"], normalize_embeddings=True, show_progress_bar=True,
    )
    settings = {key: config[key] for key in [
        "retriever", "retriever_revision", "beir_revision", "encoder_max_length", "corpus_chunk_size",
    ]}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = Path(config["data_dir"]) / "embeddings" / signature
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus = ParquetFile(corpus_path)
    batches = corpus.iter_batches(batch_size=config["corpus_chunk_size"])
    count = config["candidate_count"]
    candidates = [[] for _ in queries]
    total = (corpus.metadata.num_rows + config["corpus_chunk_size"] - 1) // config["corpus_chunk_size"]
    for index, batch in enumerate(tqdm(batches, total=total, desc="Corpus")):
        documents = batch.to_pylist()
        path = cache_dir / f"{index:05d}.npy"
        if path.exists():
            vectors = np.load(path, mmap_mode="r")
        else:
            vectors = model.encode(
                [row["title"] + "\n" + row["text"] for row in documents],
                batch_size=config["batch_size"], normalize_embeddings=True, show_progress_bar=False,
            )
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as file:
                np.save(file, vectors)
            temporary.replace(path)
        scores = query_vectors @ vectors.T
        size = min(count, len(documents))
        indices = np.argpartition(scores, -size, axis=1)[:, -size:]
        for query_index, doc_indices in enumerate(indices):
            for doc_index in doc_indices:
                document = documents[doc_index]
                candidates[query_index].append({
                    "id": document["_id"], "title": document["title"], "text": document["text"],
                    "score": float(scores[query_index, doc_index]),
                })
            candidates[query_index].sort(key=lambda row: (-row["score"], row["id"]))
            candidates[query_index] = candidates[query_index][:count]
    write_jsonl(output, [
        {"id": query["id"], "documents": documents}
        for query, documents in zip(queries, candidates)
    ])
    print(f"Saved top-{count} for {len(queries)} queries: {output}")
