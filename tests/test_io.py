import multiprocessing

from hybrid_reranker.io import append_jsonl, read_jsonl


def write_rows(path, name):
    for index in range(100):
        append_jsonl(path, {"writer": name, "index": index, "text": name * 50000})


def test_two_processes_can_append_to_the_same_file(tmp_path):
    path = tmp_path / "api.jsonl"
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=write_rows, args=(path, name)) for name in ("a", "b")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    rows = read_jsonl(path)
    assert sorted((row["writer"], row["index"]) for row in rows) == [(name, i) for name in "ab" for i in range(100)]
