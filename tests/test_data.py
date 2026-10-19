import random

import pyarrow as pa
import pyarrow.parquet as pq

from hybrid_reranker import data
from hybrid_reranker.io import read_jsonl


def write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


def write_qrels(path, ids):
    path.write_text("query-id\tcorpus-id\tscore\n" + "".join(f"{i}\tdoc-{i}-a\t1\n{i}\tdoc-{i}-b\t0\n" for i in ids))
    return str(path)


def install_scifact(monkeypatch, tmp_path, train=8, test=6):
    train_ids, test_ids = [f"t{i}" for i in range(train)], [f"c{i}" for i in range(test)]
    files = {
        ("BeIR/scifact", "queries/queries-00000-of-00001.parquet"): write_parquet(
            tmp_path / "queries.parquet", [{"_id": i, "text": f"claim {i}"} for i in [*train_ids, *test_ids, "unlabelled"]]),
        ("BeIR/scifact-qrels", "train.tsv"): write_qrels(tmp_path / "train.tsv", train_ids),
        ("BeIR/scifact-qrels", "test.tsv"): write_qrels(tmp_path / "test.tsv", test_ids),
    }
    monkeypatch.setattr(data, "download", lambda config, repo, filename, revision: files[(repo, filename)])


def test_validation_comes_from_train_and_test_from_test_without_answers(monkeypatch, tmp_path, config, run_dir):
    install_scifact(monkeypatch, tmp_path)
    config.update(validation_size=3, test_size=6)
    data.prepare(config)
    validation, test = read_jsonl(run_dir / "validation.jsonl"), read_jsonl(run_dir / "test.jsonl")
    assert (len(validation), len(test)) == (3, 6)
    assert {row["beir_split"] for row in validation} == {"train"} and {row["beir_split"] for row in test} == {"test"}
    assert all(row["id"].startswith("t") for row in validation) and all(row["id"].startswith("c") for row in test)
    assert not {row["id"] for row in validation} & {row["id"] for row in test}
    assert all("answer" not in row and "supporting_facts" not in row and row["qrels"] for row in validation + test)
    assert all(row["question"].startswith("claim ") for row in validation + test)


def test_test_lists_the_claims_without_cached_answers_first(monkeypatch, tmp_path, config, run_dir):
    install_scifact(monkeypatch, tmp_path, test=5)
    monkeypatch.setattr(data, "CACHED_CLAIMS", 3)
    config.update(validation_size=2, test_size=5)
    data.prepare(config)
    ids = [row["id"] for row in read_jsonl(run_dir / "test.jsonl")]
    shuffled = [f"c{i}" for i in range(5)]
    random.Random(config["seed"]).shuffle(shuffled)
    assert ids == shuffled[3:] + shuffled[:3]


def test_samples_are_reproducible_and_a_smaller_validation_is_a_prefix(monkeypatch, tmp_path, config, run_dir):
    install_scifact(monkeypatch, tmp_path)
    config.update(validation_size=5, test_size=6)
    data.prepare(config)
    first = read_jsonl(run_dir / "validation.jsonl")
    for name in ("validation.jsonl", "test.jsonl"):
        (run_dir / name).unlink()
    config.update(validation_size=3)
    data.prepare(config)
    assert read_jsonl(run_dir / "validation.jsonl") == first[:3]


def test_existing_question_files_are_never_overwritten(monkeypatch, tmp_path, config, run_dir, capsys):
    install_scifact(monkeypatch, tmp_path)
    config.update(validation_size=2, test_size=3)
    data.prepare(config)
    before = {path: path.read_bytes() for path in run_dir.glob("*.json*")}
    data.prepare(config)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert "Existing question files were kept" in capsys.readouterr().out


def test_refuses_a_sample_larger_than_the_data(monkeypatch, tmp_path, config, run_dir, capsys):
    install_scifact(monkeypatch, tmp_path, train=3, test=3)
    config.update(validation_size=4, test_size=3)
    data.prepare(config)
    assert "Only 3 train and 3 test" in capsys.readouterr().out and not (run_dir / "test.jsonl").exists()
