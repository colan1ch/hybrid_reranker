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


def install_scifact(monkeypatch, tmp_path, claims=8):
    ids = [f"c{i}" for i in range(claims)]
    files = {
        ("BeIR/scifact", "queries/queries-00000-of-00001.parquet"): write_parquet(
            tmp_path / "queries.parquet", [{"_id": i, "text": f"claim {i}"} for i in [*ids, "unlabelled"]]),
        ("BeIR/scifact-qrels", "test.tsv"): write_qrels(tmp_path / "test.tsv", ids),
    }
    monkeypatch.setattr(data, "download", lambda config, repo, filename, revision: files[(repo, filename)])
    return ids


def test_validation_and_test_are_disjoint_claims_of_beir_test_without_answers(monkeypatch, tmp_path, config, run_dir):
    install_scifact(monkeypatch, tmp_path)
    config.update(validation_size=3, test_size=5)
    data.prepare(config)
    validation, test = read_jsonl(run_dir / "validation.jsonl"), read_jsonl(run_dir / "test.jsonl")
    assert (len(validation), len(test)) == (3, 5)
    assert not {row["id"] for row in validation} & {row["id"] for row in test}
    assert {row["beir_split"] for row in validation + test} == {"test"} and "unlabelled" not in {row["id"] for row in validation + test}
    assert all("answer" not in row and "supporting_facts" not in row and row["qrels"] for row in validation + test)
    assert all(row["question"].startswith("claim ") for row in validation + test)


def test_samples_are_the_seeded_shuffle_in_order(monkeypatch, tmp_path, config, run_dir):
    ids = install_scifact(monkeypatch, tmp_path)
    config.update(validation_size=3, test_size=5)
    data.prepare(config)
    shuffled = list(ids)
    random.Random(config["seed"]).shuffle(shuffled)
    assert [row["id"] for row in read_jsonl(run_dir / "validation.jsonl")] == shuffled[:3]
    assert [row["id"] for row in read_jsonl(run_dir / "test.jsonl")] == shuffled[3:8]


def test_a_smaller_validation_is_a_prefix_of_a_larger_one(monkeypatch, tmp_path, config, run_dir):
    install_scifact(monkeypatch, tmp_path)
    config.update(validation_size=5, test_size=3)
    data.prepare(config)
    first = read_jsonl(run_dir / "validation.jsonl")
    for name in ("validation.jsonl", "test.jsonl"):
        (run_dir / name).unlink()
    config.update(validation_size=3, test_size=3)
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
    install_scifact(monkeypatch, tmp_path, claims=4)
    config.update(validation_size=2, test_size=3)
    data.prepare(config)
    assert "Only 4 SciFact test claims" in capsys.readouterr().out and not (run_dir / "test.jsonl").exists()
