import json

import pyarrow as pa
import pyarrow.parquet as pq

from hybrid_reranker import data


def hotpot_row(row_id, level="hard", kind="bridge", broken=False):
    """A HotpotQA-shaped row. A broken row names sentence 9 of a two-sentence paragraph."""
    return {
        "id": row_id, "answer": "an answer", "level": level, "type": kind,
        "context": {"title": ["Alpha", "Beta"], "sentences": [["Alpha one.", "Alpha two."], ["Beta one."]]},
        "supporting_facts": {"title": ["Alpha", "Beta"], "sent_id": [9 if broken else 1, 0]},
    }


def write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


def write_qrels(path, ids):
    path.write_text("query-id\tcorpus-id\tscore\n" + "".join(f"{i}\tdoc-{i}-a\t1\n{i}\tdoc-{i}-b\t1\n" for i in ids))
    return str(path)


def install_sources(monkeypatch, tmp_path):
    validation_rows = [hotpot_row("a1"), hotpot_row("a2", "easy", "comparison"), hotpot_row("a3", broken=True)]
    test_rows = [hotpot_row("b1"), hotpot_row("b2", broken=True), hotpot_row("b3", kind="comparison")]
    files = {
        ("BeIR/hotpotqa", "queries/queries-00000-of-00001.parquet"): write_parquet(
            tmp_path / "queries.parquet", [{"_id": i, "text": f"question {i}"} for i in ("a1", "a2", "a3", "b1", "b2", "b3")]),
        ("BeIR/hotpotqa-qrels", "dev.tsv"): write_qrels(tmp_path / "dev.tsv", ["a1", "a2", "a3"]),
        ("BeIR/hotpotqa-qrels", "test.tsv"): write_qrels(tmp_path / "test.tsv", ["b1", "b2", "b3"]),
        ("hotpotqa/hotpot_qa", "distractor/train-00000-of-00002.parquet"): write_parquet(tmp_path / "t0.parquet", validation_rows),
        ("hotpotqa/hotpot_qa", "distractor/train-00001-of-00002.parquet"): write_parquet(tmp_path / "t1.parquet", [hotpot_row("unused")]),
        ("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet"): write_parquet(tmp_path / "v0.parquet", test_rows),
    }
    monkeypatch.setattr(data, "download", lambda config, repo, filename, revision: files[(repo, filename)])


def test_prepare_skips_questions_whose_supporting_fact_does_not_exist(monkeypatch, tmp_path, config, run_dir, capsys):
    install_sources(monkeypatch, tmp_path)
    config.update(validation_size=10, test_size=10)
    data.prepare(config)
    ids = {split: {json.loads(line)["id"] for line in (run_dir / f"{split}.jsonl").read_text().splitlines()} for split in ("validation", "test")}
    assert ids == {"validation": {"a1", "a2"}, "test": {"b1", "b3"}}
    assert json.loads((run_dir / "skipped_queries.json").read_text()) == {"validation": ["a3"], "test": ["b2"]}
    output = capsys.readouterr().out
    assert "validation: skipped 1 queries" in output and "a3" in output and "test: skipped 1 queries" in output


def test_prepare_keeps_the_skipped_list_when_rerun_from_saved_samples(monkeypatch, tmp_path, config, run_dir):
    install_sources(monkeypatch, tmp_path)
    config.update(validation_size=10, test_size=10)
    data.prepare(config)
    before = (run_dir / "skipped_queries.json").read_text()
    data.prepare(config)
    assert (run_dir / "skipped_queries.json").read_text() == before


def test_kept_facts_carry_the_sentence_text(monkeypatch, tmp_path, config, run_dir):
    install_sources(monkeypatch, tmp_path)
    config.update(validation_size=10, test_size=10)
    data.prepare(config)
    first = next(json.loads(line) for line in (run_dir / "validation.jsonl").read_text().splitlines() if json.loads(line)["id"] == "a1")
    assert first["supporting_facts"] == [
        {"title": "Alpha", "sent_id": 1, "text": "Alpha two."}, {"title": "Beta", "sent_id": 0, "text": "Beta one."},
    ]


def test_usable_facts_rejects_negative_and_unknown_titles():
    row = {"supporting_facts": {"title": ["A"], "sent_id": [-1]}}
    assert data.usable_facts(row, {"A": ["one"]}) is None
    row = {"supporting_facts": {"title": ["Missing"], "sent_id": [0]}}
    assert data.usable_facts(row, {"A": ["one"]}) is None
