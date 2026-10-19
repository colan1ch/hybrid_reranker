import shutil

from hybrid_reranker.io import append_jsonl, load_results, read_jsonl, write_jsonl
from hybrid_reranker.merge import merge


def setup_pair(tmp_path, config):
    ours = tmp_path / "ours"
    ours.mkdir()
    queries = [{"id": f"q{i}", "question": f"question {i}?"} for i in (1, 2)]
    write_jsonl(ours / "test.jsonl", queries)
    write_jsonl(ours / "validation.jsonl", queries[:1])
    write_jsonl(ours / "test.bert.jsonl", [{"id": q["id"], "ranking": ["d1", "d2"]} for q in queries])
    (ours / "config.toml").write_text('run_dir = "x"\n')
    theirs = tmp_path / "theirs"
    shutil.copytree(ours, theirs)
    config.update(run_dir=str(ours), data_dir=str(tmp_path / "data"))
    return ours, theirs


def test_merge_adds_missing_rows_answers_and_cached_responses(tmp_path, config, monkeypatch):
    ours, theirs = setup_pair(tmp_path, config)
    monkeypatch.setattr("hybrid_reranker.merge.comparable", lambda value: None)
    append_jsonl(ours / "test.bert.jsonl", {"id": "q1", "ranking": ["d1", "d2"], "answer": "ours"})
    append_jsonl(theirs / "test.bert.jsonl", {"id": "q2", "ranking": ["d1", "d2"], "answer": "theirs"})
    write_jsonl(theirs / "test.llm.jsonl", [{"id": "q2", "ranking": ["d2", "d1"]}])
    (theirs / "thresholds.json").write_text('{"gap": {"threshold": 0.5}}')
    write_jsonl(tmp_path / "data/api.jsonl", [{"key": "a"}])
    write_jsonl(theirs / "api.jsonl", [{"key": "a"}, {"key": "b"}])
    assert merge(config, theirs, theirs / "api.jsonl")
    bert = load_results(ours / "test.bert.jsonl")
    assert bert["q1"]["answer"] == "ours" and bert["q2"]["answer"] == "theirs"
    assert list(load_results(ours / "test.llm.jsonl")) == ["q2"]
    assert (ours / "thresholds.json").exists()
    assert [row["key"] for row in read_jsonl(tmp_path / "data/api.jsonl")] == ["a", "b"]
    size = (ours / "test.bert.jsonl").read_bytes()
    assert merge(config, theirs)  # nothing new the second time
    assert (ours / "test.bert.jsonl").read_bytes() == size


def test_merge_refuses_different_questions_or_settings(tmp_path, config, monkeypatch, capsys):
    _, theirs = setup_pair(tmp_path, config)
    assert not merge(config, theirs)  # the toy snapshot is not this config
    assert "differs from this config" in capsys.readouterr().out
    monkeypatch.setattr("hybrid_reranker.merge.comparable", lambda value: None)
    write_jsonl(theirs / "test.jsonl", [{"id": "q9", "question": "?"}])
    assert not merge(config, theirs)
    assert "differs between the two runs" in capsys.readouterr().out
