import json
import socket
import tomllib
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """All tests use local fixtures; accidental network access fails the test."""
    def deny(*args, **kwargs):
        pytest.fail("Network access is disabled in tests")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
def offline_run(run_dir, config):
    """MOCK rankings: the LLM helps 10/50 queries, hurts 10/50, and ties on the rest."""
    from hybrid_reranker.controls import load_pairs
    from hybrid_reranker.io import write_json, write_jsonl

    def make(split="test", prefix=None, count=50):
        queries, bert, llm = [], [], []
        prefix = prefix or split
        for index in range(count):
            query_id = f"{prefix}-{index:03d}"
            ranking = ["gold1", "gold2"] + [f"x{i}" for i in range(18)]
            bad = ranking[2:12] + ranking[:2] + ranking[12:]
            first, second = (bad, ranking) if index < 10 else (ranking, bad) if index >= count - 10 else (ranking, ranking)
            queries.append({"id": query_id, "question": f"MOCK {index}", "answer": "yes", "mock": True,
                            "qrels": {"gold1": 1, "gold2": 1}, "supporting_facts": []})
            gap = (index + 1) / 10
            row = {"id": query_id, "ranking": first, "scores": [20 - gap * i for i in range(20)], "gap": gap,
                   "llm_called": False, "rerank_usage": {}, "context": [], "answer": "yes",
                   "answer_usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
            bert.append(row)
            llm.append({**row, "ranking": second, "llm_called": True,
                        "rerank_usage": {"prompt_tokens": 2300, "completion_tokens": 100, "total_tokens": 2400}})
        write_jsonl(run_dir / f"{split}.jsonl", queries)
        write_jsonl(run_dir / f"{split}.bert.jsonl", bert)
        write_jsonl(run_dir / f"{split}.llm.jsonl", llm)
        if not (run_dir / "thresholds.json").exists():
            write_json(run_dir / "thresholds.json", {"gap": {"threshold": 1.0}, "cosine": {"threshold": 0.5}})
        return load_pairs(config, split)
    return make


@pytest.fixture
def config(tmp_path):
    """The project settings with all output folders moved to a temporary directory and no API pauses."""
    base = tomllib.loads((ROOT / "config.toml").read_text())
    return {
        **base, "data_dir": str(tmp_path / "data"), "run_dir": str(tmp_path / "run"),
        "requests_per_minute": 600000, "tokens_per_minute": 10**12,
    }


@pytest.fixture
def run_dir(config):
    path = Path(config["run_dir"])
    path.mkdir(parents=True)
    return path


@pytest.fixture
def tokenizer():
    """The tiktoken vocabulary used for prompt sizes; skipped when it cannot be downloaded."""
    tiktoken = pytest.importorskip("tiktoken")
    try:
        tiktoken.get_encoding("o200k_harmony")
    except Exception as error:  # noqa: BLE001 - any download problem means no network
        pytest.skip(f"tokenizer unavailable: {error}")


class FakeGroq:
    """Stands in for the Groq client. `behaviour(request)` returns (content, finish_reason) for each call."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.requests = []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **request):
        self.requests.append(request)
        content, finish_reason = self.behaviour(request)
        usage = {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050}
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=types.SimpleNamespace(model_dump=lambda: dict(usage)),
        )


@pytest.fixture
def groq(monkeypatch, tokenizer):
    """Install a fake API client; the returned function takes the behaviour and returns the client."""
    from hybrid_reranker import llm

    def install(behaviour):
        client = FakeGroq(behaviour)
        monkeypatch.setattr(llm, "Groq", lambda **kwargs: client)
        return client

    return install


def permutation(count=20):
    return json.dumps({"ranking": list(range(count, 0, -1))})


def documents(count=20):
    return [{"id": f"d{i}", "title": f"Title {i}", "text": "some words " * 10, "score": 1 - i / 100} for i in range(1, count + 1)]
