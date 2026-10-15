import json
import tomllib
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


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
