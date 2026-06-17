import pytest
from prap_core.config import Settings
from prap_core.llm import LLM, LLMError, LLMResult, LLMRetryableError
from pydantic import BaseModel


class _Schema(BaseModel):
    answer: str


def _fake_response(content: str, *, prompt_tokens=3, completion_tokens=2, cost=0.0):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "_hidden_params": {"response_cost": cost},
    }


def _settings(**overrides):
    return Settings(_env_file=None, cache_enabled=False, **overrides)


def test_complete_happy_path():
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)
        return _fake_response("hi", cost=0.001)

    llm = LLM(_settings(), completion_fn=fake)
    result = llm.complete("ping", system="be terse")
    assert isinstance(result, LLMResult)
    assert result.text == "hi"
    assert result.usage.total_tokens == 5
    assert llm.usage.total_tokens == 5
    assert llm.usage.cost_usd == pytest.approx(0.001)
    assert calls[0]["messages"][0]["role"] == "system"
    assert calls[0]["messages"][1]["content"] == "ping"


def test_structured_output_parses():
    def fake(**kwargs):
        return _fake_response('{"answer": "42"}')

    llm = LLM(_settings(), completion_fn=fake)
    out = llm.complete("q", response_format=_Schema)
    assert isinstance(out, _Schema)
    assert out.answer == "42"


def test_structured_output_validation_error():
    def fake(**kwargs):
        return _fake_response('{"wrong_field": "x"}')

    llm = LLM(_settings(), completion_fn=fake)
    with pytest.raises(LLMError):
        llm.complete("q", response_format=_Schema)


def test_retry_on_retryable_then_success():
    n = {"calls": 0}

    def fake(**kwargs):
        n["calls"] += 1
        if n["calls"] < 3:
            raise RuntimeError("rate limit exceeded")
        return _fake_response("ok")

    llm = LLM(_settings(), completion_fn=fake)
    out = llm.complete("q")
    assert out.text == "ok"
    assert n["calls"] == 3


def test_non_retryable_raises_immediately():
    n = {"calls": 0}

    def fake(**kwargs):
        n["calls"] += 1
        raise RuntimeError("bad request: invalid model")

    llm = LLM(_settings(), completion_fn=fake)
    with pytest.raises(LLMError):
        llm.complete("q")
    assert n["calls"] == 1


def test_cache_hit_skips_call():
    cache: dict = {}
    n = {"calls": 0}

    def fake(**kwargs):
        n["calls"] += 1
        return _fake_response("cached-value")

    s = Settings(_env_file=None, cache_enabled=True)
    llm = LLM(s, completion_fn=fake, cache=cache)
    a = llm.complete("same")
    b = llm.complete("same")
    assert a.text == "cached-value"
    assert b.from_cache is True
    assert b.text == "cached-value"
    assert n["calls"] == 1


def test_retry_exhaustion():
    def fake(**kwargs):
        raise RuntimeError("connection timeout")

    llm = LLM(_settings(), completion_fn=fake)
    with pytest.raises(LLMRetryableError):
        llm.complete("q")


def test_unexpected_response_shape():
    def fake(**kwargs):
        return {"choices": []}

    llm = LLM(_settings(), completion_fn=fake)
    with pytest.raises(LLMError):
        llm.complete("q")


def _fake_embedding_response(vectors, *, prompt_tokens=4, cost=0.0):
    return {
        "data": [{"embedding": v} for v in vectors],
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        "_hidden_params": {"response_cost": cost},
    }


def test_embed_happy_path():
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)
        return _fake_embedding_response([[0.1, 0.2], [0.3, 0.4]], cost=0.0001)

    llm = LLM(_settings(), embedding_fn=fake)
    vectors = llm.embed(["alpha", "beta"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert calls[0]["input"] == ["alpha", "beta"]
    assert calls[0]["model"] == "openai/text-embedding-3-large"
    assert llm.usage.prompt_tokens == 4
    assert llm.usage.cost_usd == pytest.approx(0.0001)


def test_embed_empty_input_short_circuits():
    def fake(**kwargs):
        raise AssertionError("should not be called")

    llm = LLM(_settings(), embedding_fn=fake)
    assert llm.embed([]) == []


def test_embed_model_override():
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return _fake_embedding_response([[1.0]])

    llm = LLM(_settings(), embedding_fn=fake)
    llm.embed(["x"], model="openai/text-embedding-3-small")
    assert captured["model"] == "openai/text-embedding-3-small"


def test_embed_retryable_error():
    def fake(**kwargs):
        raise RuntimeError("rate limit exceeded")

    llm = LLM(_settings(), embedding_fn=fake)
    with pytest.raises(LLMRetryableError):
        llm.embed(["x"])


def test_embed_unexpected_response_shape():
    def fake(**kwargs):
        return {"data": [{"not_embedding": [1.0]}]}

    llm = LLM(_settings(), embedding_fn=fake)
    with pytest.raises(LLMError):
        llm.embed(["x"])
