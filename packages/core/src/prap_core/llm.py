from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger("prap.llm")

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMRetryableError(LLMError):
    pass


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: TokenUsage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost_usd += other.cost_usd


@dataclass
class LLMResult:
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: dict[str, Any] | None = None
    from_cache: bool = False


def _cache_key(model: str, messages: list[dict[str, Any]], schema_hash: str | None) -> str:
    payload = json.dumps(
        {"m": model, "msg": messages, "s": schema_hash}, sort_keys=True, default=str
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=24).hexdigest()


class LLM:
    """Provider-agnostic LLM client.

    Wraps `litellm.completion` with: tenacity retries, structured-output
    coercion via pydantic, on-disk response cache keyed by prompt hash, and
    per-call token / cost accounting. Pipelines must not import litellm or
    provider SDKs directly.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        completion_fn: Any = None,
        embedding_fn: Any = None,
        cache: Any = None,
    ) -> None:
        if settings is None:
            from .config import Settings as _Settings

            settings = _Settings()
        self.settings = settings
        self._completion_fn = completion_fn  # injection point for tests
        self._embedding_fn = embedding_fn  # injection point for tests
        self._cache = cache  # injection point; if None and cache_enabled, lazy-init diskcache
        self.usage = TokenUsage()

    # ---- public API ----

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_format: type[T] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        cache: bool | None = None,
    ) -> LLMResult | T:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        use_cache = self.settings.cache_enabled if cache is None else cache
        schema_hash = (
            hashlib.blake2b(
                json.dumps(response_format.model_json_schema(), sort_keys=True).encode("utf-8"),
                digest_size=8,
            ).hexdigest()
            if response_format is not None
            else None
        )

        cache_obj = self._get_cache() if use_cache else None
        key = (
            _cache_key(self.settings.llm_model, messages, schema_hash)
            if cache_obj is not None
            else None
        )
        if cache_obj is not None and key in cache_obj:
            cached: dict[str, Any] = cache_obj[key]
            result = LLMResult(text=cached["text"], usage=TokenUsage(), from_cache=True)
            if response_format is not None:
                return response_format.model_validate_json(result.text)
            return result

        result = self._call_with_retry(messages, response_format, temperature, max_tokens)
        self.usage.add(result.usage)
        if cache_obj is not None and key is not None:
            cache_obj[key] = {"text": result.text}

        if response_format is not None:
            try:
                return response_format.model_validate_json(result.text)
            except ValidationError as e:
                raise LLMError(f"structured output failed to validate: {e}") from e
        return result

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Return one embedding vector per input string.

        Wraps `litellm.embedding` so pipelines can compute vector embeddings
        (e.g. for clustering / record linkage) without importing a provider
        SDK directly. Empty input returns an empty list.
        """
        if not texts:
            return []

        embedding_fn = self._embedding_fn
        if embedding_fn is None:
            try:
                from litellm import embedding as litellm_embedding
            except ImportError as e:
                raise LLMError("litellm is required for embedding calls") from e
            embedding_fn = litellm_embedding

        kwargs: dict[str, Any] = {
            "model": model or self.settings.embedding_model,
            "input": texts,
        }
        if self.settings.llm_api_key:
            kwargs["api_key"] = self.settings.llm_api_key
        if self.settings.llm_api_base:
            kwargs["api_base"] = self.settings.llm_api_base
        if self.settings.llm_api_version:
            kwargs["api_version"] = self.settings.llm_api_version

        try:
            raw = embedding_fn(**kwargs)
        except Exception as e:
            if _is_retryable(e):
                raise LLMRetryableError(str(e)) from e
            raise LLMError(str(e)) from e

        raw_dict = _to_dict(raw)
        try:
            vectors = [list(item["embedding"]) for item in raw_dict["data"]]
        except (KeyError, TypeError) as e:
            raise LLMError(f"unexpected embedding response shape: {raw_dict}") from e

        usage_dict = raw_dict.get("usage") or {}
        prompt_tokens = int(usage_dict.get("prompt_tokens", 0) or 0)
        total_tokens = int(usage_dict.get("total_tokens", prompt_tokens) or 0)
        self.usage.add(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                cost_usd=float(raw_dict.get("_hidden_params", {}).get("response_cost", 0.0) or 0.0),
            )
        )
        return vectors

    # ---- internals ----

    def _get_cache(self) -> Any | None:
        if self._cache is not None:
            return self._cache
        if not self.settings.cache_enabled:
            return None
        try:
            from diskcache import Cache  # local import keeps cache optional in tests
        except ImportError:
            return None
        cache_dir = Path(self.settings.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(cache_dir))
        return self._cache

    def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        response_format: type[BaseModel] | None,
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResult:
        @retry(
            reraise=True,
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(LLMRetryableError),
        )
        def _run() -> LLMResult:
            return self._call_once(messages, response_format, temperature, max_tokens)

        return _run()

    def _call_once(
        self,
        messages: list[dict[str, Any]],
        response_format: type[BaseModel] | None,
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResult:
        completion = self._completion_fn
        if completion is None:
            try:
                from litellm import completion as litellm_completion
            except ImportError as e:
                raise LLMError("litellm is required for LLM calls") from e
            completion = litellm_completion

        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.settings.llm_api_key:
            kwargs["api_key"] = self.settings.llm_api_key
        if self.settings.llm_api_base:
            kwargs["api_base"] = self.settings.llm_api_base
        if self.settings.llm_api_version:
            kwargs["api_version"] = self.settings.llm_api_version
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            raw = completion(**kwargs)
        except Exception as e:
            if _is_retryable(e):
                raise LLMRetryableError(str(e)) from e
            raise LLMError(str(e)) from e

        raw_dict = _to_dict(raw)
        try:
            text = raw_dict["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {raw_dict}") from e

        usage_dict = raw_dict.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(usage_dict.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_dict.get("completion_tokens", 0) or 0),
            total_tokens=int(usage_dict.get("total_tokens", 0) or 0),
            cost_usd=float(raw_dict.get("_hidden_params", {}).get("response_cost", 0.0) or 0.0),
        )
        return LLMResult(text=text, usage=usage, raw=raw_dict)


def _to_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if hasattr(raw, "dict"):
        return raw.dict()
    if hasattr(raw, "__dict__"):
        return dict(raw.__dict__)
    raise LLMError(f"cannot coerce response to dict: {type(raw)!r}")


_RETRYABLE_KEYWORDS = (
    "rate limit",
    "ratelimit",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "503",
    "502",
    "504",
)


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in _RETRYABLE_KEYWORDS)
