"""Shared lazy LLM accessor for clustering pipelines.

All clustering sub-pipelines route LLM calls through this module so that
they share a single :class:`prap_core.llm.LLM` instance (and therefore a
single :class:`prap_core.llm.TokenUsage` aggregator).
"""

from __future__ import annotations

from threading import Lock

from prap_core.llm import LLM

_lock = Lock()
_llm: LLM | None = None


def get_llm() -> LLM:
    """Return the process-wide :class:`prap_core.llm.LLM` instance."""
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                _llm = LLM()
    return _llm


def reset_usage() -> None:
    """Reset cumulative token usage on the shared LLM client."""
    llm = get_llm()
    llm.usage.prompt_tokens = 0
    llm.usage.completion_tokens = 0
    llm.usage.total_tokens = 0
    llm.usage.cost_usd = 0.0


def get_usage_dict() -> dict:
    """Return cumulative token usage in the ``prompt_gpt`` dict format."""
    u = get_llm().usage
    return {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "call_count": 0,
    }
