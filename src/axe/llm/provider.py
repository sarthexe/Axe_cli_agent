"""Normalized provider interface, usage accounting, and shared errors."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

ThinkingMode = Literal["off", "low", "medium", "high"]
DEFAULT_THINKING_MODE: ThinkingMode = "low"

_INLINE_THINK_RE = re.compile(
    r"<(think|thinking)\b[^>]*>(.*?)</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(slots=True)
class ToolCall:
    """A tool invocation returned by an LLM."""

    name: str
    arguments: dict[str, Any]
    id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenUsage:
    """Normalized token, cache, reasoning, and latency accounting."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    ttft_ms: float = 0.0
    total_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.total_ms <= 0.0 and self.latency_ms > 0.0:
            self.total_ms = self.latency_ms
        if self.cached_prompt_tokens < 0:
            self.cached_prompt_tokens = 0
        if self.cached_prompt_tokens > self.prompt_tokens:
            self.cached_prompt_tokens = self.prompt_tokens
        if self.reasoning_tokens < 0:
            self.reasoning_tokens = 0
        if self.completion_tokens < 0:
            self.completion_tokens = 0
        if self.prompt_tokens < 0:
            self.prompt_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_miss_prompt_tokens(self) -> int:
        return max(self.prompt_tokens - self.cached_prompt_tokens, 0)

    @property
    def cache_hit_ratio(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return (self.cached_prompt_tokens / self.prompt_tokens) * 100.0


@dataclass(slots=True)
class LLMResponse:
    """Normalized provider response consumed by the agent loop."""

    content: str
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Backward-compatible alias for :attr:`content`."""
        return self.content


class ProviderError(Exception):
    """Base class for provider-layer failures."""


class ProviderUnavailableError(ProviderError):
    """The provider cannot be reached or has not been configured."""


class MissingProviderKeyError(ProviderUnavailableError):
    """A cloud provider is configured without an API key."""


class ProviderResponseError(ProviderError):
    """The provider returned an unusable or malformed response."""


class UnknownProviderError(ProviderError):
    """A model string referenced a provider that is not configured."""


class AllProvidersFailedError(ProviderError):
    """Every candidate model in a routing tier failed."""

    def __init__(
        self,
        tier: str,
        models: list[str],
        errors: list[tuple[str, Exception]],
    ) -> None:
        self.tier = tier
        self.models = models
        self.errors = errors
        details = "; ".join(f"{model}: {error}" for model, error in errors)
        super().__init__(
            f"All {len(models)} model(s) failed for tier '{tier}'"
            + (f" - {details}" if details else ".")
        )


def split_inline_reasoning(content: str | None) -> tuple[str, str]:
    """Split ``<think>...</think>`` blocks out of model content.

    Returns:
        ``(clean_content, reasoning)``. Multiple blocks are joined with a
        blank line; the tags are removed from the visible content.
    """
    if not content:
        return "", ""

    blocks = [match.group(2).strip() for match in _INLINE_THINK_RE.finditer(content)]
    reasoning = "\n\n".join(block for block in blocks if block)
    clean = _INLINE_THINK_RE.sub("", content).strip()
    return clean, reasoning


def merge_reasoning(native: str | None, inline: str) -> str | None:
    """Combine native reasoning fields with inline thinking blocks."""
    parts = [part.strip() for part in (native, inline) if part and part.strip()]
    if not parts:
        return None
    return "\n\n".join(parts)


def as_text(value: Any) -> str:
    """Normalize provider reasoning/content values into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or value)
    return str(value)


class LLMProvider(ABC):
    """Abstract provider contract used by the agent loop."""

    name: str = "provider"
    model: str = ""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        thinking_mode: ThinkingMode = DEFAULT_THINKING_MODE,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a normalized model completion."""

    def is_available(self) -> tuple[bool, str]:
        """Return ``(available, reason)`` before attempting a call.

        Cloud providers check for credentials; local providers can perform a
        lightweight health check. The default implementation is optimistic.
        """
        return True, ""
