"""LLM provider interface and common response types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCall:
    """A tool invocation returned by an LLM."""

    name: str
    arguments: dict[str, Any]
    id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponse:
    """Normalized provider response."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract provider contract used by the agent loop."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Return a model completion for the prompt."""
        raise NotImplementedError
