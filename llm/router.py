"""Multi-model router with failure-based tier escalation.

Wraps ``OpenAIProvider`` and swaps the active model string based on
consecutive API/SDK-level failures.  Tool execution errors are NOT
counted as escalation triggers — only exceptions raised by the OpenAI
SDK itself (network errors, rate limits, server errors, bad responses).

Tier ladder (all via one OPENAI_API_KEY):
  Tier 1 — gpt-4.1-mini   (default, cheapest)
  Tier 2 — gpt-4.1         (escalate after 2 Tier-1 failures)
  Tier 3 — o3-mini          (escalate after 2 Tier-2 failures)
"""

from __future__ import annotations

from typing import Any

import openai
from rich.console import Console
from rich.text import Text

from config.settings import Settings
from llm.openai_provider import OpenAIProvider
from llm.provider import LLMResponse


# Badge strings shown in the terminal when a tier is active / newly escalated
_BADGES: dict[int, tuple[str, str]] = {
    0: ("[gpt-4.1-mini]", "bold blue"),
    1: ("[↑ gpt-4.1]",    "bold magenta"),
    2: ("[↑↑ o3-mini]",   "bold yellow"),
}

# o3-mini does not support temperature; use 1 (API default) for that tier
_O3_MINI_TIER = 2


class ModelRouter:
    """Wrap an OpenAIProvider and auto-escalate through model tiers on failure.

    The router exposes the same ``.complete()`` interface as
    ``OpenAIProvider``, so the agent loop can treat it as a drop-in.

    Args:
        provider:   Base ``OpenAIProvider`` (configured with Tier 1 model).
        settings:   Full ``Settings`` — reads ``llm`` tiers and ``router``
                    escalation config.
        console:    Rich console for badge output.  Defaults to a new one.
    """

    def __init__(
        self,
        provider: OpenAIProvider,
        settings: Settings,
        console: Console | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._console = console or Console()

        # Ordered tier list from settings
        self._tiers: list[str] = [
            settings.llm.tier1_model,
            settings.llm.tier2_model,
            settings.llm.tier3_model,
        ]
        self._current_tier: int = 0
        self._consecutive_failures: int = 0
        self._escalate_after: int = settings.router.escalate_after_failures
        self._reset_on_success: bool = settings.router.reset_on_success
        self._show_badge: bool = settings.router.show_model_in_output

        # Sync provider to Tier 1
        self._apply_tier()

    # ------------------------------------------------------------------
    # Public API (same shape as LLMProvider.complete)
    # ------------------------------------------------------------------

    @property
    def current_model(self) -> str:
        """The model string currently active."""
        return self._tiers[self._current_tier]

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Call the active model tier, escalating on API/SDK failures.

        After each failure the router increments the failure counter and
        escalates the tier if the threshold is reached.  The call is then
        retried on the new tier automatically — the exception only propagates
        to the caller once all tiers have been exhausted.
        """
        _OPENAI_ERRORS = (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIStatusError,
        )

        last_exc: Exception | None = None

        # Try every tier from the current one up to the last
        for attempt in range(self._current_tier, len(self._tiers)):
            try:
                response = self._provider.complete(
                    prompt,
                    system_prompt=system_prompt,
                    tools=tools,
                    messages=messages,
                )
                self._on_success()
                return response
            except _OPENAI_ERRORS as exc:
                last_exc = exc
                self._on_failure(exc)
                # _on_failure may have escalated — if we're still on the same
                # tier (threshold not yet reached), don't advance the loop
                if self._current_tier == attempt:
                    # Not escalated yet (below threshold) — re-raise so the
                    # caller sees the error rather than silently retrying
                    raise

        # All tiers exhausted
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Tier management
    # ------------------------------------------------------------------

    def _on_success(self) -> None:
        if self._reset_on_success and self._consecutive_failures > 0:
            self._consecutive_failures = 0

    def _on_failure(self, exc: Exception) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._escalate_after:
            self._try_escalate()

    def _try_escalate(self) -> None:
        if self._current_tier < len(self._tiers) - 1:
            self._current_tier += 1
            self._consecutive_failures = 0
            self._apply_tier()
            self._print_escalation_badge()
        # Already at max tier — nothing more to do; let the error propagate

    def _apply_tier(self) -> None:
        """Point the provider at the current tier's model string."""
        model = self._tiers[self._current_tier]
        self._provider.model = model
        # o3-mini does not support custom temperature
        if self._current_tier == _O3_MINI_TIER:
            self._provider.temperature = 1
        else:
            self._provider.temperature = self._settings.llm.openai.temperature

    def _print_escalation_badge(self) -> None:
        if not self._show_badge:
            return
        badge, style = _BADGES.get(self._current_tier, ("[?]", "bold"))
        t = Text()
        t.append("⚡ Escalating to ", style="dim")
        t.append(badge, style=style)
        self._console.print(t)

    def badge_text(self) -> Text:
        """Return a Rich Text with the current model badge (for inline display)."""
        badge, style = _BADGES.get(self._current_tier, (f"[{self.current_model}]", "bold"))
        t = Text()
        t.append(badge, style=style)
        return t
