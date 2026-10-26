"""Jev-gated model router with deterministic policy and intra-loop adaptation.

Principle: *Jev judges the task; Python owns the routing policy.*

- :class:`~axe.llm.jev_decision.JevClient` performs one fast evaluation pass.
- This module maps that evaluation onto a tier, thinking budget, and model queue.
- Per-turn routing contexts let the policy downshift routine verification turns
  and escalate immediately when a tool fails or a traceback appears.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import openai
from rich.console import Console
from rich.markup import escape
from rich.text import Text

from axe.config.settings import Settings
from axe.llm.jev_decision import (
    JevClient,
    JevEvaluation,
    RoutingContext,
    evaluate_heuristic_fallback,
)
from axe.llm.provider import (
    AllProvidersFailedError,
    LLMResponse,
    ProviderError,
    ProviderUnavailableError,
    ThinkingMode,
)
from axe.llm.providers.registry import ProviderRegistry
from axe.security.airlock_middleware import AirlockMiddleware

if TYPE_CHECKING:  # pragma: no cover - typing only
    from axe.cost.tracker import CostTracker

# Badge colour per built-in tier (custom tiers fall back to "bold").
_TIER_STYLES: dict[str, str] = {
    "fast": "bold blue",
    "coder": "bold cyan",
    "architect": "bold magenta",
    "security": "bold yellow",
}
_THINKING_ORDER: tuple[ThinkingMode, ...] = ("off", "low", "medium", "high")
_ESCALATION: dict[str, str] = {
    "fast": "coder",
    "coder": "architect",
    "architect": "architect",
    "security": "security",
}
_ROUTINE_TOOLS = {"file_read", "glob_search", "grep_search", "shell"}


@dataclass(slots=True)
class TierDecision:
    """The routing decision for one completion request."""

    tier: str
    models: list[str]
    thinking_mode: ThinkingMode = "low"
    max_tokens: int = 8192
    source: str = "config"
    confidence: float = 1.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    jev_eval: JevEvaluation | None = None
    risk_flagged: bool = False
    use_ast_compression: bool = False


class ModelRouter:
    """Resolve and call models from a Jev-selected tier queue."""

    def __init__(
        self,
        registry: ProviderRegistry,
        settings: Settings,
        console: Console | None = None,
        *,
        initial_tier: str | None = None,
        model_override: str | None = None,
        jev_client: JevClient | None = None,
        tracker: "CostTracker | None" = None,
        airlock: AirlockMiddleware | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._console = console or Console()
        self._requested_tier = initial_tier or "auto"
        self._model_override = model_override
        self._show_badge = settings.router.show_model_in_output
        self._last_error: Exception | None = None
        self._last_context: RoutingContext | None = None
        self._airlock = airlock or AirlockMiddleware(settings)
        self._jev = jev_client or JevClient(
            settings,
            tracker=tracker,
            airlock=self._airlock,
        )

        if self._requested_tier in (None, "", "auto"):
            self._requested_tier = "auto"
            # Avoid a Jev network call at startup; the first real turn
            # performs the single-pass evaluation.
            self._decision = self._decision_for_tier(
                settings.router.default_fallback,
                reason="startup default; awaiting Jev evaluation",
                source="default",
            )
        else:
            self._decision = self._forced_decision(self._requested_tier)

        self._current_model = (
            self._decision.models[0] if self._decision.models else ""
        )
        self._current_provider = ""

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def current_model(self) -> str:
        return self._current_model

    @property
    def current_provider(self) -> str:
        return self._current_provider

    @property
    def current_tier(self) -> str:
        return self._decision.tier

    @property
    def decision(self) -> TierDecision:
        return self._decision

    @property
    def last_decision(self) -> TierDecision:
        return self._decision

    @property
    def last_context(self) -> RoutingContext | None:
        return self._last_context

    @property
    def requested_tier(self) -> str:
        return self._requested_tier

    @property
    def routing_mode(self) -> str:
        return "auto" if self._requested_tier == "auto" else "forced"

    @property
    def available_tiers(self) -> list[str]:
        return self._settings.router.tiers.names()

    # ------------------------------------------------------------------
    # Tier selection
    # ------------------------------------------------------------------

    def select_tier(
        self,
        context: Mapping[str, Any] | RoutingContext | None = None,
    ) -> TierDecision:
        """Choose a tier for one turn and store the latest decision."""
        ctx = self._coerce_context(context)
        self._last_context = ctx
        previous_tier = (
            self._decision.tier
            if self._decision is not None
            else self._settings.router.default_fallback
        )

        if ctx.forced_tier and ctx.forced_tier != "auto":
            decision = self._forced_decision(ctx.forced_tier)
        else:
            try:
                jev = self._jev.evaluate(ctx)
            except Exception as exc:  # noqa: BLE001 - routing must never crash
                jev = evaluate_heuristic_fallback(
                    ctx,
                    reason=f"Jev client error: {exc}",
                )
            decision = self._map_jev_to_tier(ctx, jev, previous_tier)

        self._decision = decision
        self._current_model = decision.models[0] if decision.models else ""
        self._current_provider = ""
        return decision

    def set_tier(self, name: str, *, announce: bool = True) -> None:
        """Switch between Jev auto-routing and a locked tier."""
        normalized = (name or "").strip().lower()
        previous_requested = self._requested_tier
        previous_decision = self._decision

        try:
            if normalized in ("", "auto"):
                self._requested_tier = "auto"
                if self._last_context is None:
                    self._decision = self._decision_for_tier(
                        self._settings.router.default_fallback,
                        reason="auto mode enabled; awaiting Jev evaluation",
                        source="default",
                    )
                    self._current_model = (
                        self._decision.models[0] if self._decision.models else ""
                    )
                    self._current_provider = ""
                else:
                    self.select_tier(
                        replace(self._last_context, forced_tier=None)
                    )
            else:
                if normalized not in self.available_tiers:
                    available = ", ".join(self.available_tiers)
                    raise KeyError(
                        f"Unknown tier '{normalized}'. Available tiers: {available}."
                    )
                self._requested_tier = normalized
                self._decision = self._forced_decision(normalized)
                self._current_model = (
                    self._decision.models[0] if self._decision.models else ""
                )
                self._current_provider = ""
        except Exception:
            self._requested_tier = previous_requested
            self._decision = previous_decision
            raise

        if announce:
            self._print_tier_badge()

    # ------------------------------------------------------------------
    # Policy matrix
    # ------------------------------------------------------------------

    def _map_jev_to_tier(
        self,
        ctx: RoutingContext,
        jev: JevEvaluation,
        previous_tier: str,
    ) -> TierDecision:
        jev_settings = self._settings.router.jev
        risk_flagged = (
            jev.task_intent == "security_or_audit"
            or jev.destructive_risk_prob >= jev_settings.risk_gate_threshold
        )
        use_ast = jev.ast_sufficient_prob >= jev_settings.ast_compression_threshold
        source = "heuristic" if jev.used_fallback else "jev"
        metadata: dict[str, Any] = {
            "previous_tier": previous_tier,
            "reasoning_level": jev.reasoning_level,
            "needs_thinking_prob": jev.needs_thinking_prob,
        }

        def build(
            tier: str,
            thinking: ThinkingMode,
            rule: str,
            reason: str,
        ) -> TierDecision:
            if jev.used_fallback and jev.fallback_reason:
                reason = f"heuristic_fallback({jev.fallback_reason}): {reason}"
            metadata_with_rule = {**metadata, "policy_rule": rule}
            return self._decision_for_tier(
                tier,
                thinking_mode=thinking,
                reason=reason,
                source=source,
                confidence=jev.intent_confidence,
                metadata=metadata_with_rule,
                jev_eval=jev,
                risk_flagged=risk_flagged,
                use_ast_compression=use_ast,
            )

        # Rule 1 — security and destructive risk gate (highest priority).
        if risk_flagged:
            security_config = self._settings.router.tiers.get("security")
            base = security_config.thinking_mode if security_config else "medium"
            thinking = self._at_least(base, "medium")
            return build(
                "security",
                thinking,
                "security_risk_gate",
                (
                    f"jev:intent={jev.task_intent}({jev.intent_confidence:.2f}),"
                    f"risk={jev.destructive_risk_prob:.2f} -> "
                    f"security(thinking={thinking})"
                ),
            )

        # Rule 2 — intra-loop failure escalation and routine downshifting.
        if ctx.turn_index > 0:
            if ctx.failed:
                tier = _ESCALATION.get(previous_tier, "architect")
                thinking = "high" if tier in ("architect", "security") else "medium"
                return build(
                    tier,
                    thinking,
                    "intra_loop_failure_escalation",
                    (
                        f"intra_loop:tool={ctx.last_tool_name or 'unknown'},"
                        f"failed(exit={ctx.last_exit_code},"
                        f"traceback={ctx.has_traceback}),prev={previous_tier} -> "
                        f"{tier}(thinking={thinking})"
                    ),
                )

            if (
                ctx.last_tool_name in _ROUTINE_TOOLS
                and jev.reasoning_level in ("trivial", "low")
            ):
                return build(
                    "fast",
                    "off",
                    "intra_loop_downshift",
                    (
                        f"intra_loop:tool={ctx.last_tool_name},success,"
                        f"depth={jev.reasoning_level} -> fast(thinking=off)"
                    ),
                )

        # Rule 3 — standard intent and depth routing.
        if jev.intent_confidence < jev_settings.confidence_threshold:
            if jev.reasoning_level in ("high", "extreme"):
                tier = "architect"
            else:
                tier = self._settings.router.default_fallback
            thinking = self._base_thinking(tier)
            thinking = self._adjust_thinking(tier, thinking, jev)
            return build(
                tier,
                thinking,
                "low_confidence_fallback",
                (
                    f"jev:low_confidence={jev.intent_confidence:.2f},"
                    f"intent={jev.task_intent},depth={jev.reasoning_level} -> "
                    f"{tier}(thinking={thinking})"
                ),
            )

        if (
            jev.task_intent == "search_or_read"
            and jev.reasoning_level in ("trivial", "low")
        ):
            tier = "fast"
        elif (
            jev.task_intent == "multi_file_architect"
            or jev.reasoning_level in ("high", "extreme")
        ):
            tier = "architect"
        else:
            tier = "coder"

        thinking = self._adjust_thinking(tier, self._base_thinking(tier), jev)
        return build(
            tier,
            thinking,
            "standard_intent_depth",
            (
                f"jev:intent={jev.task_intent}({jev.intent_confidence:.2f}),"
                f"depth={jev.reasoning_level} -> {tier}(thinking={thinking})"
            ),
        )

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def _forced_decision(self, tier_name: str) -> TierDecision:
        return self._decision_for_tier(
            tier_name,
            reason=f"forced_tier={tier_name} (CLI/REPL lock)",
            source="forced",
        )

    def _decision_for_tier(
        self,
        tier_name: str,
        *,
        thinking_mode: ThinkingMode | None = None,
        max_tokens: int | None = None,
        reason: str,
        source: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
        jev_eval: JevEvaluation | None = None,
        risk_flagged: bool = False,
        use_ast_compression: bool = False,
    ) -> TierDecision:
        tier_config = self._settings.router.tiers.get(tier_name)
        if tier_config is None:
            available = ", ".join(self.available_tiers)
            raise ValueError(
                f"Unknown tier '{tier_name}'. Available tiers: {available}."
            )

        models = list(tier_config.models)
        if self._model_override:
            models = [
                self._model_override,
                *[model for model in models if model != self._model_override],
            ]
        if not models:
            raise ValueError(f"Tier '{tier_name}' has no models configured.")

        return TierDecision(
            tier=tier_name,
            models=models,
            thinking_mode=(
                tier_config.thinking_mode
                if thinking_mode is None
                else thinking_mode
            ),
            max_tokens=tier_config.max_tokens if max_tokens is None else max_tokens,
            source=source,
            confidence=confidence,
            reason=reason,
            metadata=metadata or {},
            jev_eval=jev_eval,
            risk_flagged=risk_flagged,
            use_ast_compression=use_ast_compression,
        )

    def _default_thinking(self, tier_name: str) -> ThinkingMode:
        tier_config = self._settings.router.tiers.get(tier_name)
        return tier_config.thinking_mode if tier_config else "low"

    def _base_thinking(self, tier_name: str) -> ThinkingMode:
        """Apply deterministic minimum thinking budgets per tier."""
        configured = self._default_thinking(tier_name)
        if tier_name == "architect":
            return self._at_least(configured, "high")
        if tier_name == "security":
            return self._at_least(configured, "medium")
        return configured

    @staticmethod
    def _thinking_index(mode: ThinkingMode) -> int:
        try:
            return _THINKING_ORDER.index(mode)
        except ValueError:
            return 1

    def _at_least(self, mode: ThinkingMode, minimum: ThinkingMode) -> ThinkingMode:
        return _THINKING_ORDER[
            max(self._thinking_index(mode), self._thinking_index(minimum))
        ]

    def _adjust_thinking(
        self,
        tier: str,
        thinking: ThinkingMode,
        jev: JevEvaluation,
    ) -> ThinkingMode:
        if tier not in ("fast", "coder"):
            return thinking

        if jev.needs_thinking_prob >= 0.75:
            index = min(self._thinking_index(thinking) + 1, len(_THINKING_ORDER) - 1)
            return _THINKING_ORDER[index]

        if tier == "coder" and jev.needs_thinking_prob < 0.20:
            return "off"
        return thinking

    def _coerce_context(
        self,
        context: Mapping[str, Any] | RoutingContext | None,
    ) -> RoutingContext:
        if isinstance(context, RoutingContext):
            ctx = context
        else:
            data = dict(context or {})
            prompt = str(data.get("user_prompt") or data.get("prompt") or "")
            if not prompt:
                messages = data.get("messages")
                if isinstance(messages, list):
                    for message in reversed(messages):
                        if (
                            isinstance(message, Mapping)
                            and message.get("role") == "user"
                            and isinstance(message.get("content"), str)
                        ):
                            prompt = str(message["content"])
                            break

            forced = data.get("forced_tier") or data.get("tier")
            ctx = RoutingContext(
                user_prompt=prompt,
                turn_index=_as_int(data.get("turn_index"), 0),
                last_tool_name=(
                    str(data["last_tool_name"])
                    if data.get("last_tool_name")
                    else None
                ),
                last_tool_failed=_as_bool(
                    data.get("last_tool_failed")
                    or data.get("last_tool_Failed")
                    or False
                ),
                last_exit_code=_as_optional_int(data.get("last_exit_code")),
                has_traceback=_as_bool(data.get("has_traceback") or False),
                files_in_context=max(_as_int(data.get("files_in_context"), 0), 0),
                forced_tier=str(forced) if forced else None,
            )

        if ctx.forced_tier == "auto":
            ctx = replace(ctx, forced_tier=None)
        if self._requested_tier not in (None, "", "auto"):
            ctx = replace(ctx, forced_tier=self._requested_tier)
        return ctx

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def _sanitize_text(self, text: str | None, source: str) -> str | None:
        if text is None:
            return None
        if not text:
            return text
        return self._airlock.sanitize_outbound(text, source=source).sanitized_text

    def _sanitize_messages(
        self,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        if messages is None:
            return None
        return self._airlock.sanitize_messages(messages)

    def _sanitize_tools(
        self,
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        if tools is None:
            return None
        sanitized = self._airlock.sanitize_payload(
            tools,
            source="tool_schemas",
        )
        return sanitized if isinstance(sanitized, list) else tools

    def _sanitize_context(
        self,
        context: Mapping[str, Any] | RoutingContext | None,
        *,
        safe_prompt: str | None,
        safe_messages: list[dict[str, Any]] | None,
    ) -> Mapping[str, Any] | RoutingContext | None:
        if context is None:
            return {
                "prompt": safe_prompt or "",
                "messages": safe_messages,
            }
        if isinstance(context, RoutingContext):
            safe_user_prompt = self._sanitize_text(
                context.user_prompt,
                "router_context",
            )
            if safe_user_prompt == context.user_prompt:
                return context
            return replace(context, user_prompt=safe_user_prompt or "")
        sanitized = dict(context)
        for key in ("user_prompt", "prompt"):
            value = sanitized.get(key)
            if isinstance(value, str):
                sanitized[key] = self._sanitize_text(value, "router_context")
        if isinstance(sanitized.get("messages"), list):
            sanitized["messages"] = self._sanitize_messages(
                sanitized["messages"]
            )
        return sanitized

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        thinking_mode: ThinkingMode | None = None,
        max_tokens: int | None = None,
        context: Mapping[str, Any] | RoutingContext | None = None,
    ) -> LLMResponse:
        """Select a tier via Jev, then walk its queue with fallback."""
        safe_prompt = self._sanitize_text(prompt, "router_prompt") or ""
        safe_system = self._sanitize_text(system_prompt, "router_system")
        safe_messages = self._sanitize_messages(messages)
        safe_tools = self._sanitize_tools(tools)
        safe_context = self._sanitize_context(
            context,
            safe_prompt=safe_prompt,
            safe_messages=safe_messages,
        )

        decision = self.select_tier(
            safe_context
            if safe_context is not None
            else {
                "prompt": safe_prompt,
                "messages": safe_messages,
                "tools": safe_tools,
            }
        )
        errors: list[tuple[str, Exception]] = []
        effective_thinking = (
            decision.thinking_mode if thinking_mode is None else thinking_mode
        )
        effective_max_tokens = (
            decision.max_tokens if max_tokens is None else max_tokens
        )

        for candidate in decision.models:
            try:
                resolved = self._registry.resolve(candidate)
            except Exception as exc:  # noqa: BLE001 - fall through to next model
                errors.append((candidate, exc))
                self._warn(f"{candidate} could not be resolved: {exc}")
                continue

            try:
                available, reason = resolved.availability()
            except Exception as exc:  # noqa: BLE001 - health checks are best-effort
                errors.append((candidate, exc))
                self._warn(f"{candidate} availability check failed: {exc}")
                continue
            if not available:
                error = ProviderUnavailableError(reason)
                errors.append((candidate, error))
                self._warn(f"{candidate} unavailable: {reason}")
                continue

            try:
                response = resolved.provider.complete(
                    safe_prompt,
                    system_prompt=safe_system,
                    tools=safe_tools,
                    messages=safe_messages,
                    thinking_mode=effective_thinking,
                    max_tokens=effective_max_tokens,
                )
            except (ProviderError, openai.APIError, OSError, TimeoutError) as exc:
                errors.append((candidate, exc))
                self._warn(
                    f"{candidate} failed on {resolved.provider_name}: {exc}"
                )
                continue
            except Exception as exc:  # noqa: BLE001 - never break the fallback chain
                errors.append((candidate, exc))
                self._warn(
                    f"{candidate} raised an unexpected error on "
                    f"{resolved.provider_name}: {exc}"
                )
                continue

            self._current_model = candidate
            self._current_provider = resolved.provider_name
            self._last_error = None
            return response

        self._last_error = errors[-1][1] if errors else None
        raise AllProvidersFailedError(decision.tier, decision.models, errors)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _warn(self, message: str) -> None:
        self._console.print(f"[warning]Warning: {escape(message)}[/warning]")

    def _print_tier_badge(self) -> None:
        if not self._show_badge:
            return
        text = Text()
        text.append("Tier set to ", style="dim")
        text.append(self._badge_label(), style=self._tier_style())
        self._console.print(text)

    def _badge_label(self) -> str:
        return f"[{self.current_tier}: {self.current_model}]"

    def _tier_style(self) -> str:
        return _TIER_STYLES.get(self.current_tier, "bold")

    def badge_text(self) -> Text:
        """Return a Rich Text with the current tier/model badge."""
        text = Text()
        text.append(self._badge_label(), style=self._tier_style())
        return text


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
