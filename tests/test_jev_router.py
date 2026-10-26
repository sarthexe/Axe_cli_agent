"""Tests for Jev decisioning, policy routing, and intra-loop adaptation."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from rich.console import Console

from axe.agent.core import Agent
from axe.cli import repl as repl_module
from axe.config.settings import Settings, load_settings
from axe.cost.tracker import CostTracker
from axe.llm.jev_decision import (
    EVALUATION_QUESTIONS,
    JEV_PRICING_MODEL,
    REASONING_LEVELS,
    TASK_INTENT_OPTIONS,
    JevClient,
    JevEvaluation,
    RoutingContext,
    evaluate_heuristic_fallback,
)
from axe.llm.provider import LLMProvider, LLMResponse, TokenUsage, ToolCall
from axe.llm.providers import ProviderRegistry
from axe.llm.router import ModelRouter, TierDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(env: dict[str, str] | None = None) -> Settings:
    return load_settings(
        env=env or {},
        include_user=False,
        include_workspace=False,
    )


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self) -> Any:
        return self._payload


class FakeHTTPClient:
    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "json": json or {},
                "headers": headers or {},
                "timeout": timeout,
            }
        )
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _native_response() -> dict[str, Any]:
    return {
        "answers": {
            "task_intent": {
                "choice": "multi_file_architect",
                "confidence": 0.89,
                "probabilities": {
                    "search_or_read": 0.03,
                    "single_file_edit": 0.05,
                    "multi_file_architect": 0.89,
                    "security_or_audit": 0.03,
                },
            },
            "reasoning_depth": {"level": "high", "score": 0.75},
            "needs_thinking_trace": {"probability": 0.81},
            "destructive_risk": {"probability": 0.12},
            "ast_context_sufficient": {"probability": 0.77},
        },
        "usage": {"prompt_tokens": 200, "completion_tokens": 20},
    }


def _evaluation(**overrides: Any) -> JevEvaluation:
    values: dict[str, Any] = {
        "task_intent": "search_or_read",
        "intent_confidence": 0.90,
        "intent_probabilities": {
            "search_or_read": 0.90,
            "single_file_edit": 0.04,
            "multi_file_architect": 0.03,
            "security_or_audit": 0.03,
        },
        "reasoning_level": "low",
        "reasoning_score": 0.25,
        "needs_thinking_prob": 0.10,
        "destructive_risk_prob": 0.10,
        "ast_sufficient_prob": 0.80,
        "latency_ms": 120.0,
    }
    values.update(overrides)
    return JevEvaluation(**values)


class FakeJev:
    def __init__(self, evaluations: list[JevEvaluation] | Any) -> None:
        self._evaluations = evaluations
        self.contexts: list[RoutingContext] = []

    def evaluate(self, ctx: RoutingContext) -> JevEvaluation:
        self.contexts.append(ctx)
        if callable(self._evaluations):
            return self._evaluations(ctx)
        return self._evaluations.pop(0)


def _router(
    settings: Settings,
    jev: Any,
    *,
    tier: str = "auto",
    model_override: str | None = None,
) -> ModelRouter:
    return ModelRouter(
        ProviderRegistry(settings),
        settings,
        Console(quiet=True),
        initial_tier=tier,
        model_override=model_override,
        jev_client=jev,
    )


# ---------------------------------------------------------------------------
# Jev payload + parsing
# ---------------------------------------------------------------------------


def test_jev_payload_contains_all_five_questions_and_parses_native_response() -> None:
    settings = _settings()
    tracker = CostTracker()
    http = FakeHTTPClient(FakeResponse(200, _native_response()))
    client = JevClient(
        settings,
        tracker=tracker,
        http_client=http,
        env={"JEV_API_KEY": "jev-test"},
    )
    ctx = RoutingContext(
        user_prompt="Refactor the provider layer across the codebase",
        turn_index=0,
        files_in_context=4,
    )

    evaluation = client.evaluate(ctx)

    assert evaluation.used_fallback is False
    assert evaluation.task_intent == "multi_file_architect"
    assert evaluation.intent_confidence == pytest.approx(0.89)
    assert evaluation.intent_probabilities["multi_file_architect"] == pytest.approx(0.89)
    assert evaluation.reasoning_level == "high"
    assert evaluation.reasoning_score == pytest.approx(0.75)
    assert evaluation.needs_thinking_prob == pytest.approx(0.81)
    assert evaluation.destructive_risk_prob == pytest.approx(0.12)
    assert evaluation.ast_sufficient_prob == pytest.approx(0.77)
    assert evaluation.latency_ms >= 0

    call = http.calls[0]
    assert call["url"] == settings.router.jev.endpoint
    assert call["headers"]["Authorization"] == "Bearer jev-test"
    assert call["timeout"] == pytest.approx(
        settings.router.jev.timeout_ms / 1000.0
    )

    payload = call["json"]
    questions = {question["name"]: question for question in payload["questions"]}
    assert set(questions) == {
        "task_intent",
        "reasoning_depth",
        "needs_thinking_trace",
        "destructive_risk",
        "ast_context_sufficient",
    }
    assert len(EVALUATION_QUESTIONS) == 5
    assert questions["task_intent"]["type"] == "choice"
    assert questions["task_intent"]["options"] == TASK_INTENT_OPTIONS
    assert questions["reasoning_depth"]["type"] == "score"
    assert questions["reasoning_depth"]["levels"] == REASONING_LEVELS
    for name in (
        "needs_thinking_trace",
        "destructive_risk",
        "ast_context_sufficient",
    ):
        assert questions[name]["type"] == "noul"
    assert payload["context"]["user_prompt"] == ctx.user_prompt
    assert payload["context"]["files_in_context"] == 4

    assert tracker.per_tier_calls == {"jev": 1}
    assert tracker.total_prompt_tokens == 200
    assert tracker.total_completion_tokens == 20
    assert tracker.total_cost() == pytest.approx((200 + 20) * 0.02 / 1_000_000)
    assert tracker.per_model_stats()[JEV_PRICING_MODEL]["calls"] == 1


def test_jev_parses_openrouter_structured_chat_response() -> None:
    answers = {
        "task_intent": {"value": "search_or_read", "confidence": 0.91},
        "reasoning_depth": "high",
        "needs_thinking_trace": True,
        "destructive_risk": "0.82",
        "ast_context_sufficient": {"score": 0.77},
    }
    response = {
        "choices": [
            {
                "message": {
                    "content": "```json\n" + json.dumps(answers) + "\n```"
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10},
    }
    http = FakeHTTPClient(FakeResponse(200, response))
    client = JevClient(
        _settings(),
        http_client=http,
        env={"JEV_API_KEY": "jev-test"},
    )

    evaluation = client.evaluate(RoutingContext(user_prompt="grep for TODO"))

    assert evaluation.used_fallback is False
    assert evaluation.task_intent == "search_or_read"
    assert evaluation.intent_confidence == pytest.approx(0.91)
    assert evaluation.reasoning_level == "high"
    assert evaluation.reasoning_score == pytest.approx(0.75)
    assert evaluation.needs_thinking_prob == pytest.approx(1.0)
    assert evaluation.destructive_risk_prob == pytest.approx(0.82)
    assert evaluation.ast_sufficient_prob == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# Policy matrix
# ---------------------------------------------------------------------------


def test_high_destructive_risk_routes_to_security_and_flags_risk() -> None:
    settings = _settings()
    jev = FakeJev(
        [
            _evaluation(
                task_intent="single_file_edit",
                intent_confidence=0.80,
                reasoning_level="medium",
                reasoning_score=0.50,
                destructive_risk_prob=0.82,
                ast_sufficient_prob=0.40,
            )
        ]
    )
    router = _router(settings, jev)

    decision = router.select_tier(
        RoutingContext(user_prompt="remove credentials from the config")
    )

    assert decision.tier == "security"
    assert decision.risk_flagged is True
    assert decision.thinking_mode in {"medium", "high"}
    assert decision.jev_eval is not None
    assert decision.source == "jev"
    assert decision.use_ast_compression is False


def test_security_intent_routes_to_security_even_with_low_risk() -> None:
    jev = FakeJev(
        [
            _evaluation(
                task_intent="security_or_audit",
                intent_confidence=0.93,
                reasoning_level="medium",
                destructive_risk_prob=0.10,
            )
        ]
    )
    router = _router(_settings(), jev)

    decision = router.select_tier(
        RoutingContext(user_prompt="audit the auth flow")
    )

    assert decision.tier == "security"
    assert decision.risk_flagged is True


def test_search_or_read_low_depth_routes_fast_with_thinking_off() -> None:
    jev = FakeJev(
        [
            _evaluation(
                task_intent="search_or_read",
                intent_confidence=0.91,
                reasoning_level="low",
                reasoning_score=0.25,
                needs_thinking_prob=0.10,
                ast_sufficient_prob=0.90,
            )
        ]
    )
    router = _router(_settings(), jev)

    decision = router.select_tier(
        RoutingContext(user_prompt="find every TODO comment")
    )

    assert decision.tier == "fast"
    assert decision.thinking_mode == "off"
    assert decision.risk_flagged is False
    assert decision.use_ast_compression is True
    assert "search_or_read" in decision.reason


def test_multi_file_architect_routes_architect_with_high_thinking() -> None:
    jev = FakeJev(
        [
            _evaluation(
                task_intent="multi_file_architect",
                intent_confidence=0.89,
                reasoning_level="high",
                reasoning_score=0.75,
                needs_thinking_prob=0.88,
                ast_sufficient_prob=0.30,
            )
        ]
    )
    router = _router(_settings(), jev)

    decision = router.select_tier(
        RoutingContext(user_prompt="refactor the whole codebase")
    )

    assert decision.tier == "architect"
    assert decision.thinking_mode == "high"
    assert decision.confidence == pytest.approx(0.89)


def test_low_confidence_falls_back_to_coder() -> None:
    jev = FakeJev(
        [
            _evaluation(
                task_intent="single_file_edit",
                intent_confidence=0.45,
                reasoning_level="medium",
                reasoning_score=0.50,
                needs_thinking_prob=0.10,
            )
        ]
    )
    router = _router(_settings(), jev)

    decision = router.select_tier(
        RoutingContext(user_prompt="do something ambiguous")
    )

    assert decision.tier == "coder"
    assert decision.metadata["policy_rule"] == "low_confidence_fallback"
    assert decision.confidence == pytest.approx(0.45)


def test_dynamic_thinking_bumps_coder_on_high_thinking_need() -> None:
    jev = FakeJev(
        [
            _evaluation(
                task_intent="single_file_edit",
                intent_confidence=0.90,
                reasoning_level="medium",
                needs_thinking_prob=0.80,
            )
        ]
    )
    router = _router(_settings(), jev)

    decision = router.select_tier(
        RoutingContext(user_prompt="fix the complex parser bug")
    )

    assert decision.tier == "coder"
    assert decision.thinking_mode == "medium"


def test_forced_tier_bypasses_jev_entirely() -> None:
    settings = _settings()
    jev = FakeJev([])
    router = _router(settings, jev, tier="architect")

    decision = router.select_tier(
        RoutingContext(user_prompt="this should not be evaluated")
    )

    assert decision.tier == "architect"
    assert decision.source == "forced"
    assert decision.jev_eval is None
    assert jev.contexts == []
    assert router.requested_tier == "architect"
    assert router.routing_mode == "forced"


def test_set_tier_switches_between_auto_and_locked_modes() -> None:
    router = _router(_settings(), FakeJev([]))

    router.set_tier("fast", announce=False)
    assert router.requested_tier == "fast"
    assert router.routing_mode == "forced"
    assert router.current_tier == "fast"
    assert router.last_decision.source == "forced"

    router.set_tier("auto", announce=False)
    assert router.requested_tier == "auto"
    assert router.routing_mode == "auto"
    assert router.current_tier == "coder"


def test_tier_status_displays_decision_and_jev_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jev = FakeJev(
        [
            _evaluation(
                task_intent="multi_file_architect",
                intent_confidence=0.89,
                reasoning_level="high",
                reasoning_score=0.75,
                needs_thinking_prob=0.88,
                destructive_risk_prob=0.12,
                ast_sufficient_prob=0.81,
                latency_ms=145.0,
            )
        ]
    )
    router = _router(_settings(), jev)
    router.select_tier(RoutingContext(user_prompt="refactor the router"))

    buffer = io.StringIO()
    test_console = Console(
        file=buffer,
        width=200,
        force_terminal=False,
        color_system=None,
    )
    monkeypatch.setattr(repl_module, "console", test_console)

    repl_module._print_tier_status(SimpleNamespace(router=router))  # type: ignore[arg-type]
    output = buffer.getvalue()

    assert "Routing mode:" in output
    assert "auto (Jev)" in output
    assert "Selected:" in output
    assert "architect" in output
    assert "thinking=high" in output
    assert "intent=multi_file_architect (0.89)" in output
    assert "risk=0.12" in output
    assert "latency=145ms" in output
    assert "risk_flagged=False" in output
    assert "ast_compression=True" in output


# ---------------------------------------------------------------------------
# Intra-loop downshifting / escalation
# ---------------------------------------------------------------------------


def test_successful_routine_tool_turn_downshifts_architect_to_fast() -> None:
    jev = FakeJev(
        [
            _evaluation(
                task_intent="multi_file_architect",
                intent_confidence=0.89,
                reasoning_level="high",
                reasoning_score=0.75,
                needs_thinking_prob=0.80,
            ),
            _evaluation(
                task_intent="search_or_read",
                intent_confidence=0.91,
                reasoning_level="low",
                reasoning_score=0.20,
                needs_thinking_prob=0.10,
            ),
        ]
    )
    router = _router(_settings(), jev)

    turn_zero = router.select_tier(
        RoutingContext(user_prompt="refactor the router", turn_index=0)
    )
    turn_one = router.select_tier(
        RoutingContext(
            user_prompt="refactor the router",
            turn_index=1,
            last_tool_name="grep_search",
            last_tool_failed=False,
            last_exit_code=0,
            has_traceback=False,
            files_in_context=2,
        )
    )

    assert turn_zero.tier == "architect"
    assert turn_one.tier == "fast"
    assert turn_one.thinking_mode == "off"
    assert turn_one.metadata["policy_rule"] == "intra_loop_downshift"


def test_failed_shell_escalates_fast_to_coder_with_thinking() -> None:
    jev = FakeJev(
        [
            _evaluation(
                task_intent="search_or_read",
                intent_confidence=0.91,
                reasoning_level="low",
                needs_thinking_prob=0.10,
            ),
            _evaluation(
                task_intent="single_file_edit",
                intent_confidence=0.80,
                reasoning_level="medium",
                needs_thinking_prob=0.30,
            ),
        ]
    )
    router = _router(_settings(), jev)

    turn_zero = router.select_tier(
        RoutingContext(user_prompt="find the bug", turn_index=0)
    )
    turn_one = router.select_tier(
        RoutingContext(
            user_prompt="find the bug",
            turn_index=1,
            last_tool_name="shell",
            last_tool_failed=True,
            last_exit_code=1,
            has_traceback=True,
        )
    )

    assert turn_zero.tier == "fast"
    assert turn_one.tier == "coder"
    assert turn_one.thinking_mode == "medium"
    assert turn_one.metadata["policy_rule"] == "intra_loop_failure_escalation"


def test_failed_shell_escalates_coder_to_architect_with_high_thinking() -> None:
    jev = FakeJev(
        [
            _evaluation(
                task_intent="single_file_edit",
                intent_confidence=0.90,
                reasoning_level="medium",
                needs_thinking_prob=0.20,
            ),
            _evaluation(
                task_intent="single_file_edit",
                intent_confidence=0.90,
                reasoning_level="medium",
                needs_thinking_prob=0.20,
            ),
        ]
    )
    router = _router(_settings(), jev)

    turn_zero = router.select_tier(
        RoutingContext(user_prompt="fix the parser", turn_index=0)
    )
    turn_one = router.select_tier(
        RoutingContext(
            user_prompt="fix the parser",
            turn_index=1,
            last_tool_name="shell",
            last_tool_failed=False,
            last_exit_code=1,
            has_traceback=False,
        )
    )

    assert turn_zero.tier == "coder"
    assert turn_one.tier == "architect"
    assert turn_one.thinking_mode == "high"


# ---------------------------------------------------------------------------
# Resilience / fallback
# ---------------------------------------------------------------------------


def test_missing_api_key_uses_heuristic_without_http_call() -> None:
    settings = _settings()
    http = FakeHTTPClient(FakeResponse(200, _native_response()))
    client = JevClient(settings, http_client=http, env={})

    evaluation = client.evaluate(
        RoutingContext(user_prompt="delete the credentials file")
    )

    assert evaluation.used_fallback is True
    assert evaluation.fallback_reason is not None
    assert "key" in evaluation.fallback_reason.lower()
    assert http.calls == []
    assert evaluation.destructive_risk_prob >= settings.router.jev.risk_gate_threshold
    assert evaluation.task_intent == "security_or_audit"


def test_disabled_jev_uses_heuristic_without_http_call() -> None:
    settings = _settings()
    settings.router.jev.enabled = False
    http = FakeHTTPClient(FakeResponse(200, _native_response()))
    client = JevClient(settings, http_client=http, env={"JEV_API_KEY": "test"})

    evaluation = client.evaluate(RoutingContext(user_prompt="list files"))

    assert evaluation.used_fallback is True
    assert evaluation.fallback_reason == "Jev disabled"
    assert http.calls == []


def test_http_500_falls_back_without_raising() -> None:
    http = FakeHTTPClient(FakeResponse(500, {"error": "boom"}))
    client = JevClient(
        _settings(),
        http_client=http,
        env={"JEV_API_KEY": "test"},
    )

    evaluation = client.evaluate(RoutingContext(user_prompt="search for tests"))

    assert evaluation.used_fallback is True
    assert evaluation.fallback_reason is not None
    assert "HTTP 500" in evaluation.fallback_reason


def test_timeout_falls_back_without_raising() -> None:
    http = FakeHTTPClient(error=httpx.TimeoutException("timed out"))
    client = JevClient(
        _settings(),
        http_client=http,
        env={"JEV_API_KEY": "test"},
    )

    evaluation = client.evaluate(RoutingContext(user_prompt="search for tests"))

    assert evaluation.used_fallback is True
    assert evaluation.fallback_reason is not None
    assert "timeout" in evaluation.fallback_reason.lower()


def test_jev_uses_provider_key_when_jev_key_is_absent() -> None:
    settings = _settings({"OPENROUTER_API_KEY": "or-provider-key"})
    http = FakeHTTPClient(FakeResponse(200, _native_response()))
    client = JevClient(
        settings,
        http_client=http,
        env={"OPENROUTER_API_KEY": "or-provider-key"},
    )

    client.evaluate(RoutingContext(user_prompt="refactor"))

    assert http.calls[0]["headers"]["Authorization"] == "Bearer or-provider-key"


def test_heuristic_fallback_detects_risk_and_failure_complexity() -> None:
    risky = evaluate_heuristic_fallback(
        RoutingContext(user_prompt="rotate auth secrets and delete credentials")
    )
    failed = evaluate_heuristic_fallback(
        RoutingContext(
            user_prompt="grep for TODO",
            turn_index=1,
            last_tool_failed=True,
            last_exit_code=1,
            has_traceback=True,
        )
    )

    assert risky.used_fallback is True
    assert risky.task_intent == "security_or_audit"
    assert risky.destructive_risk_prob >= 0.65
    assert failed.reasoning_score >= 0.50
    assert failed.needs_thinking_prob >= 0.50


def test_heuristic_downshifts_successful_routine_turns() -> None:
    evaluation = evaluate_heuristic_fallback(
        RoutingContext(
            user_prompt="refactor the whole codebase",
            turn_index=1,
            last_tool_name="grep_search",
            last_tool_failed=False,
            last_exit_code=0,
            has_traceback=False,
        )
    )

    assert evaluation.reasoning_level in {"trivial", "low"}
    assert evaluation.destructive_risk_prob < 0.65


def test_router_uses_heuristic_source_when_jev_falls_back() -> None:
    settings = _settings()
    client = JevClient(settings, env={})
    router = _router(settings, client)

    decision = router.select_tier(
        RoutingContext(user_prompt="grep for TODO comments")
    )

    assert decision.source == "heuristic"
    assert decision.jev_eval is not None
    assert decision.jev_eval.used_fallback is True
    assert decision.tier == "fast"


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


class FakeDecisionProvider(LLMProvider):
    name = "fake"
    model = "fake/model"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_decision = TierDecision(
            tier="fast",
            models=["fake/model"],
            thinking_mode="off",
            max_tokens=256,
            reason="test decision",
        )

    def select_tier(self, context: RoutingContext) -> TierDecision:
        return self.last_decision

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        return LLMResponse(
            content="done",
            model="fake/model",
            provider="fake",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
        )


def test_agent_passes_routing_context_and_stores_last_decision() -> None:
    settings = _settings()
    provider = FakeDecisionProvider()
    agent = Agent(
        provider,
        MagicMock(),
        settings,
        console=Console(quiet=True),
    )

    result = agent.run("hello")

    assert result == "done"
    context = provider.calls[0]["context"]
    assert isinstance(context, RoutingContext)
    assert context.user_prompt == "hello"
    assert context.turn_index == 0
    assert agent.last_routing_context is context
    assert agent.last_decision is provider.last_decision


def test_agent_tracks_tool_failures_tracebacks_and_files() -> None:
    settings = _settings()
    agent = Agent(
        FakeDecisionProvider(),
        MagicMock(),
        settings,
        console=Console(quiet=True),
    )

    agent._record_tool_outcome(
        "shell",
        {"command": "false"},
        "Command exited with code 1:\nTraceback (most recent call last):\nboom",
        "error",
    )
    failed_context = agent._build_routing_context("fix it", 1)

    assert failed_context.last_tool_name == "shell"
    assert failed_context.last_tool_failed is True
    assert failed_context.last_exit_code == 1
    assert failed_context.has_traceback is True
    assert failed_context.failed is True

    agent._record_tool_outcome(
        "file_read",
        {"path": "src/axe/app.py"},
        "file contents",
        "success",
    )
    success_context = agent._build_routing_context("fix it", 2)

    assert success_context.last_tool_name == "file_read"
    assert success_context.last_tool_failed is False
    assert success_context.last_exit_code is None
    assert success_context.has_traceback is False
    assert success_context.files_in_context == 1


class _ToolCallingProvider(FakeDecisionProvider):
    def __init__(self) -> None:
        super().__init__()
        self._turn = 0

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        self._turn += 1
        if self._turn == 1:
            return LLMResponse(
                content="",
                model="fake/model",
                provider="fake",
                tool_calls=[
                    ToolCall(
                        name="file_read",
                        arguments={"path": "src/axe/app.py"},
                        id="call_1",
                    )
                ],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
            )
        return LLMResponse(
            content="done",
            model="fake/model",
            provider="fake",
            usage=TokenUsage(prompt_tokens=12, completion_tokens=2),
        )


class _ToolRegistryStub:
    def get_all_schemas(self) -> list[dict[str, Any]]:
        return []

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        assert name == "file_read"
        assert arguments == {"path": "src/axe/app.py"}
        return "Command exited with code 1:\nTraceback (most recent call last):\nboom"


def test_agent_loop_advances_routing_context_across_tool_turns() -> None:
    settings = _settings()
    provider = _ToolCallingProvider()
    agent = Agent(
        provider,
        _ToolRegistryStub(),  # type: ignore[arg-type]
        settings,
        console=Console(quiet=True),
    )

    result = agent.run("fix the failing parser")

    assert result == "done"
    assert len(provider.calls) == 2
    first_context = provider.calls[0]["context"]
    second_context = provider.calls[1]["context"]
    assert first_context.turn_index == 0
    assert second_context.turn_index == 1
    assert second_context.last_tool_name == "file_read"
    assert second_context.last_tool_failed is True
    assert second_context.last_exit_code == 1
    assert second_context.has_traceback is True
    assert second_context.files_in_context == 1
