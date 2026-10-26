"""Jev single-pass decision engine for fast tier routing.

Jev evaluates five orthogonal questions in one ``/evaluate`` call. The client
normalizes TypeSafe-native and OpenRouter/OpenJev structured responses into a
single :class:`JevEvaluation`. Every failure path falls back to a deterministic
local heuristic so routing can never crash or block the CLI.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from axe.config.settings import Settings
from axe.security.airlock_middleware import AirlockMiddleware

# ---------------------------------------------------------------------------
# Question schema
# ---------------------------------------------------------------------------

TASK_INTENT_OPTIONS = [
    "search_or_read",
    "single_file_edit",
    "multi_file_architect",
    "security_or_audit",
]
REASONING_LEVELS = ["trivial", "low", "medium", "high", "extreme"]

# The canonical batch is sent in a single /evaluate request.
EVALUATION_QUESTIONS: list[dict[str, Any]] = [
    {
        "name": "task_intent",
        "type": "choice",
        "question": "What is the primary engineering action required for this turn?",
        "options": list(TASK_INTENT_OPTIONS),
    },
    {
        "name": "reasoning_depth",
        "type": "score",
        "question": (
            "How much multi-step reasoning or algorithmic complexity "
            "does this step require?"
        ),
        "levels": list(REASONING_LEVELS),
    },
    {
        "name": "needs_thinking_trace",
        "type": "noul",
        "question": (
            "Does solving this step require deep chain-of-thought planning "
            "before calling a tool?"
        ),
    },
    {
        "name": "destructive_risk",
        "type": "noul",
        "question": (
            "Could this request or the likely next tool call modify critical "
            "configs, delete data, execute risky shell commands, or touch "
            "secrets/auth?"
        ),
    },
    {
        "name": "ast_context_sufficient",
        "type": "noul",
        "question": (
            "Can this turn be answered using file trees and AST function "
            "signatures without dumping full raw file bodies?"
        ),
    },
]

JEV_PRICING_MODEL = "typesafe/jev-latest"

# Alias used by integrations that refer to the batch by its generic name.
QUESTION_BATCH = EVALUATION_QUESTIONS

_QUESTION_ALIASES: dict[str, tuple[str, ...]] = {
    "task_intent": ("task_intent", "intent"),
    "reasoning_depth": ("reasoning_depth", "reasoning_level"),
    "needs_thinking_trace": (
        "needs_thinking_trace",
        "needs_thinking",
        "needs_thinking_prob",
    ),
    "destructive_risk": ("destructive_risk", "destructive_risk_prob", "risk"),
    "ast_context_sufficient": (
        "ast_context_sufficient",
        "ast_sufficient",
        "ast_sufficient_prob",
    ),
}


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


@dataclass(slots=True, init=False)
class RoutingContext:
    """Per-turn context handed to Jev and the router policy."""

    user_prompt: str = ""
    turn_index: int = 0
    last_tool_name: str | None = None
    last_tool_failed: bool = False
    last_exit_code: int | None = None
    has_traceback: bool = False
    files_in_context: int = 0
    forced_tier: str | None = None

    def __init__(
        self,
        user_prompt: str = "",
        turn_index: int = 0,
        last_tool_name: str | None = None,
        last_tool_failed: bool = False,
        last_exit_code: int | None = None,
        has_traceback: bool = False,
        files_in_context: int = 0,
        forced_tier: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Accept the historical spec spelling ``last_tool_Failed`` as well.
        if "last_tool_Failed" in kwargs:
            last_tool_failed = bool(kwargs.pop("last_tool_Failed"))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected RoutingContext argument(s): {unknown}")

        self.user_prompt = user_prompt
        self.turn_index = max(int(turn_index or 0), 0)
        self.last_tool_name = last_tool_name
        self.last_tool_failed = _as_bool(last_tool_failed)
        self.last_exit_code = _as_optional_int(last_exit_code)
        self.has_traceback = _as_bool(has_traceback)
        self.files_in_context = max(int(files_in_context or 0), 0)
        self.forced_tier = forced_tier

    @property
    def last_tool_Failed(self) -> bool:
        """Compatibility alias for the historical spec spelling."""
        return self.last_tool_failed

    @last_tool_Failed.setter
    def last_tool_Failed(self, value: Any) -> None:
        self.last_tool_failed = _as_bool(value)

    @property
    def is_continuation(self) -> bool:
        return self.turn_index > 0

    @property
    def failed(self) -> bool:
        return (
            self.last_tool_failed
            or self.has_traceback
            or self.last_exit_code not in (None, 0)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_prompt": self.user_prompt,
            "turn_index": self.turn_index,
            "last_tool_name": self.last_tool_name,
            "last_tool_failed": self.last_tool_failed,
            "last_exit_code": self.last_exit_code,
            "has_traceback": self.has_traceback,
            "files_in_context": self.files_in_context,
            "forced_tier": self.forced_tier,
        }


@dataclass(slots=True)
class JevEvaluation:
    """Normalized output of one Jev decision pass."""

    task_intent: str = "search_or_read"
    intent_confidence: float = 0.0
    intent_probabilities: dict[str, float] = field(default_factory=dict)
    reasoning_level: str = "low"
    reasoning_score: float = 0.25
    needs_thinking_prob: float = 0.0
    destructive_risk_prob: float = 0.0
    ast_sufficient_prob: float = 0.0
    latency_ms: float = 0.0
    used_fallback: bool = False
    fallback_reason: str | None = None


# ---------------------------------------------------------------------------
# Deterministic heuristic fallback
# ---------------------------------------------------------------------------

_SECURITY_RE = re.compile(
    r"\b(?:security|audit|auth(?:entication|orization)?|secret|credentials?|"
    r"passw(?:or)?d|api[-_ ]?key|tokens?|vulnerab\w*|exploit|cve|permission|"
    r"chmod|sudo|encrypt|decrypt|\.env|airlock)\b",
    re.IGNORECASE,
)
_DESTRUCTIVE_HIGH_RE = re.compile(
    r"(rm\s+-rf|drop\s+(?:the\s+)?(?:database|db|table)|"
    r"(?:delete|remove|overwrite|truncate)\s+[^\n]*\b(?:data|database|db|"
    r"table|files?|everything)\b|"
    r"wipe|destroy|reset\s+--hard|force\s+push|mkfs|dd\s+if=|chmod\s+777|"
    r"sudo\s+rm|\bsecrets?\b|credentials?|\.env|deploy|migration)",
    re.IGNORECASE,
)
_DESTRUCTIVE_MODERATE_RE = re.compile(
    r"\b(?:delete|remove|overwrite|truncate|chmod)\b",
    re.IGNORECASE,
)
_MULTI_FILE_RE = re.compile(
    r"\b(?:refactor|architect\w*|multi[-_ ]?file|codebase|redesign|migrat\w*|"
    r"package|module|system[-_ ]?wide|performance|optimi[sz]\w*|rewrite|across)\b",
    re.IGNORECASE,
)
_EDIT_RE = re.compile(
    r"\b(?:edit|fix|change|update|rename|add|implement|write|modify|patch|"
    r"bugfix|create|remove)\b",
    re.IGNORECASE,
)
_COMPLEX_RE = re.compile(
    r"\b(?:algorithm|concurren\w*|distributed|race\s+condition|deadlock|"
    r"complex|debug|traceback|exception|regex|optimi[sz]\w*|architecture|"
    r"fail\w*|tests?|bug\w*|error\w*|stack\s+trace)\b",
    re.IGNORECASE,
)
_FULL_BODY_RE = re.compile(
    r"\b(?:full\s+file|entire\s+file|whole\s+file|raw\s+file|dump)\b",
    re.IGNORECASE,
)
_ROUTINE_TOOLS = {"file_read", "glob_search", "grep_search", "shell"}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _level_from_score(score: float) -> str:
    if score < 0.20:
        return "trivial"
    if score < 0.40:
        return "low"
    if score < 0.60:
        return "medium"
    if score < 0.80:
        return "high"
    return "extreme"


def _score_from_level(level: str) -> float:
    normalized = level.strip().lower()
    if normalized in REASONING_LEVELS:
        return REASONING_LEVELS.index(normalized) / (len(REASONING_LEVELS) - 1)
    return 0.25


def _probabilities(selected: str, confidence: float) -> dict[str, float]:
    confidence = _clamp(confidence)
    others = [option for option in TASK_INTENT_OPTIONS if option != selected]
    remaining = (1.0 - confidence) / len(others) if others else 0.0
    return {
        option: confidence if option == selected else remaining
        for option in TASK_INTENT_OPTIONS
    }


def evaluate_heuristic_fallback(
    ctx: RoutingContext,
    reason: str = "deterministic heuristic fallback",
) -> JevEvaluation:
    """Produce a valid evaluation without any network or model call."""
    ctx = ctx or RoutingContext()
    prompt = ctx.user_prompt or ""
    lower = prompt.lower()

    if _SECURITY_RE.search(prompt):
        task_intent = "security_or_audit"
        confidence = 0.90
    elif _MULTI_FILE_RE.search(prompt):
        task_intent = "multi_file_architect"
        confidence = 0.84
    elif _EDIT_RE.search(prompt):
        task_intent = "single_file_edit"
        confidence = 0.80
    else:
        task_intent = "search_or_read"
        confidence = 0.78

    score = 0.15
    if _COMPLEX_RE.search(prompt):
        score += 0.22
    if task_intent == "multi_file_architect":
        score += 0.22
    if task_intent == "security_or_audit":
        score += 0.12
    score += min(max(ctx.turn_index, 0) * 0.05, 0.20)
    score += min(max(ctx.files_in_context, 0) * 0.02, 0.10)
    if ctx.last_tool_failed:
        score += 0.20
    if ctx.has_traceback:
        score += 0.20
    if ctx.last_exit_code not in (None, 0):
        score += 0.15
    if "think" in lower or "plan" in lower:
        score += 0.05
    if (
        ctx.turn_index > 0
        and not ctx.failed
        and ctx.last_tool_name in _ROUTINE_TOOLS
    ):
        # A successful read/search/verification step is usually routine even
        # when the overall task is complex; let the policy downshift it.
        score = min(score, 0.30)
    score = _clamp(score)

    needs_thinking = _clamp(
        score * 0.85
        + (0.18 if task_intent == "multi_file_architect" else 0.0)
        + (0.20 if ctx.failed else 0.0)
    )

    if _DESTRUCTIVE_HIGH_RE.search(prompt):
        destructive_risk = 0.86
    elif _DESTRUCTIVE_MODERATE_RE.search(prompt):
        destructive_risk = 0.55
    elif task_intent == "security_or_audit":
        destructive_risk = 0.72
    elif task_intent in {"single_file_edit", "multi_file_architect"}:
        destructive_risk = 0.34
    elif ctx.has_traceback or ctx.last_tool_failed:
        destructive_risk = 0.28
    else:
        destructive_risk = 0.08

    if _FULL_BODY_RE.search(prompt):
        ast_sufficient = 0.20
    elif task_intent == "search_or_read" and ctx.files_in_context <= 3:
        ast_sufficient = 0.90
    elif task_intent == "single_file_edit":
        ast_sufficient = 0.65
    elif task_intent == "security_or_audit":
        ast_sufficient = 0.35
    else:
        ast_sufficient = 0.30

    return JevEvaluation(
        task_intent=task_intent,
        intent_confidence=confidence,
        intent_probabilities=_probabilities(task_intent, confidence),
        reasoning_level=_level_from_score(score),
        reasoning_score=score,
        needs_thinking_prob=needs_thinking,
        destructive_risk_prob=_clamp(destructive_risk),
        ast_sufficient_prob=_clamp(ast_sufficient),
        latency_ms=0.0,
        used_fallback=True,
        fallback_reason=reason,
    )


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def build_evaluation_payload(
    ctx: RoutingContext,
    model: str = JEV_PRICING_MODEL,
) -> dict[str, Any]:
    """Build the single-pass ``/evaluate`` request payload."""
    ctx = ctx or RoutingContext()
    return {
        "model": model,
        "input": ctx.user_prompt,
        "context": ctx.as_dict(),
        "questions": [dict(question) for question in EVALUATION_QUESTIONS],
        "temperature": 0.0,
        "max_tokens": 512,
    }


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------

_ANSWER_CONTAINERS = (
    "answers",
    "results",
    "evaluations",
    "responses",
    "response",
    "output",
    "data",
    "result",
)


def _parse_json_object(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, Mapping) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, Mapping) else None
            except json.JSONDecodeError:
                return None
    return None


def _extract_answers(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Unwrap native, nested, and OpenAI-chat response shapes."""
    if "choices" in data:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, Mapping) else None
            if isinstance(message, Mapping):
                parsed = _parse_json_object(message.get("content"))
                if parsed is not None:
                    nested = _extract_answers(parsed)
                    return nested if nested is not None else parsed
                for call in message.get("tool_calls") or []:
                    if not isinstance(call, Mapping):
                        continue
                    function = call.get("function")
                    if isinstance(function, Mapping):
                        parsed = _parse_json_object(function.get("arguments"))
                        if parsed is not None:
                            nested = _extract_answers(parsed)
                            return nested if nested is not None else parsed

    for container in _ANSWER_CONTAINERS:
        value = data.get(container)
        if isinstance(value, Mapping):
            nested = _extract_answers(value)
            return nested if nested is not None else value
        if isinstance(value, list):
            mapped = _extract_answer_list(value)
            if mapped:
                return mapped
        parsed = _parse_json_object(value)
        if parsed is not None:
            nested = _extract_answers(parsed)
            return nested if nested is not None else parsed

    return data


def _extract_answer_list(items: list[Any]) -> dict[str, Any] | None:
    question_names = {
        question["question"]: question["name"] for question in EVALUATION_QUESTIONS
    }
    mapped: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name") or item.get("id")
        if not name:
            question = item.get("question")
            name = question_names.get(str(question)) if question else None
        if not name:
            continue
        mapped[str(name)] = item.get("answer", item.get("value", item))
    return mapped or None


def _lookup(answers: Mapping[str, Any], name: str) -> Any:
    lowered = {
        re.sub(r"[\s-]+", "_", str(key).strip().lower()): value
        for key, value in answers.items()
    }
    for alias in _QUESTION_ALIASES[name]:
        key = re.sub(r"[\s-]+", "_", alias.lower())
        if key in lowered:
            return lowered[key]
    for question in EVALUATION_QUESTIONS:
        if question["name"] != name:
            continue
        key = re.sub(r"[\s-]+", "_", str(question["question"]).lower())
        if key in lowered:
            return lowered[key]
    return None


def _choice_value(answer: Any) -> tuple[str | None, dict[str, float]]:
    probabilities: dict[str, float] = {}
    value: Any = answer

    if isinstance(answer, Mapping):
        raw_probabilities = (
            answer.get("probabilities")
            or answer.get("distribution")
            or answer.get("scores")
        )
        if isinstance(raw_probabilities, Mapping):
            for key, raw in raw_probabilities.items():
                try:
                    probabilities[str(key)] = _clamp(float(raw))
                except (TypeError, ValueError):
                    continue
        for key in ("choice", "label", "value", "intent", "selected", "answer"):
            if key in answer and answer[key] is not None:
                value = answer[key]
                break

    if isinstance(value, str):
        normalized = re.sub(r"[\s-]+", "_", value.strip().lower())
        for option in TASK_INTENT_OPTIONS:
            if normalized == option:
                return option, probabilities
        if "security" in normalized or "audit" in normalized:
            return "security_or_audit", probabilities
        if "multi" in normalized or "architect" in normalized:
            return "multi_file_architect", probabilities
        if "edit" in normalized or "fix" in normalized:
            return "single_file_edit", probabilities
        if "search" in normalized or "read" in normalized:
            return "search_or_read", probabilities

    if probabilities:
        known = {
            option: probabilities[option]
            for option in TASK_INTENT_OPTIONS
            if option in probabilities
        }
        if known:
            return max(known, key=known.get), probabilities
    return None, probabilities


def _answer_confidence(answer: Any, probabilities: Mapping[str, float]) -> float:
    if isinstance(answer, Mapping):
        for key in ("confidence", "probability", "score", "prob"):
            if key in answer and answer[key] is not None:
                try:
                    value = float(answer[key])
                    return _clamp(value / 100.0 if value > 1.0 else value)
                except (TypeError, ValueError):
                    pass
    if probabilities:
        return _clamp(max(probabilities.values()))
    return 0.0


def _reasoning_value(answer: Any) -> tuple[str | None, float | None]:
    level: str | None = None
    score: float | None = None

    if isinstance(answer, Mapping):
        raw_level = (
            answer.get("level")
            or answer.get("label")
            or answer.get("choice")
            or answer.get("answer")
        )
        if raw_level is None:
            candidate = answer.get("value")
            if isinstance(candidate, str):
                raw_level = candidate
        if isinstance(raw_level, str):
            level = raw_level.strip().lower()
        elif isinstance(raw_level, (int, float)):
            score = float(raw_level)
        for key in ("score", "value", "confidence", "prob"):
            if key in answer and answer[key] is not None:
                if isinstance(answer[key], str) and answer[key].strip().lower() in REASONING_LEVELS:
                    level = answer[key].strip().lower()
                    continue
                try:
                    score = float(answer[key])
                except (TypeError, ValueError):
                    pass
                break
    elif isinstance(answer, str):
        level = answer.strip().lower()
    elif isinstance(answer, (int, float)):
        score = float(answer)

    if score is not None:
        if 0.0 <= score <= 1.0:
            normalized_score = score
        elif 0.0 <= score <= 4.0:
            normalized_score = score / 4.0
        elif 0.0 <= score <= 100.0:
            normalized_score = score / 100.0
        else:
            normalized_score = 0.25
        score = _clamp(normalized_score)

    if level not in REASONING_LEVELS:
        level = None
    if level is None and score is not None:
        level = _level_from_score(score)
    if score is None and level is not None:
        score = _score_from_level(level)
    return level, score


def _noul_value(answer: Any) -> float | None:
    value: Any = answer
    if isinstance(answer, Mapping):
        for key in (
            "probability",
            "prob",
            "value",
            "score",
            "confidence",
            "boolean",
            "answer",
        ):
            if key in answer and answer[key] is not None:
                value = answer[key]
                break

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return 1.0
        if lowered in {"false", "no", "n", "0"}:
            return 0.0
        try:
            numeric = float(lowered)
        except ValueError:
            return None
        return _clamp(numeric / 100.0 if numeric > 1.0 else numeric)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return _clamp(numeric / 100.0 if numeric > 1.0 else numeric)
    return None


def _find_usage(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    usage = data.get("usage")
    if isinstance(usage, Mapping):
        return usage
    for container in _ANSWER_CONTAINERS:
        nested = data.get(container)
        if isinstance(nested, Mapping):
            usage = nested.get("usage")
            if isinstance(usage, Mapping):
                return usage
    return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class JevClient:
    """Resilient single-pass Jev ``/evaluate`` client."""

    def __init__(
        self,
        settings: Settings,
        *,
        tracker: Any | None = None,
        http_client: httpx.Client | None = None,
        env: Mapping[str, str] | None = None,
        airlock: AirlockMiddleware | None = None,
    ) -> None:
        self.settings = settings
        self.jev = settings.router.jev
        self._tracker = tracker
        self._http_client = http_client
        self._env = os.environ if env is None else env
        self._timeout_seconds = max(self.jev.timeout_ms, 1) / 1000.0
        self._airlock = airlock or AirlockMiddleware(settings)

    def _sanitize_context(self, ctx: RoutingContext) -> RoutingContext:
        if not ctx.user_prompt:
            return ctx
        safe_prompt = self._airlock.sanitize_outbound(
            ctx.user_prompt,
            source="jev",
        ).sanitized_text
        if safe_prompt == ctx.user_prompt:
            return ctx
        return replace(ctx, user_prompt=safe_prompt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, ctx: RoutingContext) -> JevEvaluation:
        """Evaluate a routing context; never raises."""
        ctx = self._sanitize_context(ctx or RoutingContext())

        if not self.jev.enabled:
            return evaluate_heuristic_fallback(ctx, reason="Jev disabled")

        try:
            api_key = self._resolve_api_key()
        except Exception as exc:  # noqa: BLE001 - routing must never crash
            return evaluate_heuristic_fallback(
                ctx,
                reason=f"Jev credential resolution failed: {exc}",
            )
        if not api_key:
            return evaluate_heuristic_fallback(
                ctx,
                reason="missing JEV_API_KEY or provider API key",
            )

        payload = build_evaluation_payload(ctx, self.jev.model)
        started = time.perf_counter()
        try:
            response = self._post(self.jev.endpoint, payload, self._headers(api_key))
            if response.status_code >= 400:
                return evaluate_heuristic_fallback(
                    ctx,
                    reason=f"Jev HTTP {response.status_code}",
                )
            data = response.json()
        except httpx.TimeoutException:
            return evaluate_heuristic_fallback(
                ctx,
                reason=f"Jev timeout after {self.jev.timeout_ms}ms",
            )
        except httpx.HTTPError as exc:
            return evaluate_heuristic_fallback(
                ctx,
                reason=f"Jev request failed: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - routing must never crash
            return evaluate_heuristic_fallback(
                ctx,
                reason=f"Jev request failed: {exc}",
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(data, Mapping):
            return evaluate_heuristic_fallback(
                ctx,
                reason="Jev returned a non-object response",
            )

        try:
            evaluation = self._parse_response(data, ctx, latency_ms)
        except Exception as exc:  # noqa: BLE001 - never crash on malformed payloads
            return evaluate_heuristic_fallback(
                ctx,
                reason=f"Jev response parse failed: {exc}",
            )
        if evaluation is None:
            return evaluate_heuristic_fallback(
                ctx,
                reason="unrecognized Jev response shape",
            )

        self._record_usage(data, latency_ms)
        return evaluation

    # ------------------------------------------------------------------
    # Credentials / transport
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str:
        env_key = str(self._env.get("JEV_API_KEY", "") or "").strip()
        if env_key:
            return env_key

        configured = str(self.jev.api_key or "").strip()
        if configured:
            return configured

        provider = self.settings.providers.get(self.jev.provider)
        if provider is not None:
            return provider.resolved_api_key(self._env)
        return ""

    def _get_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=self._timeout_seconds)
        return self._http_client

    def _post(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> Any:
        client = self._get_client()
        try:
            return client.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
            )
        except TypeError:
            # Minimal test doubles may not accept a per-request timeout.
            return client.post(url, json=payload, headers=headers)

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        data: Mapping[str, Any],
        ctx: RoutingContext,
        latency_ms: float,
    ) -> JevEvaluation | None:
        answers = _extract_answers(data)
        if not isinstance(answers, Mapping):
            return None

        recognized = any(
            _lookup(answers, name) is not None for name in _QUESTION_ALIASES
        )
        if not recognized:
            return None

        heuristic = evaluate_heuristic_fallback(ctx, reason="partial Jev response")

        intent_answer = _lookup(answers, "task_intent")
        intent, probabilities = _choice_value(intent_answer)
        if intent is None:
            intent = heuristic.task_intent
            probabilities = _probabilities(intent, heuristic.intent_confidence)
        confidence = _answer_confidence(intent_answer, probabilities)
        has_explicit_confidence = isinstance(intent_answer, Mapping) and any(
            key in intent_answer
            for key in ("confidence", "probability", "score", "prob")
        )
        if confidence <= 0 and not has_explicit_confidence:
            if probabilities:
                confidence = max(probabilities.values())
            elif intent_answer is not None:
                confidence = 0.9
            else:
                confidence = heuristic.intent_confidence
        if not probabilities:
            probabilities = _probabilities(intent, confidence)

        level, score = _reasoning_value(_lookup(answers, "reasoning_depth"))
        if level is None:
            level = heuristic.reasoning_level
        if score is None:
            score = heuristic.reasoning_score

        needs_thinking = _noul_value(_lookup(answers, "needs_thinking_trace"))
        destructive = _noul_value(_lookup(answers, "destructive_risk"))
        ast_sufficient = _noul_value(_lookup(answers, "ast_context_sufficient"))

        return JevEvaluation(
            task_intent=intent,
            intent_confidence=_clamp(confidence),
            intent_probabilities={
                option: _clamp(float(probabilities.get(option, 0.0)))
                for option in TASK_INTENT_OPTIONS
            },
            reasoning_level=level,
            reasoning_score=_clamp(score),
            needs_thinking_prob=(
                heuristic.needs_thinking_prob
                if needs_thinking is None
                else needs_thinking
            ),
            destructive_risk_prob=(
                heuristic.destructive_risk_prob
                if destructive is None
                else destructive
            ),
            ast_sufficient_prob=(
                heuristic.ast_sufficient_prob
                if ast_sufficient is None
                else ast_sufficient
            ),
            latency_ms=latency_ms,
            used_fallback=False,
            fallback_reason=None,
        )

    def _record_usage(self, data: Mapping[str, Any], latency_ms: float) -> None:
        if self._tracker is None:
            return
        usage = dict(_find_usage(data) or {})
        usage.setdefault("latency_ms", latency_ms)
        usage.setdefault("total_ms", latency_ms)
        try:
            self._tracker.record(
                model=JEV_PRICING_MODEL,
                usage=usage,
                tier="jev",
                provider=self.jev.provider,
            )
        except Exception:  # noqa: BLE001 - accounting must never break routing
            pass


def fallback_evaluation(
    ctx: RoutingContext,
    reason: str,
) -> JevEvaluation:
    """Small public wrapper used by tests and callers."""
    return replace(evaluate_heuristic_fallback(ctx, reason), used_fallback=True)
