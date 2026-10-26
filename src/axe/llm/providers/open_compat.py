"""OpenAI-compatible provider for hosted and local open-weight endpoints.

Covers OpenRouter, DeepSeek, Zhipu/GLM, Moonshot/Kimi, Fireworks, vLLM, and
llama.cpp. Each instance is bound to a single model so the registry can resolve
per-model ``base_url`` / API key combinations on the fly.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import openai
from openai import OpenAI

from axe.llm.provider import (
    LLMProvider,
    LLMResponse,
    MissingProviderKeyError,
    ProviderResponseError,
    ThinkingMode,
    TokenUsage,
    ToolCall,
    as_text,
    merge_reasoning,
    split_inline_reasoning,
)

# Providers that run locally and therefore do not require an API key.
_LOCAL_PROVIDERS = {"local_vllm", "llamacpp"}

# Message fields accepted by OpenAI-compatible chat endpoints.
_OPENAI_MESSAGE_FIELDS = {"role", "content", "name", "tool_call_id", "tool_calls"}


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
        except Exception:  # pragma: no cover - defensive
            return {}
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _nested_int(raw: Mapping[str, Any], parent: str, child: str) -> int:
    nested = raw.get(parent)
    if isinstance(nested, Mapping):
        try:
            return int(nested.get(child, 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _message_field(message: Any, name: str) -> Any:
    value = getattr(message, name, None)
    if value not in (None, ""):
        return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, Mapping) and name in extra:
        return extra[name]
    dumped = _to_mapping(message)
    return dumped.get(name)


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from a pydantic object or mapping."""
    result = getattr(value, name, None)
    if result is None and isinstance(value, Mapping):
        result = value.get(name)
    return default if result is None else result


class OpenCompatProvider(LLMProvider):
    """Provider for OpenAI-compatible REST endpoints."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        api_key: str = "",
        api_key_env: str = "",
        default_headers: Mapping[str, str] | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        requested_model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.requested_model = requested_model or model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.timeout = timeout
        self.default_headers = dict(default_headers or {})
        self._local = name in _LOCAL_PROVIDERS
        self._client = client

    # ------------------------------------------------------------------
    # Availability / client lifecycle
    # ------------------------------------------------------------------

    def is_available(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, f"provider '{self.name}' has no base_url configured"
        if not self._local and not self.api_key:
            hint = self.api_key_env or "API key"
            return False, f"missing {hint} for provider '{self.name}'"
        return True, ""

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._local and not self.api_key:
            hint = self.api_key_env or "API key"
            raise MissingProviderKeyError(
                f"Missing {hint} for provider '{self.name}'. Set it in .env or "
                f"providers.{self.name}.api_key."
            )
        self._client = OpenAI(
            api_key=self.api_key or "local-no-key",
            base_url=self.base_url,
            timeout=self.timeout,
            default_headers=self.default_headers or None,
        )
        return self._client

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        thinking_mode: ThinkingMode = "low",
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        input_messages = self._build_messages(prompt, system_prompt, messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": input_messages,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = self._build_tools(tools)

        thinking_body = self._thinking_body(thinking_mode)
        if thinking_body:
            payload["extra_body"] = thinking_body

        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        started = time.perf_counter()
        response = self._create(client, payload)
        if stream:
            return self._normalize_stream(response, started)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return self._normalize_response(response, elapsed_ms)

    def _create(self, client: Any, payload: dict[str, Any]) -> Any:
        """Create a completion, retrying once without unsupported extras."""
        try:
            return client.chat.completions.create(**payload)
        except openai.BadRequestError as exc:
            retry = dict(payload)
            changed = False
            message = str(exc).lower()
            if "stream_options" in retry and "stream_options" in message:
                retry.pop("stream_options", None)
                changed = True
            if "temperature" in retry and "temperature" in message:
                retry.pop("temperature", None)
                changed = True
            if "extra_body" in retry and self._is_unsupported_param_error(exc):
                retry.pop("extra_body", None)
                changed = True
            if not changed:
                raise
            return client.chat.completions.create(**retry)

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if messages is not None:
            return [self._normalize_message(message) for message in messages]

        built: list[dict[str, Any]] = []
        if system_prompt:
            built.append({"role": "system", "content": system_prompt})
        built.append({"role": "user", "content": prompt})
        return built

    def _normalize_message(self, message: Mapping[str, Any]) -> dict[str, Any]:
        clean = {
            key: value
            for key, value in message.items()
            if key in _OPENAI_MESSAGE_FIELDS
        }
        tool_calls = clean.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_calls: list[dict[str, Any]] = []
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    continue
                call_copy = dict(call)
                function = call_copy.get("function")
                if isinstance(function, Mapping):
                    function_copy = dict(function)
                    arguments = function_copy.get("arguments")
                    if isinstance(arguments, Mapping):
                        function_copy["arguments"] = json.dumps(arguments)
                    call_copy["function"] = function_copy
                normalized_calls.append(call_copy)
            clean["tool_calls"] = normalized_calls
        return clean

    def _build_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for tool in tools:
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            spec: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description", "")),
                },
            }
            parameters = tool.get("parameters")
            if isinstance(parameters, Mapping):
                spec["function"]["parameters"] = dict(parameters)
            prepared.append(spec)
        return prepared

    def _thinking_body(self, thinking_mode: ThinkingMode) -> dict[str, Any]:
        """Map the normalized thinking mode onto provider-specific parameters."""
        model = self.model.lower()
        provider = self.name.lower()

        if provider == "openrouter":
            effort = "none" if thinking_mode == "off" else thinking_mode
            return {"reasoning": {"effort": effort}}

        if provider in {"zhipu", "deepseek", "moonshot"}:
            enabled = thinking_mode != "off"
            return {"thinking": {"type": "enabled" if enabled else "disabled"}}

        if provider in {"local_vllm", "llamacpp"} or "qwen" in model:
            return {"enable_thinking": thinking_mode != "off"}

        # Unknown strict endpoint: send nothing provider-specific.
        return {}

    @staticmethod
    def _is_unsupported_param_error(exc: openai.BadRequestError) -> bool:
        message = str(exc).lower()
        needles = (
            "extra_body",
            "reasoning",
            "thinking",
            "enable_thinking",
            "unsupported parameter",
            "unknown parameter",
            "unrecognized",
        )
        return any(needle in message for needle in needles)

    # ------------------------------------------------------------------
    # Streaming normalization
    # ------------------------------------------------------------------

    def _normalize_stream(self, stream: Any, started: float) -> LLMResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_accumulator: dict[int, dict[str, str]] = {}
        usage_obj: Any = None
        ttft_ms = 0.0

        for chunk in stream:
            usage = _field(chunk, "usage")
            if usage is not None:
                usage_obj = usage

            choices = _field(chunk, "choices", []) or []
            if not choices:
                continue
            delta = _field(choices[0], "delta")
            if delta is None:
                continue

            content = as_text(_field(delta, "content"))
            reasoning = as_text(_field(delta, "reasoning_content")) or as_text(
                _field(delta, "reasoning")
            )
            tool_deltas = _field(delta, "tool_calls", []) or []
            if (content or reasoning or tool_deltas) and ttft_ms <= 0:
                ttft_ms = (time.perf_counter() - started) * 1000.0

            if content:
                content_parts.append(content)
            if reasoning:
                reasoning_parts.append(reasoning)
            self._accumulate_tool_call_deltas(tool_accumulator, tool_deltas)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        content = "".join(content_parts)
        clean_content, inline_reasoning = split_inline_reasoning(content)
        reasoning = merge_reasoning("".join(reasoning_parts), inline_reasoning)
        tool_calls = self._tool_calls_from_deltas(tool_accumulator)
        usage = self._extract_usage(usage_obj, elapsed_ms)
        usage.ttft_ms = ttft_ms
        generation_ms = elapsed_ms - ttft_ms
        if usage.completion_tokens > 0 and generation_ms > 0:
            usage.tokens_per_sec = usage.completion_tokens / (generation_ms / 1000.0)

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": clean_content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in tool_calls
            ]

        return LLMResponse(
            content=clean_content,
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            model=self.requested_model,
            provider=self.name,
            usage=usage,
            raw={
                "stream": True,
                "message": assistant_message,
                "usage": _to_mapping(usage_obj),
            },
        )

    @staticmethod
    def _accumulate_tool_call_deltas(
        accumulator: dict[int, dict[str, str]],
        deltas: list[Any],
    ) -> None:
        for item in deltas:
            try:
                index = int(_field(item, "index", 0) or 0)
            except (TypeError, ValueError):
                index = 0
            entry = accumulator.setdefault(
                index, {"id": "", "name": "", "arguments": ""}
            )

            call_id = _field(item, "id")
            if call_id:
                entry["id"] = str(call_id)

            function = _field(item, "function") or {}
            name = _field(function, "name")
            if name:
                entry["name"] += str(name)
            arguments = _field(function, "arguments")
            if arguments:
                if isinstance(arguments, Mapping):
                    entry["arguments"] += json.dumps(dict(arguments))
                else:
                    entry["arguments"] += str(arguments)

    @staticmethod
    def _tool_calls_from_deltas(
        accumulator: dict[int, dict[str, str]],
    ) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index in sorted(accumulator):
            entry = accumulator[index]
            name = entry["name"].strip()
            if not name:
                continue
            arguments: dict[str, Any] = {}
            raw_arguments = entry["arguments"].strip()
            if raw_arguments:
                try:
                    parsed = json.loads(raw_arguments)
                    if isinstance(parsed, Mapping):
                        arguments = dict(parsed)
                except json.JSONDecodeError:
                    arguments = {}
            calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    id=entry["id"] or f"call_{index}",
                    raw=dict(entry),
                )
            )
        return calls

    # ------------------------------------------------------------------
    # Response normalization
    # ------------------------------------------------------------------

    def _normalize_response(self, response: Any, elapsed_ms: float) -> LLMResponse:
        choices = getattr(response, "choices", None) or _to_mapping(response).get("choices")
        if not choices:
            raise ProviderResponseError(
                f"Provider '{self.name}' returned no choices for model '{self.model}'."
            )
        message = getattr(choices[0], "message", None)
        if message is None and isinstance(choices[0], Mapping):
            message = choices[0].get("message")
        if message is None:
            raise ProviderResponseError(
                f"Provider '{self.name}' returned a response without a message."
            )

        content = as_text(_message_field(message, "content"))
        native_reasoning = as_text(
            _message_field(message, "reasoning_content")
            or _message_field(message, "reasoning")
            or _message_field(message, "thinking")
        )
        clean_content, inline_reasoning = split_inline_reasoning(content)
        reasoning = merge_reasoning(native_reasoning, inline_reasoning)
        tool_calls = self._extract_tool_calls(message)
        usage = self._extract_usage(getattr(response, "usage", None), elapsed_ms)

        raw = _to_mapping(response)
        assistant_message = _to_mapping(message)
        assistant_message.pop("reasoning_content", None)
        assistant_message.pop("reasoning", None)
        assistant_message.pop("thinking", None)
        assistant_message.setdefault("role", "assistant")
        assistant_message["content"] = clean_content
        raw["message"] = assistant_message

        return LLMResponse(
            content=clean_content,
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            model=self.requested_model,
            provider=self.name,
            usage=usage,
            raw=raw,
        )

    def _extract_tool_calls(self, message: Any) -> list[ToolCall]:
        raw_calls = getattr(message, "tool_calls", None)
        if raw_calls is None:
            raw_calls = _to_mapping(message).get("tool_calls")
        if not raw_calls:
            return []

        calls: list[ToolCall] = []
        for index, item in enumerate(raw_calls):
            item_type = getattr(item, "type", None)
            if item_type is None and isinstance(item, Mapping):
                item_type = item.get("type")
            if item_type not in (None, "function"):
                continue

            function = getattr(item, "function", None)
            if function is None and isinstance(item, Mapping):
                function = item.get("function")
            if function is None:
                continue

            name = getattr(function, "name", None)
            raw_arguments = getattr(function, "arguments", None)
            if isinstance(function, Mapping):
                name = function.get("name", name)
                raw_arguments = function.get("arguments", raw_arguments)
            if not name:
                continue

            arguments: dict[str, Any] = {}
            if isinstance(raw_arguments, str):
                try:
                    parsed = json.loads(raw_arguments)
                    if isinstance(parsed, Mapping):
                        arguments = dict(parsed)
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(raw_arguments, Mapping):
                arguments = dict(raw_arguments)

            call_id = getattr(item, "id", None)
            if call_id is None and isinstance(item, Mapping):
                call_id = item.get("id")

            calls.append(
                ToolCall(
                    name=str(name),
                    arguments=arguments,
                    id=str(call_id or f"call_{index}"),
                    raw=_to_mapping(item),
                )
            )
        return calls

    @staticmethod
    def _extract_usage(usage: Any, elapsed_ms: float) -> TokenUsage:
        raw = _to_mapping(usage)

        # Some SDK builds keep provider-specific usage fields (DeepSeek cache
        # counters) outside model_dump(); read them directly as a fallback.
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "prompt_cache_hit_tokens",
            "cache_read_input_tokens",
            "reasoning_tokens",
        ):
            if not raw.get(name):
                value = getattr(usage, name, None)
                if value is not None:
                    raw[name] = value
        if not isinstance(raw.get("prompt_tokens_details"), Mapping):
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                raw["prompt_tokens_details"] = _to_mapping(details)
        if not isinstance(raw.get("completion_tokens_details"), Mapping):
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                raw["completion_tokens_details"] = _to_mapping(details)

        prompt_tokens = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
        completion_tokens = int(
            raw.get("completion_tokens") or raw.get("output_tokens") or 0
        )
        cached_tokens = (
            _nested_int(raw, "prompt_tokens_details", "cached_tokens")
            or _nested_int(raw, "prompt_tokens_details", "cache_read_tokens")
            or int(raw.get("prompt_cache_hit_tokens") or 0)
            or int(raw.get("cache_read_input_tokens") or 0)
        )
        reasoning_tokens = (
            _nested_int(raw, "completion_tokens_details", "reasoning_tokens")
            or int(raw.get("reasoning_tokens") or 0)
        )
        tokens_per_sec = (
            completion_tokens / (elapsed_ms / 1000.0)
            if elapsed_ms > 0 and completion_tokens > 0
            else 0.0
        )
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=elapsed_ms,
            total_ms=elapsed_ms,
            tokens_per_sec=tokens_per_sec,
        )
