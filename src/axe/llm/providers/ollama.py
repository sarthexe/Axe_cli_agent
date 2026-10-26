"""Native Ollama provider with health checks and local model discovery."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import httpx

from axe.llm.provider import (
    LLMProvider,
    LLMResponse,
    ProviderResponseError,
    ProviderUnavailableError,
    ThinkingMode,
    TokenUsage,
    ToolCall,
    as_text,
    merge_reasoning,
    split_inline_reasoning,
)


class OllamaProvider(LLMProvider):
    """Provider for a local (or remote) Ollama daemon.

    Models are addressed with an ``ollama/`` prefix at the registry layer; this
    class always receives the stripped tag, e.g. ``qwen3-coder:14b``.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        api_key: str = "",
        temperature: float = 0.0,
        timeout: float = 60.0,
        requested_model: str | None = None,
        http_client: httpx.Client | None = None,
        health_timeout: float = 1.5,
        tags_ttl_seconds: float = 5.0,
        name: str = "ollama",
    ) -> None:
        self.name = name
        self.model = model
        self.requested_model = requested_model or f"ollama/{model}"
        self.base_url = self._native_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self._http_client = http_client
        self._health_timeout = health_timeout
        self._tags_ttl_seconds = tags_ttl_seconds
        self._tags_cache: list[str] | None = None
        self._tags_cached_at: float = 0.0
        self._health_cache: tuple[float, bool, str] | None = None

    # ------------------------------------------------------------------
    # Availability / discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _native_base_url(base_url: str) -> str:
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3].rstrip("/")
        return url

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _get_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=self.timeout,
                headers=self._headers(),
            )
        return self._http_client

    def check_health(self) -> bool:
        """Return ``True`` if the Ollama daemon answers ``/api/tags``."""
        now = time.monotonic()
        if self._health_cache is not None:
            cached_at, healthy, _reason = self._health_cache
            if now - cached_at < self._tags_ttl_seconds:
                return healthy

        try:
            response = self._get_client().get(
                f"{self.base_url}/api/tags",
                timeout=self._health_timeout,
            )
            healthy = response.status_code == 200
            reason = "" if healthy else f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001 - health checks must never raise
            healthy = False
            reason = str(exc)

        self._health_cache = (now, healthy, reason)
        return healthy

    def list_local_models(self, *, force: bool = False) -> list[str]:
        """Return installed Ollama model tags, or ``[]`` if unavailable."""
        now = time.monotonic()
        if (
            not force
            and self._tags_cache is not None
            and now - self._tags_cached_at < self._tags_ttl_seconds
        ):
            return list(self._tags_cache)

        try:
            response = self._get_client().get(
                f"{self.base_url}/api/tags",
                timeout=self._health_timeout,
            )
            if response.status_code != 200:
                return []
            payload = response.json()
        except Exception:  # noqa: BLE001 - discovery is best-effort
            return []

        raw_models = payload.get("models") if isinstance(payload, Mapping) else None
        names: list[str] = []
        if isinstance(raw_models, list):
            for item in raw_models:
                if isinstance(item, Mapping):
                    name = item.get("name") or item.get("model")
                    if name:
                        names.append(str(name))
        self._tags_cache = names
        self._tags_cached_at = now
        return list(names)

    def is_model_pulled(self, installed: list[str] | None = None) -> bool:
        names = self.list_local_models() if installed is None else installed
        if not names:
            return False
        if self.model in names or f"{self.model}:latest" in names:
            return True
        if ":" not in self.model:
            return any(name.split(":", 1)[0] == self.model for name in names)
        return False

    def is_available(self) -> tuple[bool, str]:
        if not self.check_health():
            reason = "daemon unreachable"
            if self._health_cache is not None and self._health_cache[2]:
                reason = self._health_cache[2]
            return False, f"Ollama {reason} at {self.base_url}"
        if not self.is_model_pulled():
            return False, (
                f"model '{self.model}' is not pulled locally "
                f"(run: ollama pull {self.model})"
            )
        return True, ""

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
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(prompt, system_prompt, messages),
            "stream": False,
            "think": self._think_value(thinking_mode),
        }
        if tools:
            payload["tools"] = self._build_tools(tools)

        options: dict[str, Any] = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if options:
            payload["options"] = options

        started = time.perf_counter()
        try:
            response = self._post_chat(payload, allow_think_retry=True)
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Ollama is unreachable at {self.base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"Ollama timed out at {self.base_url}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Ollama request failed at {self.base_url}: {exc}"
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code >= 400:
            body = response.text[:300]
            raise ProviderResponseError(
                f"Ollama HTTP {response.status_code} for model '{self.model}': {body}"
            )

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(
                f"Ollama returned invalid JSON for model '{self.model}'."
            ) from exc
        if not isinstance(data, Mapping):
            raise ProviderResponseError("Ollama returned an unexpected response shape.")

        return self._normalize_response(data, elapsed_ms)

    def _post_chat(
        self,
        payload: dict[str, Any],
        *,
        allow_think_retry: bool,
    ) -> httpx.Response:
        client = self._get_client()
        url = f"{self.base_url}/api/chat"
        response = client.post(url, json=payload, headers=self._headers())
        if (
            allow_think_retry
            and response.status_code == 400
            and "think" in payload
        ):
            retry_payload = dict(payload)
            retry_payload.pop("think", None)
            response = client.post(url, json=retry_payload, headers=self._headers())
        return response

    @staticmethod
    def _think_value(thinking_mode: ThinkingMode) -> bool | str:
        if thinking_mode == "off":
            return False
        return thinking_mode

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if messages is None:
            built: list[dict[str, Any]] = []
            if system_prompt:
                built.append({"role": "system", "content": system_prompt})
            built.append({"role": "user", "content": prompt})
            return built

        # Map tool_call_id -> tool name so native tool results include the
        # ``tool_name`` field Ollama expects.
        call_names: dict[str, str] = {}
        for message in messages:
            for call in message.get("tool_calls") or []:
                if not isinstance(call, Mapping):
                    continue
                call_id = str(call.get("id") or "")
                function = call.get("function")
                if call_id and isinstance(function, Mapping) and function.get("name"):
                    call_names[call_id] = str(function["name"])

        normalized: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content")
            if content is None:
                content = ""

            clean: dict[str, Any] = {"role": role, "content": content}
            if message.get("name"):
                clean["name"] = message["name"]

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                clean["tool_calls"] = [
                    self._normalize_tool_call(call, index)
                    for index, call in enumerate(tool_calls)
                    if isinstance(call, Mapping)
                ]

            if role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                tool_name = message.get("tool_name") or call_names.get(call_id)
                if tool_name:
                    clean["tool_name"] = str(tool_name)
                if call_id:
                    clean["tool_call_id"] = call_id

            normalized.append(clean)
        return normalized

    @staticmethod
    def _normalize_tool_call(call: Mapping[str, Any], index: int) -> dict[str, Any]:
        function = call.get("function")
        if not isinstance(function, Mapping):
            function = {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                arguments = dict(parsed) if isinstance(parsed, Mapping) else {}
            except json.JSONDecodeError:
                arguments = {}
        elif isinstance(arguments, Mapping):
            arguments = dict(arguments)
        else:
            arguments = {}
        return {
            "id": str(call.get("id") or f"call_{index}"),
            "type": "function",
            "function": {
                "name": str(function.get("name", "")),
                "arguments": arguments,
            },
        }

    @staticmethod
    def _build_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    # ------------------------------------------------------------------
    # Response normalization
    # ------------------------------------------------------------------

    def _normalize_response(
        self,
        data: Mapping[str, Any],
        elapsed_ms: float,
    ) -> LLMResponse:
        message = data.get("message")
        if not isinstance(message, Mapping):
            raise ProviderResponseError(
                f"Ollama response for model '{self.model}' has no message."
            )

        content = as_text(message.get("content"))
        native_reasoning = as_text(
            message.get("thinking")
            or message.get("reasoning_content")
            or message.get("reasoning")
        )
        clean_content, inline_reasoning = split_inline_reasoning(content)
        reasoning = merge_reasoning(native_reasoning, inline_reasoning)
        tool_calls = self._extract_tool_calls(message.get("tool_calls"))
        usage = self._extract_usage(data, elapsed_ms, reasoning)

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
                        "arguments": call.arguments,
                    },
                }
                for call in tool_calls
            ]

        raw = dict(data)
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

    def _extract_tool_calls(self, raw_calls: Any) -> list[ToolCall]:
        if not isinstance(raw_calls, list):
            return []
        calls: list[ToolCall] = []
        for index, item in enumerate(raw_calls):
            if not isinstance(item, Mapping):
                continue
            function = item.get("function")
            if not isinstance(function, Mapping):
                continue
            name = function.get("name")
            if not name:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    arguments = dict(parsed) if isinstance(parsed, Mapping) else {}
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(arguments, Mapping):
                arguments = dict(arguments)
            else:
                arguments = {}
            calls.append(
                ToolCall(
                    name=str(name),
                    arguments=arguments,
                    id=str(item.get("id") or f"call_{index}"),
                    raw=dict(item),
                )
            )
        return calls

    @staticmethod
    def _extract_usage(
        data: Mapping[str, Any],
        elapsed_ms: float,
        reasoning: str | None,
    ) -> TokenUsage:
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)

        eval_duration_ns = float(data.get("eval_duration") or 0.0)
        total_duration_ns = float(data.get("total_duration") or 0.0)
        load_duration_ns = float(data.get("load_duration") or 0.0)
        prompt_eval_ns = float(data.get("prompt_eval_duration") or 0.0)

        if eval_duration_ns > 0 and completion_tokens > 0:
            tokens_per_sec = completion_tokens / (eval_duration_ns / 1_000_000_000)
        elif elapsed_ms > 0 and completion_tokens > 0:
            tokens_per_sec = completion_tokens / (elapsed_ms / 1000.0)
        else:
            tokens_per_sec = 0.0

        total_ms = total_duration_ns / 1_000_000 if total_duration_ns > 0 else elapsed_ms
        ttft_ms = (
            (load_duration_ns + prompt_eval_ns) / 1_000_000
            if (load_duration_ns + prompt_eval_ns) > 0
            else 0.0
        )

        reported_reasoning_tokens = int(data.get("reasoning_eval_count") or 0)
        if reported_reasoning_tokens == 0 and reasoning:
            # Ollama does not tokenize reasoning separately. Approximate only
            # for display, never for billing (local inference is unpriced).
            reported_reasoning_tokens = max(1, len(reasoning) // 4)

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=0,
            reasoning_tokens=reported_reasoning_tokens,
            latency_ms=elapsed_ms,
            total_ms=total_ms,
            ttft_ms=ttft_ms,
            tokens_per_sec=tokens_per_sec,
        )
