"""Tests for multi-provider resolution, reasoning normalization, and fallback."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from rich.console import Console

from axe.config.settings import Settings, load_settings
from axe.llm.provider import (
    AllProvidersFailedError,
    LLMProvider,
    LLMResponse,
    ProviderUnavailableError,
    TokenUsage,
    UnknownProviderError,
    split_inline_reasoning,
)
from axe.llm.providers import OllamaProvider, OpenCompatProvider, ProviderRegistry
from axe.llm.providers.registry import ResolvedModel
from axe.llm.router import ModelRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(env: dict[str, str] | None = None) -> Settings:
    return load_settings(
        env=env or {},
        include_user=False,
        include_workspace=False,
    )


def _dump(value: Any) -> Any:
    if isinstance(value, FakeDumpable):
        return value.model_dump()
    if isinstance(value, list):
        return [_dump(item) for item in value]
    return value


class FakeDumpable:
    """Minimal pydantic-like object with ``model_dump``."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            key: _dump(value)
            for key, value in self.__dict__.items()
            if not callable(value)
        }


class FakeCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.completions = FakeCompletions(response)
        self.chat = FakeDumpable(completions=self.completions)


class FakeStreamCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return iter(self.chunks)


class FakeStreamClient:
    def __init__(self, chunks: list[Any]) -> None:
        self.completions = FakeStreamCompletions(chunks)
        self.chat = FakeDumpable(completions=self.completions)


def _stream_chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[Any] | None = None,
    usage: Any = None,
) -> Any:
    delta = FakeDumpable(content=content)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    if tool_calls is not None:
        delta.tool_calls = tool_calls
    choices = [FakeDumpable(delta=delta)] if (content or reasoning or tool_calls) else []
    return FakeDumpable(choices=choices, usage=usage)


def _openai_response(
    *,
    content: str | None = "hello",
    reasoning_content: str | None = None,
    reasoning: str | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    cached_tokens: int = 40,
    reasoning_tokens: int = 5,
) -> Any:
    message = FakeDumpable(role="assistant", content=content, tool_calls=[])
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    if reasoning is not None:
        message.reasoning = reasoning
    usage = FakeDumpable(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=FakeDumpable(cached_tokens=cached_tokens),
        completion_tokens_details=FakeDumpable(
            reasoning_tokens=reasoning_tokens
        ),
    )
    return FakeDumpable(
        choices=[FakeDumpable(message=message)],
        usage=usage,
    )


def _opencompat(
    client: FakeClient,
    *,
    name: str = "openrouter",
    model: str = "deepseek/deepseek-v4",
    api_key: str = "test-key",
) -> OpenCompatProvider:
    return OpenCompatProvider(
        name=name,
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        client=client,
        requested_model=model,
    )


class FakeHTTPResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeHTTPClient:
    def __init__(
        self,
        *,
        get_response: FakeHTTPResponse | None = None,
        post_response: FakeHTTPResponse | None = None,
        get_error: Exception | None = None,
    ) -> None:
        self.get_response = get_response
        self.post_response = post_response
        self.get_error = get_error
        self.get_calls: list[str] = []
        self.post_calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, timeout: float | None = None) -> FakeHTTPResponse:
        del timeout
        self.get_calls.append(url)
        if self.get_error is not None:
            raise self.get_error
        assert self.get_response is not None
        return self.get_response

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> FakeHTTPResponse:
        del headers
        self.post_calls.append((url, json or {}))
        assert self.post_response is not None
        return self.post_response


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


def test_deepseek_routes_direct_when_key_is_configured() -> None:
    test_key = "sk-" + "deepseek"
    settings = _settings({"DEEPSEEK_API_KEY": test_key})
    resolved = ProviderRegistry(settings).resolve("deepseek/deepseek-v4")

    assert resolved.provider_name == "deepseek"
    assert resolved.model == "deepseek-v4"
    assert isinstance(resolved.provider, OpenCompatProvider)
    assert resolved.provider.base_url == "https://api.deepseek.com/v1"
    assert resolved.provider.api_key == test_key


def test_deepseek_routes_via_openrouter_without_key() -> None:
    resolved = ProviderRegistry(_settings()).resolve("deepseek/deepseek-v4")

    assert resolved.provider_name == "openrouter"
    assert resolved.model == "deepseek/deepseek-v4"
    assert isinstance(resolved.provider, OpenCompatProvider)
    assert resolved.provider.base_url == "https://openrouter.ai/api/v1"


def test_ollama_prefix_routes_to_native_provider() -> None:
    resolved = ProviderRegistry(_settings()).resolve("ollama/qwen3-coder:14b")

    assert resolved.provider_name == "ollama"
    assert resolved.model == "qwen3-coder:14b"
    assert resolved.requested == "ollama/qwen3-coder:14b"
    assert isinstance(resolved.provider, OllamaProvider)
    assert resolved.provider.base_url == "http://localhost:11434"


def test_vllm_prefix_routes_to_local_open_compat_provider() -> None:
    resolved = ProviderRegistry(_settings()).resolve("vllm/qwen3.8-27b")

    assert resolved.provider_name == "local_vllm"
    assert resolved.model == "qwen3.8-27b"
    assert isinstance(resolved.provider, OpenCompatProvider)
    assert resolved.availability() == (True, "")


def test_explicit_provider_colon_syntax() -> None:
    resolved = ProviderRegistry(_settings()).resolve("zhipu:glm-5.3")

    assert resolved.provider_name == "zhipu"
    assert resolved.model == "glm-5.3"
    assert resolved.provider.base_url == "https://open.bigmodel.cn/api/paas/v4"


def test_unknown_vendor_defaults_to_openrouter() -> None:
    resolved = ProviderRegistry(_settings()).resolve("qwen/qwen3-coder-next")

    assert resolved.provider_name == "openrouter"
    assert resolved.model == "qwen/qwen3-coder-next"


# ---------------------------------------------------------------------------
# Reasoning + usage normalization
# ---------------------------------------------------------------------------


def test_split_inline_reasoning_handles_multiple_blocks() -> None:
    clean, reasoning = split_inline_reasoning(
        "before<think>first</think>middle<thinking>second</thinking>after"
    )

    assert clean == "beforemiddleafter"
    assert reasoning == "first\n\nsecond"


def test_native_reasoning_and_cache_usage_are_normalized() -> None:
    client = FakeClient(_openai_response(reasoning_content="native reasoning"))
    provider = _opencompat(client)

    response = provider.complete("hi", thinking_mode="high", max_tokens=256)

    assert response.content == "hello"
    assert response.reasoning_content == "native reasoning"
    assert response.model == "deepseek/deepseek-v4"
    assert response.provider == "openrouter"
    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 20
    assert response.usage.cached_prompt_tokens == 40
    assert response.usage.reasoning_tokens == 5
    assert response.usage.latency_ms > 0
    assert response.usage.tokens_per_sec > 0
    assert "reasoning_content" not in response.raw["message"]

    payload = client.completions.calls[0]
    assert payload["max_tokens"] == 256
    assert payload["extra_body"] == {"reasoning": {"effort": "high"}}


def test_deepseek_prompt_cache_hit_tokens_are_extracted() -> None:
    message = FakeDumpable(role="assistant", content="ok", tool_calls=[])
    usage = FakeDumpable(
        prompt_tokens=100,
        completion_tokens=10,
        prompt_cache_hit_tokens=60,
        completion_tokens_details=FakeDumpable(reasoning_tokens=3),
    )
    response_obj = FakeDumpable(
        choices=[FakeDumpable(message=message)],
        usage=usage,
    )
    client = FakeClient(response_obj)
    provider = _opencompat(client, name="deepseek", model="deepseek-v4")

    response = provider.complete("hi")

    assert response.usage.prompt_tokens == 100
    assert response.usage.cached_prompt_tokens == 60
    assert response.usage.reasoning_tokens == 3


def test_inline_think_tags_are_stripped_from_content() -> None:
    client = FakeClient(_openai_response(content="<think>hidden</think>visible"))
    provider = _opencompat(client)

    response = provider.complete("hi")

    assert response.content == "visible"
    assert response.reasoning_content == "hidden"


def test_native_and_inline_reasoning_are_merged() -> None:
    client = FakeClient(
        _openai_response(
            content="<think>inline</think>answer",
            reasoning_content="native",
        )
    )
    provider = _opencompat(client)

    response = provider.complete("hi")

    assert response.content == "answer"
    assert response.reasoning_content == "native\n\ninline"


def test_thinking_off_uses_provider_specific_disable_flag() -> None:
    openrouter_client = FakeClient(_openai_response())
    _opencompat(openrouter_client).complete("hi", thinking_mode="off")
    assert openrouter_client.completions.calls[0]["extra_body"] == {
        "reasoning": {"effort": "none"}
    }

    vllm_client = FakeClient(_openai_response())
    vllm = _opencompat(vllm_client, name="local_vllm", model="qwen3-coder")
    vllm.complete("hi", thinking_mode="off")
    assert vllm_client.completions.calls[0]["extra_body"] == {
        "enable_thinking": False
    }


def test_streaming_normalizes_reasoning_and_tool_call_deltas() -> None:
    usage = FakeDumpable(
        prompt_tokens=10,
        completion_tokens=5,
        prompt_tokens_details=FakeDumpable(cached_tokens=4),
        completion_tokens_details=FakeDumpable(reasoning_tokens=2),
    )
    chunks = [
        _stream_chunk(content="Hel", reasoning="why"),
        _stream_chunk(
            content="lo",
            tool_calls=[
                FakeDumpable(
                    index=0,
                    id="call_1",
                    function=FakeDumpable(
                        name="shell",
                        arguments='{"command":',
                    ),
                )
            ],
        ),
        _stream_chunk(
            tool_calls=[
                FakeDumpable(
                    index=0,
                    function=FakeDumpable(arguments='"ls"}'),
                )
            ]
        ),
        _stream_chunk(usage=usage),
    ]
    client = FakeStreamClient(chunks)
    provider = _opencompat(client)

    response = provider.complete("hi", stream=True, thinking_mode="high")

    assert response.content == "Hello"
    assert response.reasoning_content == "why"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "shell"
    assert response.tool_calls[0].arguments == {"command": "ls"}
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.cached_prompt_tokens == 4
    assert response.usage.reasoning_tokens == 2
    assert response.usage.ttft_ms >= 0
    assert response.usage.latency_ms > 0
    assert response.usage.tokens_per_sec > 0

    payload = client.completions.calls[0]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# Ollama discovery + throughput
# ---------------------------------------------------------------------------


def test_ollama_health_and_model_discovery() -> None:
    http = FakeHTTPClient(
        get_response=FakeHTTPResponse(
            200,
            {"models": [{"name": "qwen3-coder:14b"}, {"name": "llama3:8b"}]},
        )
    )
    provider = OllamaProvider(model="qwen3-coder:14b", http_client=http)

    assert provider.check_health() is True
    assert provider.list_local_models() == ["qwen3-coder:14b", "llama3:8b"]
    assert provider.is_available() == (True, "")


def test_ollama_missing_model_is_unavailable() -> None:
    http = FakeHTTPClient(
        get_response=FakeHTTPResponse(200, {"models": [{"name": "llama3:8b"}]})
    )
    provider = OllamaProvider(model="qwen3-coder:14b", http_client=http)

    available, reason = provider.is_available()

    assert available is False
    assert "not pulled" in reason


def test_ollama_daemon_down_is_unavailable() -> None:
    http = FakeHTTPClient(get_error=httpx.ConnectError("connection refused"))
    provider = OllamaProvider(model="qwen3-coder:14b", http_client=http)

    assert provider.check_health() is False
    available, reason = provider.is_available()
    assert available is False
    assert "connection refused" in reason


def test_ollama_completion_parses_reasoning_and_throughput() -> None:
    payload = {
        "message": {
            "role": "assistant",
            "content": "<think>inline</think>answer",
            "thinking": "native",
        },
        "prompt_eval_count": 12,
        "eval_count": 6,
        "eval_duration": 1_000_000_000,
        "total_duration": 2_000_000_000,
        "load_duration": 100_000_000,
        "prompt_eval_duration": 400_000_000,
    }
    http = FakeHTTPClient(post_response=FakeHTTPResponse(200, payload))
    provider = OllamaProvider(
        model="qwen3-coder:14b",
        requested_model="ollama/qwen3-coder:14b",
        http_client=http,
    )

    response = provider.complete("hi", thinking_mode="off", max_tokens=256)

    assert response.content == "answer"
    assert response.reasoning_content == "native\n\ninline"
    assert response.model == "ollama/qwen3-coder:14b"
    assert response.provider == "ollama"
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 6
    assert response.usage.cached_prompt_tokens == 0
    assert response.usage.reasoning_tokens > 0
    assert response.usage.tokens_per_sec == pytest.approx(6.0)
    assert response.usage.total_ms == pytest.approx(2000.0)
    assert response.usage.ttft_ms == pytest.approx(500.0)

    _url, request = http.post_calls[0]
    assert request["think"] is False
    assert request["options"]["num_predict"] == 256


# ---------------------------------------------------------------------------
# Cross-provider fallback
# ---------------------------------------------------------------------------


class FakeProvider(LLMProvider):
    def __init__(
        self,
        name: str,
        *,
        available: bool = True,
        reason: str = "",
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.model = name
        self._available = available
        self._reason = reason
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def is_available(self) -> tuple[bool, str]:
        return self._available, self._reason

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        if self._error is not None:
            raise self._error
        return LLMResponse(
            content="ok",
            model=self.model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
        )


class FakeRegistry:
    def __init__(self, providers: dict[str, FakeProvider]) -> None:
        self.providers = providers

    def resolve(self, model: str) -> ResolvedModel:
        provider = self.providers.get(model)
        if provider is None:
            raise UnknownProviderError(f"no provider for {model}")
        return ResolvedModel(
            requested=model,
            provider_name=provider.name,
            model=model,
            provider=provider,
        )


def _fallback_router(
    providers: dict[str, FakeProvider],
    *,
    tier: str = "coder",
    model_override: str | None = None,
) -> tuple[ModelRouter, Settings]:
    settings = _settings()
    settings.router.tiers.coder.models = ["first/model", "second/model"]
    settings.router.tiers.coder.thinking_mode = "low"
    settings.router.tiers.coder.max_tokens = 8192
    router = ModelRouter(
        FakeRegistry(providers),  # type: ignore[arg-type]
        settings,
        console=Console(quiet=True),
        initial_tier=tier,
        model_override=model_override,
    )
    return router, settings


def test_fallback_skips_unavailable_provider() -> None:
    first = FakeProvider(
        "first", available=False, reason="missing FIRST_API_KEY"
    )
    second = FakeProvider("second")
    router, _settings_obj = _fallback_router(
        {"first/model": first, "second/model": second}
    )

    response = router.complete("hello")

    assert response.content == "ok"
    assert first.calls == []
    assert len(second.calls) == 1
    assert second.calls[0]["thinking_mode"] == "low"
    assert second.calls[0]["max_tokens"] == 8192
    assert router.current_model == "second/model"
    assert router.current_provider == "second"


def test_fallback_after_provider_error() -> None:
    first = FakeProvider("first", error=ProviderUnavailableError("rate limited"))
    second = FakeProvider("second")
    router, _ = _fallback_router(
        {"first/model": first, "second/model": second}
    )

    response = router.complete("hello")

    assert response.provider == "second"
    assert len(first.calls) == 1
    assert len(second.calls) == 1


def test_all_candidates_failing_raises_with_details() -> None:
    first = FakeProvider("first", available=False, reason="offline")
    second = FakeProvider("second", error=ProviderUnavailableError("boom"))
    router, _ = _fallback_router(
        {"first/model": first, "second/model": second}
    )

    with pytest.raises(AllProvidersFailedError) as exc_info:
        router.complete("hello")

    assert exc_info.value.tier == "coder"
    assert len(exc_info.value.errors) == 2
    assert "first/model" in str(exc_info.value)
    assert "second/model" in str(exc_info.value)


def test_select_tier_hook_accepts_context_and_model_override() -> None:
    router, settings = _fallback_router(
        {"first/model": FakeProvider("first")},
        tier="auto",
    )
    decision = router.select_tier({"tier": "architect"})
    assert decision.tier == "architect"
    assert decision.source == "forced"
    assert decision.models == settings.router.tiers["architect"].models

    override_router, _ = _fallback_router(
        {"first/model": FakeProvider("first")},
        model_override="override/model",
    )
    assert override_router.current_model == "override/model"
    assert override_router.decision.models[0] == "override/model"
