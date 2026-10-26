"""Tests for Airlock sanitization, MCP client, and default fallback wiring."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from axe.agent.core import Agent
from axe.cli import repl as repl_module
from axe.config.settings import Settings, load_settings
from axe.llm.jev_decision import JevClient, RoutingContext
from axe.llm.provider import LLMProvider, LLMResponse, TokenUsage, ToolCall
from axe.llm.providers import OpenCompatProvider
from axe.llm.providers.registry import ResolvedModel
from axe.llm.router import ModelRouter
from axe.mcp.bridge import register_mcp_tools
from axe.mcp.client import (
    MCPClientManager,
    MCPHTTPClient,
    MCPStdioClient,
)
from axe.mcp.default_servers import AirlockScanTool, setup_default_servers
from axe.security.airlock_middleware import AirlockMiddleware
from axe.security.secret_vault import SecretVault
from axe.tools.file_read import FileReadTool
from axe.tools.file_write import FileWriteTool
from axe.tools.registry import ToolRegistry

SECRET = "sk-" + "or-v1-" + "9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c"
PLACEHOLDER = "<AIRLOCK_SECRET_001>"


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _settings() -> Settings:
    return load_settings(
        env={},
        include_user=False,
        include_workspace=False,
    )


def _dump(value: Any) -> Any:
    if isinstance(value, Dumpable):
        return value.model_dump()
    if isinstance(value, list):
        return [_dump(item) for item in value]
    return value


class Dumpable:
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


class FakeOpenAIClient:
    def __init__(self, response: Any) -> None:
        self.completions = FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def _openai_response() -> Any:
    message = Dumpable(role="assistant", content="ok", tool_calls=[])
    usage = Dumpable(prompt_tokens=10, completion_tokens=2)
    return Dumpable(
        choices=[Dumpable(message=message)],
        usage=usage,
    )


class OneProviderRegistry:
    def __init__(self, provider: OpenCompatProvider) -> None:
        self.provider = provider

    def resolve(self, model: str) -> ResolvedModel:
        return ResolvedModel(
            requested=model,
            provider_name="openrouter",
            model=model,
            provider=self.provider,
        )


class FakeHTTPResponse:
    def __init__(self, status_code: int = 200, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class FakeHTTPClient:
    def __init__(self, response: FakeHTTPResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeHTTPResponse:
        self.calls.append(
            {
                "url": url,
                "json": json or {},
                "headers": headers or {},
                "timeout": timeout,
            }
        )
        return self.response


class ScriptedProvider(LLMProvider):
    name = "fake"
    model = "fake/model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Vault and middleware
# ---------------------------------------------------------------------------


def test_vault_registration_is_deterministic_across_sources() -> None:
    vault = SecretVault()
    first = vault.register(SECRET, "api_key", "builtin-airlock-cascade", ".env")
    second = vault.register(SECRET, "api_key", "builtin-airlock-cascade", "config.py")

    assert first == PLACEHOLDER
    assert second == PLACEHOLDER
    assert len(vault) == 1
    assert vault.stats()["active_placeholders"] == 1
    assert vault.stats()["total_secrets_redacted"] == 2
    restored, count = vault.restore_text(f"key={PLACEHOLDER}")
    assert restored == f"key={SECRET}"
    assert count == 1


def test_middleware_sanitizes_messages_and_restores_tool_args() -> None:
    middleware = AirlockMiddleware(_settings())
    original = [
        {
            "role": "tool",
            "content": f"OPENROUTER_API_KEY={SECRET}",
            "nested": [f"Authorization: Bearer {SECRET}"],
        }
    ]

    sanitized = middleware.sanitize_messages(original)

    serialized = json.dumps(sanitized)
    assert SECRET not in serialized
    assert PLACEHOLDER in serialized
    assert SECRET in original[0]["content"]  # deep copy, unchanged source

    restored = middleware.restore_inbound_tool_call(
        "file_write",
        {"path": "config.py", "content": f'KEY = "{PLACEHOLDER}"'},
    )
    assert restored["content"] == f'KEY = "{SECRET}"'

    edited = middleware.restore_inbound_tool_call(
        "file_edit",
        {
            "path": "config.py",
            "target": f'"{PLACEHOLDER}"',
            "replacement": f'"{PLACEHOLDER}"\n# rotated',
        },
    )
    assert SECRET in edited["target"]
    assert SECRET in edited["replacement"]

    shell = middleware.restore_inbound_tool_call(
        "shell",
        {"command": f"echo {PLACEHOLDER}"},
    )
    assert shell["command"] == f"echo {SECRET}"

    nested, count = middleware.vault.restore_tool_args(
        {"items": [{"value": f"Bearer {PLACEHOLDER}"}]}
    )
    assert nested["items"][0]["value"] == f"Bearer {SECRET}"
    assert count == 1


def test_file_read_redaction_and_file_write_restoration_round_trip(
    tmp_path: Path,
) -> None:
    settings = _settings()
    middleware = AirlockMiddleware(settings)
    source = tmp_path / ".env"
    source.write_text(f'OPENROUTER_API_KEY="{SECRET}"\n', encoding="utf-8")

    read_output = FileReadTool(settings.context).execute({"path": str(source)})
    result = middleware.sanitize_outbound(read_output, source="file_read")
    assert result.sanitized_text.count(PLACEHOLDER) == 1
    assert SECRET not in result.sanitized_text

    target = tmp_path / "config.py"
    args = middleware.restore_inbound_tool_call(
        "file_write",
        {
            "path": str(target),
            "content": f'API_KEY = "{PLACEHOLDER}"\n',
        },
    )
    FileWriteTool().execute(args)

    assert target.read_text(encoding="utf-8") == f'API_KEY = "{SECRET}"\n'


def test_pii_and_secret_stats_never_expose_raw_values() -> None:
    middleware = AirlockMiddleware(_settings())
    result = middleware.sanitize_outbound(
        f"email dev@example.com key={SECRET}",
        source="test",
    )
    stats = middleware.stats()

    assert "dev@example.com" not in json.dumps(stats)
    assert SECRET not in json.dumps(stats)
    assert "dev@example.com" not in result.sanitized_text
    assert SECRET not in result.sanitized_text
    assert stats["total_secrets_redacted"] >= 1
    assert stats["total_pii_redacted"] >= 1
    assert "<AIRLOCK_PII_" in result.sanitized_text


def test_scan_proposed_write_detects_secrets_and_unsafe_patterns() -> None:
    middleware = AirlockMiddleware(_settings())

    findings = middleware.scan_proposed_write(
        "src/config.py",
        f'API_KEY = "{SECRET}"\neval(user_input)\n',
    )

    categories = {finding.category for finding in findings}
    assert "api_key" in categories
    assert "unsafe_pattern" in categories
    public = json.dumps([finding.public_dict() for finding in findings])
    assert SECRET not in public


# ---------------------------------------------------------------------------
# Router / Jev sanitization guarantees
# ---------------------------------------------------------------------------


class NeverUsedJev:
    def __init__(self) -> None:
        self.contexts: list[RoutingContext] = []

    def evaluate(self, ctx: RoutingContext) -> Any:
        self.contexts.append(ctx)
        raise AssertionError("forced router should not call Jev")


def test_router_sanitizes_open_compat_request_payload() -> None:
    settings = _settings()
    middleware = AirlockMiddleware(settings)
    fake_client = FakeOpenAIClient(_openai_response())
    provider = OpenCompatProvider(
        name="openrouter",
        model="deepseek/deepseek-v4",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        client=fake_client,
        requested_model="deepseek/deepseek-v4",
    )
    router = ModelRouter(
        OneProviderRegistry(provider),
        settings,
        Console(quiet=True),
        initial_tier="coder",
        jev_client=NeverUsedJev(),
        airlock=middleware,
    )

    router.complete(
        "",
        messages=[
            {"role": "system", "content": f"System has {SECRET}"},
            {"role": "user", "content": f"Use OPENROUTER_API_KEY={SECRET}"},
        ],
    )

    payload = json.dumps(fake_client.completions.calls[0])
    assert SECRET not in payload
    assert PLACEHOLDER in payload


def test_jev_sanitizes_prompt_before_http_request() -> None:
    settings = _settings()
    middleware = AirlockMiddleware(settings)
    response = {
        "answers": {
            "task_intent": {"choice": "search_or_read", "confidence": 0.9},
            "reasoning_depth": {"level": "low"},
            "needs_thinking_trace": {"probability": 0.1},
            "destructive_risk": {"probability": 0.1},
            "ast_context_sufficient": {"probability": 0.9},
        }
    }
    http = FakeHTTPClient(FakeHTTPResponse(200, response))
    client = JevClient(
        settings,
        http_client=http,
        env={"JEV_API_KEY": "test"},
        airlock=middleware,
    )

    client.evaluate(
        RoutingContext(user_prompt=f"Use the key {SECRET} to search")
    )

    payload = json.dumps(http.calls[0]["json"])
    assert SECRET not in payload
    assert PLACEHOLDER in payload


class FakeMCPManager:
    status = "connected"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def has_scan_tool(self, names: Any = None) -> bool:
        return True

    def find_scan_tool(self, names: Any = None) -> tuple[str, str]:
        return "code-airlock", "airlock_scan"

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        server: str | None = None,
    ) -> Any:
        self.calls.append(
            {"name": name, "arguments": arguments or {}, "server": server}
        )
        return {
            "findings": [
                {
                    "value": SECRET,
                    "category": "api_key",
                    "source": "mcp",
                }
            ]
        }


def test_middleware_uses_mcp_primary_scanner_when_available() -> None:
    manager = FakeMCPManager()
    middleware = AirlockMiddleware(_settings(), mcp_manager=manager)

    result = middleware.sanitize_outbound(f"key={SECRET}", source="mcp_test")

    assert result.redaction_count == 1
    assert SECRET not in result.sanitized_text
    assert PLACEHOLDER in result.sanitized_text
    assert result.findings[0].detector_source == "code-airlock-mcp"
    assert middleware.mode == "mcp_connected"
    assert manager.calls[0]["name"] == "airlock_scan"


def test_agent_sanitizes_tool_outputs_and_restores_write_placeholders(
    tmp_path: Path,
) -> None:
    settings = _settings()
    middleware = AirlockMiddleware(settings)
    middleware.sanitize_outbound(f"OPENROUTER_API_KEY={SECRET}", source="seed")

    target = tmp_path / "written.txt"
    provider = ScriptedProvider(
        [
            LLMResponse(
                content="",
                model="fake/model",
                provider="fake",
                tool_calls=[
                    ToolCall(
                        name="file_write",
                        arguments={
                            "path": str(target),
                            "content": f'KEY="{PLACEHOLDER}"',
                        },
                        id="call_1",
                    )
                ],
                usage=TokenUsage(),
            ),
            LLMResponse(content="done", model="fake/model", provider="fake"),
        ]
    )
    registry = ToolRegistry()
    registry.register(FileWriteTool())
    agent = Agent(
        provider,
        registry,
        settings,
        console=Console(quiet=True),
        airlock=middleware,
    )

    assert agent.run("write the key") == "done"
    assert target.read_text(encoding="utf-8") == f'KEY="{SECRET}"'
    # The model never sees the raw value.
    assert SECRET not in json.dumps(provider.calls)


def test_agent_sanitizes_file_read_results_before_followup_call(
    tmp_path: Path,
) -> None:
    settings = _settings()
    middleware = AirlockMiddleware(settings)
    source = tmp_path / "secret.txt"
    source.write_text(f"token={SECRET}\n", encoding="utf-8")

    provider = ScriptedProvider(
        [
            LLMResponse(
                content="",
                model="fake/model",
                provider="fake",
                tool_calls=[
                    ToolCall(
                        name="file_read",
                        arguments={"path": str(source)},
                        id="call_1",
                    )
                ],
                usage=TokenUsage(),
            ),
            LLMResponse(content="done", model="fake/model", provider="fake"),
        ]
    )
    registry = ToolRegistry()
    registry.register(FileReadTool(settings.context))
    agent = Agent(
        provider,
        registry,
        settings,
        console=Console(quiet=True),
        airlock=middleware,
    )

    agent.run("read the token file")

    followup_messages = provider.calls[1]["messages"]
    serialized = json.dumps(followup_messages)
    assert SECRET not in serialized
    assert PLACEHOLDER in serialized


# ---------------------------------------------------------------------------
# MCP stdio / bridge / fallback
# ---------------------------------------------------------------------------

MOCK_MCP_SERVER = r"""
import json, sys
for line in sys.stdin:
    try:
        msg = json.loads(line)
    except Exception:
        continue
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "mock-airlock", "version": "1.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "airlock_scan",
                    "description": "Scan text",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                }
            ]
        }
    elif method == "tools/call":
        args = msg.get("params", {}).get("arguments", {})
        result = {
            "content": [{"type": "text", "text": "scanned " + json.dumps(args)}],
            "isError": False,
        }
    else:
        result = {}
    if req_id is not None:
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}), flush=True)
"""


def test_mcp_stdio_handshake_discovery_and_call() -> None:
    client = MCPStdioClient(
        sys.executable,
        ["-u", "-c", MOCK_MCP_SERVER],
        name="mock-airlock",
        timeout=5.0,
    )
    try:
        assert client.start() is True
        assert client.status == "connected"
        assert client.server_info["serverInfo"]["name"] == "mock-airlock"

        tools = client.list_tools()
        assert [tool.name for tool in tools] == ["airlock_scan"]

        result = client.call_tool("airlock_scan", {"text": "hello"})
        assert result["isError"] is False
        assert "hello" in result["content"][0]["text"]
    finally:
        client.close()
    assert client.status == "closed"


def test_mcp_bridge_registers_discovered_tools() -> None:
    manager = MCPClientManager()
    manager.add_stdio_server(
        "mock-airlock",
        sys.executable,
        ["-u", "-c", MOCK_MCP_SERVER],
        timeout=5.0,
    )
    manager.start_all()
    registry = ToolRegistry()
    try:
        registered = register_mcp_tools(registry, manager)
        assert registered == ["airlock_scan"]
        adapter = registry.get("airlock_scan")
        assert adapter is not None
        output = adapter.execute({"text": "hello"})
        assert "scanned" in output
        assert "hello" in output
    finally:
        manager.close_all()


def test_missing_code_airlock_binary_falls_back_to_builtin_scanner() -> None:
    settings = _settings()
    settings.security.airlock.mcp_command = "axe-missing-code-airlock-binary"
    middleware = AirlockMiddleware(settings)
    registry = ToolRegistry()

    manager = setup_default_servers(
        settings,
        registry,
        middleware,
        Console(quiet=True),
    )

    assert manager.status == "fallback_builtin"
    assert middleware.mode == "builtin_cascade"
    assert registry.has("airlock_scan")
    tool = registry.get("airlock_scan")
    assert isinstance(tool, AirlockScanTool)

    output = tool.execute({"text": f"OPENROUTER_API_KEY={SECRET}"})
    assert SECRET not in output
    assert "<AIRLOCK_SECRET_" in output
    assert "api_key" in output


def test_default_server_boot_registers_external_mcp_tools() -> None:
    settings = _settings()
    settings.security.airlock.mcp_command = sys.executable
    settings.security.airlock.mcp_args = ["-u", "-c", MOCK_MCP_SERVER]
    middleware = AirlockMiddleware(settings)
    registry = ToolRegistry()

    manager = setup_default_servers(
        settings,
        registry,
        middleware,
        Console(quiet=True),
    )
    try:
        assert manager.status == "connected"
        assert registry.has("airlock_scan")
        assert middleware.mode == "mcp_connected"
        adapter = registry.get("airlock_scan")
        assert adapter is not None
        output = adapter.execute({"text": "hello"})
        assert "scanned" in output
    finally:
        manager.close_all()


def test_mcp_http_parse_json_and_sse_responses() -> None:
    json_response = SimpleNamespace(
        headers={"content-type": "application/json"},
        json=lambda: {"jsonrpc": "2.0", "id": 3, "result": {"ok": True}},
    )
    parsed = MCPHTTPClient._parse_http_response(json_response, 3)
    assert parsed["result"]["ok"] is True

    sse_response = SimpleNamespace(
        headers={"content-type": "text/event-stream"},
        text='data: {"jsonrpc": "2.0", "id": 4, "result": {"ok": true}}\n',
    )
    parsed = MCPHTTPClient._parse_http_response(sse_response, 4)
    assert parsed["result"]["ok"] is True


# ---------------------------------------------------------------------------
# REPL /airlock
# ---------------------------------------------------------------------------


def test_repl_airlock_status_and_scan_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    middleware = AirlockMiddleware(settings)
    middleware.sanitize_outbound(f"key={SECRET}", source="status_test")
    session = SimpleNamespace(
        settings=settings,
        airlock=middleware,
        agent=SimpleNamespace(registry=ToolRegistry()),
    )
    buffer = io.StringIO()
    test_console = Console(
        file=buffer,
        width=200,
        force_terminal=False,
        color_system=None,
    )
    monkeypatch.setattr(repl_module, "console", test_console)

    assert repl_module.handle_repl_command("/airlock", session) is True
    status_output = buffer.getvalue()
    assert "Mode:" in status_output
    assert "Total secrets redacted: 1" in status_output
    assert "Active vault placeholders: 1" in status_output
    assert SECRET not in status_output

    secret_file = tmp_path / "secret.env"
    secret_file.write_text(f"OPENROUTER_API_KEY={SECRET}\n", encoding="utf-8")
    buffer.truncate(0)
    buffer.seek(0)
    assert (
        repl_module.handle_repl_command(
            f"/airlock scan {secret_file}",
            session,
        )
        is True
    )
    scan_output = buffer.getvalue()
    assert "Airlock Scan" in scan_output
    assert "<AIRLOCK_SECRET_" in scan_output
    assert SECRET not in scan_output
