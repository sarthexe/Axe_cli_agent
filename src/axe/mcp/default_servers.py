"""Default MCP server bootstrapping and built-in Airlock scan tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from axe.mcp.bridge import register_mcp_tools
from axe.mcp.client import MCPClientManager
from axe.security.airlock_middleware import (
    AIRLOCK_SCAN_TOOL_NAMES,
    AirlockMiddleware,
    RedactionMatch,
)
from axe.tools.base import BaseTool
from axe.tools.registry import ToolRegistry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rich.console import Console

    from axe.config.settings import Settings

DEFAULT_AIRLOCK_SERVER_NAME = "code-airlock"


def _finding_summary(
    findings: list[RedactionMatch],
    source: str,
) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    for finding in findings:
        by_category[finding.category] = by_category.get(finding.category, 0) + 1
    return {
        "source": source,
        "finding_count": len(findings),
        "by_category": by_category,
        "findings": [finding.public_dict() for finding in findings],
    }


class AirlockScanTool(BaseTool):
    """Built-in local Airlock scan tool used when MCP is unavailable."""

    name = "airlock_scan"
    description = (
        "Scan text, a file, or a directory for secrets, credentials, and PII. "
        "Returns placeholders and categories only; raw secrets are never exposed."
    )

    def __init__(self, middleware: AirlockMiddleware) -> None:
        self._middleware = middleware

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to scan.",
                },
                "text": {
                    "type": "string",
                    "description": "Inline text to scan.",
                },
            },
            "additionalProperties": False,
        }

    def validate(self, arguments: dict[str, Any]) -> None:
        if not arguments.get("path") and not arguments.get("text"):
            raise ValueError("Provide either 'path' or 'text'.")

    def execute(self, arguments: dict[str, Any]) -> str:
        path = str(arguments.get("path") or "").strip()
        text = arguments.get("text")

        if path:
            if not Path(path).expanduser().exists():
                return f"Error: Path not found: {path}"
            findings = self._middleware.scan_path(path)
            source = path
        elif isinstance(text, str):
            findings = self._middleware.scan_text(
                text,
                source_hint="airlock_scan_text",
            )
            source = "inline_text"
        else:
            return "Error: Provide either 'path' or 'text'."

        return json.dumps(
            _finding_summary(findings, source),
            ensure_ascii=True,
            indent=2,
        )


def register_builtin_airlock_tool(
    registry: ToolRegistry,
    middleware: AirlockMiddleware,
) -> AirlockScanTool:
    """Ensure the native ``airlock_scan`` tool is registered."""
    existing = registry.get("airlock_scan")
    if isinstance(existing, AirlockScanTool):
        return existing
    tool = AirlockScanTool(middleware)
    registry.register(tool)
    return tool


def build_default_mcp_manager(settings: "Settings") -> MCPClientManager:
    """Build the manager and configure the default code-airlock stdio server."""
    manager = MCPClientManager()
    airlock = settings.security.airlock
    if not airlock.enabled or not airlock.default_mcp_enabled:
        return manager
    manager.add_stdio_server(
        DEFAULT_AIRLOCK_SERVER_NAME,
        airlock.mcp_command,
        list(airlock.mcp_args),
        timeout=5.0,
    )
    return manager


def setup_default_servers(
    settings: "Settings",
    registry: ToolRegistry,
    middleware: AirlockMiddleware,
    console: "Console | None" = None,
) -> MCPClientManager:
    """Boot code-airlock if available and always guarantee a scan tool.

    If the external server cannot start, the built-in cascade-backed
    ``airlock_scan`` tool is registered instead. This is intentionally
    non-fatal.
    """
    airlock = settings.security.airlock
    manager = build_default_mcp_manager(settings)

    if manager.servers:
        manager.start_all()
        if manager.connected:
            register_mcp_tools(registry, manager)
        else:
            statuses = ", ".join(
                f"{name}={status}"
                for name, status in manager.server_statuses.items()
            )
            if console is not None:
                console.print(
                    "[dim]Airlock: external code-airlock MCP unavailable "
                    f"({statuses}); using built-in cascade scanner.[/dim]"
                )

    middleware.set_mcp_manager(manager)

    external_scan = manager.has_scan_tool(AIRLOCK_SCAN_TOOL_NAMES)
    if not external_scan and (airlock.enabled or airlock.default_mcp_enabled):
        register_builtin_airlock_tool(registry, middleware)

    return manager
