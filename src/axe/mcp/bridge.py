"""Bridge discovered MCP tools into Axe's native tool registry."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from axe.mcp.client import MCPClientManager, MCPTool
from axe.tools.base import BaseTool
from axe.tools.registry import ToolRegistry


def format_mcp_result(result: Any) -> str:
    """Normalize MCP ``tools/call`` results into a tool output string."""
    is_error = False
    if isinstance(result, Mapping):
        is_error = bool(result.get("isError") or result.get("is_error"))

    text = ""
    if isinstance(result, str):
        text = result
    elif isinstance(result, Mapping):
        content = result.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    if item.get("type") == "text" and item.get("text") is not None:
                        parts.append(str(item["text"]))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=True))
                else:
                    parts.append(str(item))
            text = "\n".join(parts)
        elif content is not None:
            text = str(content)
        elif result.get("text") is not None:
            text = str(result["text"])
        else:
            text = json.dumps(result, ensure_ascii=True, indent=2)
    elif result is None:
        text = "MCP tool completed with no output."
    else:
        text = str(result)

    if is_error:
        return f"Error: {text}" if text else "Error: MCP tool failed."
    return text


class MCPToolAdapter(BaseTool):
    """Expose one MCP tool as a native Axe :class:`BaseTool`."""

    def __init__(
        self,
        tool: MCPTool,
        manager: MCPClientManager,
        server_name: str,
        *,
        name: str | None = None,
    ) -> None:
        self.name = name or tool.name
        self.description = tool.description or f"MCP tool '{tool.name}' from {server_name}."
        self._tool = tool
        self._manager = manager
        self._server_name = server_name

    def parameters(self) -> dict[str, Any]:
        schema = self._tool.input_schema
        if isinstance(schema, Mapping) and schema:
            return dict(schema)
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            result = self._manager.call_tool(
                self._tool.name,
                arguments,
                server=self._server_name,
            )
            return format_mcp_result(result)
        except Exception as exc:  # noqa: BLE001 - tool errors are model-visible text
            return f"Error calling MCP tool '{self._tool.name}': {exc}"


def register_mcp_tools(
    registry: ToolRegistry,
    manager: MCPClientManager,
) -> list[str]:
    """Register every discovered MCP tool with collision-safe names."""
    registered: list[str] = []
    for server_name, server in manager.servers.items():
        for tool in server.tools:
            name = tool.name
            if registry.has(name):
                name = f"{server_name}__{tool.name}"
            adapter = MCPToolAdapter(
                tool,
                manager,
                server_name,
                name=name,
            )
            registry.register(adapter)
            registered.append(name)
    return registered
