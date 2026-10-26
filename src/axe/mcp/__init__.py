"""MCP client, bridge, and default server bootstrapping."""

from __future__ import annotations

from axe.mcp.bridge import MCPToolAdapter, format_mcp_result, register_mcp_tools
from axe.mcp.client import (
    MCPClientBase,
    MCPClientManager,
    MCPError,
    MCPHTTPClient,
    MCPServer,
    MCPSSEClient,
    MCPStdioClient,
    MCPTimeoutError,
    MCPTool,
)
from axe.mcp.default_servers import (
    AirlockScanTool,
    build_default_mcp_manager,
    register_builtin_airlock_tool,
    setup_default_servers,
)

__all__ = [
    "AirlockScanTool",
    "MCPClientBase",
    "MCPClientManager",
    "MCPError",
    "MCPHTTPClient",
    "MCPServer",
    "MCPSSEClient",
    "MCPStdioClient",
    "MCPTimeoutError",
    "MCPTool",
    "MCPToolAdapter",
    "build_default_mcp_manager",
    "format_mcp_result",
    "register_builtin_airlock_tool",
    "register_mcp_tools",
    "setup_default_servers",
]
