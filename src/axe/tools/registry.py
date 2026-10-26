"""Tool registry."""

from __future__ import annotations

from typing import Any

from axe.tools.base import BaseTool


class ToolRegistry:
    """Manages available tools and generates LLM tool schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if not tool.name:
            raise ValueError("Tool name cannot be empty.")
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Return a list of tool schemas for the LLM."""
        schemas: list[dict[str, Any]] = []
        for name, tool in self._tools.items():
            schemas.append({
                "name": name,
                "description": tool.description,
                "parameters": tool.parameters()
            })
        return schemas

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with the given arguments."""
        if name not in self._tools:
            return f"Error: Tool '{name}' not found."
        
        tool = self._tools[name]
        try:
            tool.validate(arguments)
            return tool.execute(arguments)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"
