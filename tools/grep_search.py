"""Grep Search tool for searching file contents."""

from __future__ import annotations

import os
import re
from typing import Any

from tools.base import BaseTool


class GrepSearchTool(BaseTool):
    """Searches for regex patterns inside files."""

    name: str = "grep_search"
    description: str = "Recursively searches for a regex pattern in the contents of all files in the given directory."

    def parameters(self) -> dict[str, Any]:
        """Schema for the grep search tool."""
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The python regex pattern to search for in file contents.",
                },
                "dir_path": {
                    "type": "string",
                    "description": "The directory to search in, defaults to '.'",
                }
            },
            "required": ["pattern"],
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        pattern = arguments.get("pattern")
        dir_path = arguments.get("dir_path", ".")

        if not pattern:
            return "Error: Pattern must be provided."

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern '{pattern}': {e}"

        ignore_dirs = {".git", "node_modules", ".venv", "venv", "__pycache__"}
        
        matches = []
        try:
            for root, dirs, files in os.walk(dir_path):
                dirs[:] = [d for d in dirs if d not in ignore_dirs]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for idx, line in enumerate(f):
                                if regex.search(line):
                                    matches.append(f"{file_path}:{idx + 1}:{line.rstrip('\\n')}")
                                    if len(matches) > 1000:
                                        return "\\n".join(matches) + "\\n... [Output truncated to 1000 matches]"
                    except (UnicodeDecodeError, PermissionError):
                        # Skip binary files or unreadable files
                        pass
        except Exception as e:
            return f"Error executing grep search: {e}"

        if not matches:
            return f"No matches found for pattern '{pattern}'."

        return "\\n".join(matches)
