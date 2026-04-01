"""Glob Search tool for finding files."""

from __future__ import annotations

import os
import fnmatch
from typing import Any

from tools.base import BaseTool


class GlobSearchTool(BaseTool):
    """Searches for files matching a pattern."""

    name: str = "glob_search"
    description: str = "Finds files matching a pattern. Standard ignore dirs (.git, node_modules, .venv) are skipped."

    def parameters(self) -> dict[str, Any]:
        """Schema for the glob search tool."""
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The pattern to search for (e.g., '*.py', '*test*').",
                }
            },
            "required": ["pattern"],
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        pattern = arguments.get("pattern")
        if not pattern:
            return "Error: Pattern must be provided."

        ignore_dirs = {".git", "node_modules", ".venv", "venv", "__pycache__"}
        
        # Load .gitignore if present
        gitignores = []
        if os.path.exists(".gitignore"):
            try:
                with open(".gitignore", "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            gitignores.append(line)
            except Exception:
                pass
                
        matches = []
        try:
            for root, dirs, files in os.walk("."):
                # Modifying dirs in-place to skip ignored directories entirely
                dirs[:] = [d for d in dirs if d not in ignore_dirs]
                
                for file in files:
                    rel_root = os.path.relpath(root, ".")
                    if rel_root == ".":
                        file_path = file
                    else:
                        file_path = os.path.join(rel_root, file)

                    if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(file_path, pattern):
                        # Simple gitignore check
                        ignored = False
                        for g in gitignores:
                            if fnmatch.fnmatch(file_path, g) or fnmatch.fnmatch(file_path, g.strip("/") + "/*"):
                                ignored = True
                                break
                        if not ignored:
                            matches.append(file_path)
                            
        except Exception as e:
            return f"Error executing glob search: {e}"

        if not matches:
            return f"No files found matching pattern '{pattern}'."

        return "\\n".join(sorted(matches))
