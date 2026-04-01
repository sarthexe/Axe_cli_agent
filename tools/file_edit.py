"""File Edit tool for modifying files."""

from __future__ import annotations

import os
import difflib
from typing import Any

from tools.base import BaseTool


class FileEditTool(BaseTool):
    """Replaces exact strings in a file."""

    name: str = "file_edit"
    description: str = "Edits a file by replacing an exact occurrences of 'target' string with 'replacement' string. The target must exist exactly once in the file."

    def parameters(self) -> dict[str, Any]:
        """Schema for the file edit tool."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit.",
                },
                "target": {
                    "type": "string",
                    "description": "The exact string to find in the file. Must match exactly once, including all whitespace and newlines.",
                },
                "replacement": {
                    "type": "string",
                    "description": "The exact string to replace the target with.",
                }
            },
            "required": ["path", "target", "replacement"],
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        path = arguments.get("path")
        target = arguments.get("target")
        replacement = arguments.get("replacement")

        if not path or target is None or replacement is None:
            return "Error: Missing required arguments."

        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."
        if not os.path.isfile(path):
            return f"Error: '{path}' is not a file."

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return f"Error: File '{path}' appears to be binary or has non-UTF-8 encoding."
        except Exception as e:
            return f"Error reading file '{path}': {e}"

        count = content.count(target)
        if count == 0:
            return "Error: The target string was not found in the file. Ensure you have the exact match, including all whitespace."
        elif count > 1:
            return f"Error: The target string was found {count} times. The target must be unique to ensure safe editing."

        new_content = content.replace(target, replacement)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            return f"Error writing to file '{path}': {e}"

        # Generate a unified diff
        diff = difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
        
        diff_text = "".join(diff)
        return f"Successfully edited {path}.\\n\\nDiff:\\n```diff\\n{diff_text}```"
