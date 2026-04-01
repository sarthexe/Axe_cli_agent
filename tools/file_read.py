"""File Read tool for reading file contents."""

from __future__ import annotations

import os
from typing import Any

from config.settings import ContextSettings
from tools.base import BaseTool


class FileReadTool(BaseTool):
    """Reads a file from the disk and returns its contents with line numbers."""

    name: str = "file_read"
    description: str = "Reads a file and returns its contents with line numbers. Use start_line and end_line to read specific parts of large files."

    def __init__(self, context_settings: ContextSettings) -> None:
        self.context_settings = context_settings

    def parameters(self) -> dict[str, Any]:
        """Schema for the file read tool."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional starting line number (1-indexed).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional ending line number (1-indexed).",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    def validate(self, arguments: dict[str, Any]) -> None:
        path = arguments.get("path")
        if not path:
            raise ValueError("File path must be provided.")

    def execute(self, arguments: dict[str, Any]) -> str:
        path = arguments["path"]
        start_line = arguments.get("start_line")
        end_line = arguments.get("end_line")

        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."
        if not os.path.isfile(path):
            return f"Error: '{path}' is not a file."

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return f"Error: File '{path}' appears to be binary or has non-UTF-8 encoding."
        except Exception as e:
            return f"Error reading file '{path}': {e}"

        total_lines = len(lines)
        start = max(1, start_line) if start_line is not None else 1
        end = min(total_lines, end_line) if end_line is not None else total_lines

        if start > total_lines:
            return f"Error: start_line ({start}) is beyond the end of the file ({total_lines} lines)."
        if start > end:
            return f"Error: start_line ({start}) cannot be greater than end_line ({end})."

        # Cap at max_file_lines
        max_lines = self.context_settings.max_file_lines
        if (end - start + 1) > max_lines:
            end = start + max_lines - 1
            truncated = True
        else:
            truncated = False

        output = [f"--- {path} (lines {start}-{end} of {total_lines}) ---"]
        for i in range(start - 1, end):
            output.append(f"{i + 1:4d} | {lines[i].rstrip('\\n')}")

        if truncated:
            output.append(f"... [Truncated. Max {max_lines} lines allowed. Use start_line and end_line to read the rest.]")

        return "\n".join(output)
