"""File Write tool for creating and writing to files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.base import BaseTool


class FileWriteTool(BaseTool):
    """Creates a new file or overwrites an existing file."""

    name: str = "file_write"
    description: str = "Creates a new file or completely overwrites an existing file with the provided content. Missing directories will be created automatically."

    def parameters(self) -> dict[str, Any]:
        """Schema for the file write tool."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to create or overwrite.",
                },
                "content": {
                    "type": "string",
                    "description": "The complete content to write to the file.",
                }
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        path_str = arguments.get("path")
        content = arguments.get("content")

        if not path_str:
            return "Error: File path must be provided."
        
        path = Path(path_str)

        try:
            # Auto-create parent directories securely
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
                
            return f"Successfully wrote {len(content)} characters to {path_str}."
        except Exception as e:
            return f"Error writing to file '{path_str}': {e}"
