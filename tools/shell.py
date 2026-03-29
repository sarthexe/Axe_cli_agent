"""Shell tool for executing system commands."""

from __future__ import annotations

import subprocess
import os
from typing import Any

from config.settings import SandboxSettings
from tools.base import BaseTool


class ShellTool(BaseTool):
    """Executes a bash command in the terminal."""

    name: str = "shell"
    description: str = "Executes a shell command and returns its output (stdout/stderr combined)."

    def __init__(self, sandbox: SandboxSettings) -> None:
        self.sandbox = sandbox

    def parameters(self) -> dict[str, Any]:
        """Schema for the shell tool."""
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command line string to execute.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    def validate(self, arguments: dict[str, Any]) -> None:
        pass  # We handle blocking directly in execute to return an explanatory string

    def execute(self, arguments: dict[str, Any]) -> str:
        """Executes the shell command synchronously with a hard timeout."""
        command = arguments["command"]

        # Bug 5: Explanation for blocked commands
        for blocked in self.sandbox.blocked_commands:
            if blocked in command:
                return (
                    f"[BLOCKED] The command '{command}' was blocked for safety. "
                    f"Commands matching rm -rf, mkfs, dd, and fork bombs are not allowed. "
                    f"If you need to delete files, use a more targeted command."
                )

        try:
            # Bug 3: Using subprocess.run to strictly enforce timeout
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.sandbox.command_timeout_seconds,
                cwd=os.getcwd()
            )
        except subprocess.TimeoutExpired:
            return f"[ERROR] Command timed out after {self.sandbox.command_timeout_seconds} seconds."
        except Exception as e:
            return f"Error executing shell command: {e}"

        output = result.stdout + ("\n" + result.stderr if result.stderr else "")
        exit_code = result.returncode

        # Truncation logic (cap at max output bytes)
        output_bytes = len(output.encode("utf-8"))
        if output_bytes > self.sandbox.max_output_bytes:
            # Cap the output based on lines, retaining the last 200 lines
            lines = output.splitlines()
            truncated_output = "\n".join(lines[-200:])
            output = f"[...truncated {output_bytes // 1024}KB, showing last 200 lines]\n{truncated_output}"

        if exit_code != 0:
            res = f"Command exited with code {exit_code}:\n{output}"
            return res.strip()

        return output.strip() if output.strip() else "Process completed with no output."
