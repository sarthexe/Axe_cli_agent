"""Shell tool for executing system commands."""

from __future__ import annotations

import asyncio
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
        cmd = arguments.get("command", "")
        # Basic check for exact blocked commands based on sandbox settings
        for blocked in self.sandbox.blocked_commands:
            if blocked in cmd:
                raise ValueError(f"Command contains blocked pattern: {blocked}")

    def execute(self, arguments: dict[str, Any]) -> str:
        """Synchronous wrapper for asyncio shell execution."""
        command = arguments["command"]
        return asyncio.run(self._run_async(command))

    async def _run_async(self, command: str) -> str:
        try:
            # We combine stdout and stderr into stdout to capture all output easily
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.getcwd(),
            )
            
            try:
                stdout_data, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.sandbox.command_timeout_seconds
                )
            except asyncio.TimeoutError:
                # Kill process tree gracefully (if possible), or just kill the shell process
                process.kill()
                await process.wait()
                return f"Error: Command timed out after {self.sandbox.command_timeout_seconds} seconds."

            output = stdout_data.decode(errors="replace") if stdout_data else ""
            exit_code = process.returncode
            
            # Truncation logic (cap at max output bytes)
            output_bytes = len(output.encode("utf-8"))
            if output_bytes > self.sandbox.max_output_bytes:
                # Cap the output based on lines, retaining the last 200 lines
                lines = output.splitlines()
                truncated_output = "\\n".join(lines[-200:])
                output = f"[...truncated {output_bytes // 1024}KB, showing last 200 lines]\\n{truncated_output}"

            if exit_code != 0:
                result = f"Command exited with code {exit_code}:\\n{output}"
                return result.strip()
            
            return output.strip() if output.strip() else "Process completed with no output."

        except Exception as e:
            return f"Error executing shell command: {e}"
