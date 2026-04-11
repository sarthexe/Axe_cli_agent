"""Shell tool for executing system commands."""

from __future__ import annotations

import subprocess
import os
import signal
import threading
from typing import Any

from config.settings import SandboxSettings
from sandbox.permissions import CommandSafety
from tools.base import BaseTool


class ShellTool(BaseTool):
    """Executes a bash command in the terminal."""

    name: str = "shell"
    description: str = "Executes a shell command and returns its output (stdout/stderr combined)."

    def __init__(
        self,
        sandbox: SandboxSettings,
        safety: CommandSafety | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.safety = safety or CommandSafety(sandbox_settings=sandbox)

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
        command = str(arguments.get("command", "")).strip()
        if not command:
            return "Error: Missing required argument 'command'."

        decision = self.safety.analyze(command)
        if decision.blocked:
            return self.safety.blocked_message(command, decision)

        try:
            # Bug 3: Using subprocess.Popen with a reader thread to enforce size and timeout limits
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout to prevent pipe blocking
                text=True,
                cwd=os.getcwd(),
                preexec_fn=os.setsid
            )
            
            output_lines = []
            total_bytes = 0
            max_bytes = self.sandbox.max_output_bytes

            def read_stream(stream):
                nonlocal total_bytes
                for line in stream:
                    if total_bytes >= max_bytes:
                        try:
                            # Kill process group if limit exceeded
                            os.killpg(process.pid, signal.SIGKILL)
                        except OSError:
                            pass
                        break
                    output_lines.append(line)
                    # text=True streams strings, so encode to count realistic byte load
                    total_bytes += len(line.encode("utf-8"))

            t = threading.Thread(target=read_stream, args=(process.stdout,))
            t.start()
            t.join(timeout=self.sandbox.command_timeout_seconds)

            if t.is_alive():
                # Timeout hit — kill the process group
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                t.join() # Wait for thread to finish reading closed pipe
                return f"[ERROR] Command timed out after {self.sandbox.command_timeout_seconds} seconds."

            process.wait()
            exit_code = process.returncode
            output = "".join(output_lines)

            if total_bytes >= max_bytes:
                # Truncate to last 200 lines if we hit the cap
                lines = output.splitlines()
                truncated_output = "\n".join(lines[-200:])
                output = f"[...truncated {total_bytes // 1024}KB, showing last 200 lines due to size limit hit]\n{truncated_output}"

            if exit_code != 0:
                res = f"Command exited with code {exit_code}:\n{output}"
                return res.strip()

            return output.strip() if output.strip() else "Process completed with no output."

        except Exception as e:
            return f"Error executing shell command: {e}"
