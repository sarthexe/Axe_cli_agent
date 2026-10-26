"""Lightweight MCP (Model Context Protocol) JSON-RPC 2.0 clients.

Supports newline-delimited JSON-RPC over ``stdio`` and structured JSON / SSE
over HTTP. Missing binaries and failed handshakes are non-fatal: clients mark
themselves ``fallback_builtin`` so Axe can keep running with its local scanner.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Any

MCP_PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "axe"
CLIENT_VERSION = "0.2.0"


class MCPError(RuntimeError):
    """Base class for MCP transport/protocol errors."""


class MCPTimeoutError(MCPError):
    """Raised when an MCP request exceeds its configured timeout."""


@dataclass(slots=True)
class MCPTool:
    """Discovered MCP tool metadata."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    server: str = ""


class MCPClientBase(ABC):
    """Shared MCP handshake, discovery, and invocation behavior."""

    def __init__(
        self,
        *,
        name: str = "mcp",
        timeout: float = 8.0,
    ) -> None:
        self.name = name
        self.timeout = timeout
        self.status = "disconnected"
        self.last_error: str | None = None
        self.server_info: dict[str, Any] = {}
        self.tools: list[MCPTool] = []
        self.notifications: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> dict[str, Any]:
        """Perform the MCP initialize handshake."""
        result = self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": CLIENT_NAME,
                    "version": CLIENT_VERSION,
                },
            },
        )
        self._notify("notifications/initialized", {})
        self.server_info = dict(result) if isinstance(result, Mapping) else {}
        self.status = "connected"
        return self.server_info

    def list_tools(self) -> list[MCPTool]:
        """Discover tools, following pagination cursors when present."""
        tools: list[MCPTool] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            result = self._request("tools/list", params)
            if not isinstance(result, Mapping):
                break
            raw_tools = result.get("tools")
            if isinstance(raw_tools, list):
                for spec in raw_tools:
                    if not isinstance(spec, Mapping):
                        continue
                    name = spec.get("name")
                    if not name:
                        continue
                    schema = (
                        spec.get("inputSchema")
                        or spec.get("input_schema")
                        or spec.get("parameters")
                        or {}
                    )
                    tools.append(
                        MCPTool(
                            name=str(name),
                            description=str(spec.get("description") or ""),
                            input_schema=dict(schema) if isinstance(schema, Mapping) else {},
                            server=self.name,
                        )
                    )
            next_cursor = result.get("nextCursor") or result.get("next_cursor")
            if not next_cursor:
                break
            cursor = str(next_cursor)
        self.tools = tools
        return tools

    def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        """Invoke an MCP tool and return its raw result object."""
        return self._request(
            "tools/call",
            {"name": tool_name, "arguments": dict(arguments or {})},
        )

    def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        """Send a JSON-RPC request (public testing/compat helper)."""
        return self._request(method, dict(params or {}))

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        self._notify(method, dict(params or {}))

    def close(self) -> None:
        """Close the client. Subclasses release transport resources."""
        self.status = "closed"

    @property
    def stderr_tail(self) -> list[str]:
        return []

    # ------------------------------------------------------------------
    # Transport contract
    # ------------------------------------------------------------------

    @abstractmethod
    def start(self) -> bool:
        """Start/connect the transport and perform the handshake."""

    @abstractmethod
    def _request(self, method: str, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _notify(self, method: str, params: dict[str, Any]) -> None:
        raise NotImplementedError


class MCPStdioClient(MCPClientBase):
    """JSON-RPC 2.0 over newline-delimited ``stdio``."""

    _EOF = object()

    def __init__(
        self,
        command: str | Sequence[str],
        args: Sequence[str] | None = None,
        *,
        name: str = "mcp",
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 8.0,
        process: subprocess.Popen[str] | None = None,
    ) -> None:
        super().__init__(name=name, timeout=timeout)
        if isinstance(command, (list, tuple)):
            parts = [str(part) for part in command]
            if not parts:
                raise ValueError("MCP command cannot be empty.")
            self.command = parts[0]
            self.args = parts[1:]
        else:
            self.command = str(command)
            self.args = [str(arg) for arg in (args or [])]
        self.cwd = cwd
        self.env = dict(env) if env is not None else None
        self._process = process
        self._pending: dict[Any, queue.Queue[Any]] = {}
        self._write_lock = threading.Lock()
        self._ids = count(1)
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: deque[str] = deque(maxlen=100)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        if self.status == "connected":
            return True
        if self._process is None:
            command = [self.command, *self.args]
            process_env = None
            if self.env is not None:
                process_env = {**os.environ, **self.env}
            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    cwd=self.cwd,
                    env=process_env,
                )
            except FileNotFoundError:
                self.status = "fallback_builtin"
                self.last_error = f"MCP command not found: {self.command}"
                return False
            except OSError as exc:
                self.status = "fallback_builtin"
                self.last_error = f"Could not start MCP command '{self.command}': {exc}"
                return False

        self._start_threads()
        try:
            self.initialize()
        except Exception as exc:  # noqa: BLE001 - handshake failures are non-fatal
            self.last_error = f"MCP handshake failed: {exc}"
            self._terminate()
            self.status = "fallback_builtin"
            return False
        return True

    def _start_threads(self) -> None:
        if self._process is None:
            return
        if self._reader_thread is None or not self._reader_thread.is_alive():
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name=f"mcp-reader-{self.name}",
                daemon=True,
            )
            self._reader_thread.start()
        if (
            self._stderr_thread is None
            and self._process.stderr is not None
        ):
            self._stderr_thread = threading.Thread(
                target=self._stderr_loop,
                name=f"mcp-stderr-{self.name}",
                daemon=True,
            )
            self._stderr_thread.start()

    def close(self) -> None:
        self._terminate()
        self.status = "closed"

    def _terminate(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.5)
            except Exception:  # noqa: BLE001 - best-effort shutdown
                try:
                    process.kill()
                except Exception:
                    pass
        for pending in list(self._pending.values()):
            try:
                pending.put_nowait(self._EOF)
            except queue.Full:
                pass
        self._pending.clear()
        for thread in (self._reader_thread, self._stderr_thread):
            if (
                thread is not None
                and thread.is_alive()
                and thread is not threading.current_thread()
            ):
                thread.join(timeout=1.0)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        process = self._process
        if process is None or process.stdin is None:
            raise MCPError("MCP stdio process is not running.")

        request_id = next(self._ids)
        response_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._pending[request_id] = response_queue
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        try:
            with self._write_lock:
                process.stdin.write(json.dumps(payload) + "\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._pending.pop(request_id, None)
            raise MCPError(f"Failed to write MCP request: {exc}") from exc

        try:
            message = response_queue.get(timeout=self.timeout)
        except queue.Empty as exc:
            self._pending.pop(request_id, None)
            raise MCPTimeoutError(
                f"MCP request '{method}' timed out after {self.timeout:.1f}s."
            ) from exc
        finally:
            self._pending.pop(request_id, None)

        if message is self._EOF:
            raise MCPError("MCP server closed the stream before responding.")
        if not isinstance(message, Mapping):
            raise MCPError("MCP server returned a non-object response.")
        error = message.get("error")
        if error:
            if isinstance(error, Mapping):
                detail = error.get("message") or error.get("data") or error
            else:
                detail = error
            raise MCPError(f"MCP error from '{method}': {detail}")
        return message.get("result")

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        payload = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        try:
            with self._write_lock:
                process.stdin.write(json.dumps(payload) + "\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            return

    # ------------------------------------------------------------------
    # Reader threads
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._stderr_lines.append(f"non-json stdout: {line[:300]}")
                    continue
                if not isinstance(message, Mapping):
                    continue
                if "id" in message and message.get("id") is not None:
                    pending = self._pending.get(message.get("id"))
                    if pending is not None:
                        try:
                            pending.put_nowait(message)
                        except queue.Full:
                            pass
                else:
                    self.notifications.append(dict(message))
        finally:
            for pending in list(self._pending.values()):
                try:
                    pending.put_nowait(self._EOF)
                except queue.Full:
                    pass

    def _stderr_loop(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for line in process.stderr:
                text = line.rstrip()
                if text:
                    self._stderr_lines.append(text[:500])
        except (OSError, ValueError):
            return

    @property
    def stderr_tail(self) -> list[str]:
        return list(self._stderr_lines)

    @property
    def is_running(self) -> bool:
        return bool(self._process and self._process.poll() is None)


class MCPHTTPClient(MCPClientBase):
    """JSON-RPC over HTTP, accepting JSON or SSE responses."""

    def __init__(
        self,
        url: str,
        *,
        name: str = "mcp-http",
        headers: Mapping[str, str] | None = None,
        timeout: float = 12.0,
        client: Any | None = None,
    ) -> None:
        super().__init__(name=name, timeout=timeout)
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **dict(headers or {}),
        }
        self._client = client
        self._ids = count(1)

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def start(self) -> bool:
        if self.status == "connected":
            return True
        try:
            self.initialize()
        except Exception as exc:  # noqa: BLE001 - handshake failures are non-fatal
            self.last_error = f"MCP HTTP handshake failed: {exc}"
            self.status = "fallback_builtin"
            return False
        return True

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001 - best-effort
                pass
        self.status = "closed"

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        request_id = next(self._ids)
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        response = self._get_client().post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        if getattr(response, "status_code", 200) >= 400:
            raise MCPError(
                f"MCP HTTP request failed with status {response.status_code}."
            )
        message = self._parse_http_response(response, request_id)
        if not isinstance(message, Mapping):
            raise MCPError("MCP HTTP server returned a non-object response.")
        error = message.get("error")
        if error:
            raise MCPError(f"MCP error from '{method}': {error}")
        return message.get("result")

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        try:
            self._get_client().post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
        except Exception:  # noqa: BLE001 - notifications are best-effort
            return

    @staticmethod
    def _parse_http_response(response: Any, request_id: int) -> Any:
        content_type = str(
            getattr(response, "headers", {}).get("content-type", "")
        ).lower()
        if "text/event-stream" in content_type:
            text = response.text
            for line in text.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, Mapping):
                    if message.get("id") == request_id or "result" in message:
                        return message
            return None
        return response.json()


@dataclass(slots=True)
class MCPServer:
    """One managed MCP server."""
    name: str
    client: MCPClientBase
    tools: list[MCPTool] = field(default_factory=list)

    @property
    def status(self) -> str:
        return self.client.status


class MCPClientManager:
    """Manage multiple MCP servers and expose their tools to Axe."""

    def __init__(self) -> None:
        self.servers: dict[str, MCPServer] = {}
        self._status = "fallback_builtin"

    # ------------------------------------------------------------------
    # Server management
    # ------------------------------------------------------------------

    def add_client(self, name: str, client: MCPClientBase) -> MCPClientBase:
        self.servers[name] = MCPServer(name=name, client=client)
        return client

    def add_stdio_server(
        self,
        name: str,
        command: str | Sequence[str],
        args: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> MCPStdioClient:
        client = MCPStdioClient(
            command,
            args,
            name=name,
            **kwargs,
        )
        self.add_client(name, client)
        return client

    def add_http_server(self, name: str, url: str, **kwargs: Any) -> MCPHTTPClient:
        client = MCPHTTPClient(url, name=name, **kwargs)
        self.add_client(name, client)
        return client

    def add_sse_server(self, name: str, url: str, **kwargs: Any) -> MCPHTTPClient:
        """Alias for streamable HTTP/SSE servers."""
        return self.add_http_server(name, url, **kwargs)

    def start_all(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for name, server in self.servers.items():
            try:
                started = server.client.start()
            except Exception as exc:  # noqa: BLE001 - startup is non-fatal
                server.client.status = "fallback_builtin"
                server.client.last_error = str(exc)
                started = False
            results[name] = started
            if started:
                try:
                    server.tools = server.client.list_tools()
                except Exception as exc:  # noqa: BLE001 - discovery is best-effort
                    server.client.last_error = f"tool discovery failed: {exc}"
                    server.tools = []
        self._refresh_status()
        return results

    def start(self) -> dict[str, bool]:
        """Alias for :meth:`start_all`."""
        return self.start_all()

    def discover_tools(self) -> dict[str, list[MCPTool]]:
        for name, server in self.servers.items():
            if server.client.status != "connected":
                continue
            try:
                server.tools = server.client.list_tools()
            except Exception as exc:  # noqa: BLE001 - discovery is best-effort
                server.client.last_error = f"tool discovery failed: {exc}"
        return {name: list(server.tools) for name, server in self.servers.items()}

    def close_all(self) -> None:
        for server in self.servers.values():
            try:
                server.client.close()
            except Exception:  # noqa: BLE001 - shutdown is best-effort
                pass
        self._refresh_status()

    def close(self) -> None:
        """Alias for :meth:`close_all`."""
        self.close_all()

    # ------------------------------------------------------------------
    # Tool access
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[MCPTool]:
        tools: list[MCPTool] = []
        for server in self.servers.values():
            tools.extend(server.tools)
        return tools

    def list_tools(self, server: str | None = None) -> list[MCPTool]:
        if server is not None:
            entry = self.servers.get(server)
            return list(entry.tools) if entry else []
        return self.tools

    def find_tool(
        self,
        tool_name: str,
        server: str | None = None,
    ) -> tuple[str, MCPTool] | None:
        for name, entry in self.servers.items():
            if server is not None and name != server:
                continue
            for tool in entry.tools:
                if tool.name == tool_name:
                    return name, tool
        return None

    def find_scan_tool(
        self,
        names: Sequence[str] | None = None,
    ) -> tuple[str, str] | None:
        candidates = set(names or ())
        for name, entry in self.servers.items():
            for tool in entry.tools:
                if not candidates or tool.name in candidates:
                    return name, tool.name
        return None

    def has_tool(self, tool_name: str) -> bool:
        return self.find_tool(tool_name) is not None

    def has_scan_tool(self, names: Sequence[str] | None = None) -> bool:
        return self.find_scan_tool(names) is not None

    def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        server: str | None = None,
    ) -> Any:
        found = self.find_tool(tool_name, server=server)
        if found is None:
            raise MCPError(f"MCP tool '{tool_name}' was not discovered.")
        server_name, _tool = found
        return self.servers[server_name].client.call_tool(tool_name, arguments)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def server_statuses(self) -> dict[str, str]:
        return {name: server.status for name, server in self.servers.items()}

    @property
    def connected(self) -> bool:
        return any(server.status == "connected" for server in self.servers.values())

    @property
    def status(self) -> str:
        self._refresh_status()
        return self._status

    def _refresh_status(self) -> None:
        statuses = [server.status for server in self.servers.values()]
        if any(status == "connected" for status in statuses):
            self._status = "connected"
        elif any(status == "fallback_builtin" for status in statuses):
            self._status = "fallback_builtin"
        elif statuses and all(status == "closed" for status in statuses):
            self._status = "closed"
        elif statuses:
            self._status = "disconnected"
        else:
            self._status = "fallback_builtin"


# Backward-compatible name for HTTP/SSE transport users.
MCPSSEClient = MCPHTTPClient
