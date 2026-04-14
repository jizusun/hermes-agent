"""Kiro CLI ACP client using the official ACP Python SDK.

Spawns `kiro-cli acp`, drives it via the typed SDK, and exposes an
OpenAI-compatible chat.completions.create() interface for Hermes.
"""

from __future__ import annotations

import asyncio
import os
import signal
import shlex
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from acp import PROTOCOL_VERSION, Client, connect_to_agent, text_block
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ClientCapabilities,
    Implementation,
    PermissionOption,
    ReadTextFileResponse,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallStart,
    ToolCallProgress,
    WriteTextFileResponse,
)

from agent.copilot_acp_client import (
    _extract_tool_calls_from_text,
    _format_messages_as_prompt,
    _ensure_path_within_cwd,
    _ACPChatNamespace,
)

ACP_MARKER_BASE_URL = "acp://kiro"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def _kill_proc_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the process and its entire process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _resolve_command() -> str:
    return os.getenv("HERMES_KIRO_ACP_COMMAND", "").strip() or "kiro-cli"


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_KIRO_ACP_ARGS", "").strip()
    return shlex.split(raw) if raw else ["acp"]


class _KiroClient(Client):
    """ACP Client handler that collects text/thought chunks and handles server requests."""

    def __init__(self, cwd: str):
        self.text_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self._cwd = cwd

    async def request_permission(self, options: list[PermissionOption], **kwargs: Any) -> RequestPermissionResponse:
        allow = next((o for o in options if o.kind in ("allow_once", "allow_always")), options[0])
        return RequestPermissionResponse(outcome={"outcome": "selected", "optionId": allow.option_id})

    async def read_text_file(self, path: str, session_id: str, limit: int | None = None, line: int | None = None, **kwargs: Any) -> ReadTextFileResponse:
        resolved = _ensure_path_within_cwd(path, self._cwd)
        content = resolved.read_text() if resolved.exists() else ""
        if isinstance(line, int) and line > 1:
            lines = content.splitlines(keepends=True)
            start = line - 1
            end = start + limit if isinstance(limit, int) and limit > 0 else None
            content = "".join(lines[start:end])
        return ReadTextFileResponse(content=content)

    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs: Any) -> WriteTextFileResponse | None:
        resolved = _ensure_path_within_cwd(path, self._cwd)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return None

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        if isinstance(update, AgentMessageChunk) and isinstance(update.content, TextContentBlock):
            self.text_parts.append(update.content.text)
        elif isinstance(update, AgentThoughtChunk) and isinstance(update.content, TextContentBlock):
            self.reasoning_parts.append(update.content.text)

    async def ext_notification(self, method: str, params: dict) -> None:
        pass

    async def ext_method(self, method: str, params: dict) -> dict:
        return {}


class KiroACPClient:
    """OpenAI-client-compatible facade for kiro-cli ACP using the official SDK.
    
    Keeps a persistent process and session across requests.
    """

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, acp_command: str | None = None, acp_args: list[str] | None = None, acp_cwd: str | None = None, **_: Any):
        self.api_key = api_key or "kiro-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._command = acp_command or _resolve_command()
        self._args = list(acp_args or _resolve_args())
        self._cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._conn: Any = None
        self._session_id: str | None = None
        self._client: _KiroClient | None = None
        # Eagerly start kiro-cli acp
        self._run(self._ensure_connected(30.0))

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._loop_thread.start()
        return self._loop

    def _run(self, coro):
        """Run a coroutine on the persistent event loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._ensure_loop()).result()

    async def _ensure_connected(self, timeout: float) -> None:
        """Ensure we have a live process, connection, and session."""
        if self._proc is not None and self._proc.returncode is None and self._session_id:
            return  # already connected

        # Clean up any dead process
        if self._proc is not None:
            _kill_proc_tree(self._proc)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass

        self._client = _KiroClient(self._cwd)
        self._proc = await asyncio.create_subprocess_exec(
            self._command, *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=50 * 1024 * 1024,
            start_new_session=True,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("kiro-cli ACP process did not expose stdio pipes.")

        self._conn = connect_to_agent(self._client, self._proc.stdin, self._proc.stdout)

        await asyncio.wait_for(self._conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(fs={"readTextFile": True, "writeTextFile": True}, terminal=True),
            client_info=Implementation(name="hermes-agent", title="Hermes Agent", version="0.0.0"),
        ), timeout=timeout)

        session = await asyncio.wait_for(self._conn.new_session(mcp_servers=[], cwd=self._cwd), timeout=timeout)
        self._session_id = session.session_id

    def close(self) -> None:
        self.is_closed = True
        if self._conn:
            try:
                self._run(self._conn.close())
            except Exception:
                pass
        if self._proc is not None and self._proc.returncode is None:
            _kill_proc_tree(self._proc)
            try:
                self._run(asyncio.wait_for(self._proc.wait(), timeout=3))
            except Exception:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
        self._proc = None
        self._conn = None
        self._session_id = None
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None

    def _extract_latest_user_message(self, messages: list[dict[str, Any]]) -> str:
        """Extract the last user message text from the messages array."""
        for msg in reversed(messages or []):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    return "\n".join(parts)
        return ""

    def _create_chat_completion(self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None, timeout: float | None = None, tools: list[dict[str, Any]] | None = None, tool_choice: Any = None, **_: Any) -> Any:
        prompt_text = self._extract_latest_user_message(messages or [])
        if not prompt_text:
            # Fallback: format full history (e.g. system-only or tool messages)
            prompt_text = _format_messages_as_prompt(messages or [], model=model, tools=tools, tool_choice=tool_choice)

        loop = self._ensure_loop()
        response_text, reasoning_text = self._run(
            self._send_prompt(prompt_text, timeout_seconds=float(timeout or _DEFAULT_TIMEOUT_SECONDS))
        )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0, prompt_tokens_details=SimpleNamespace(cached_tokens=0))
        message = SimpleNamespace(content=cleaned_text, tool_calls=tool_calls, reasoning=reasoning_text or None, reasoning_content=reasoning_text or None, reasoning_details=None)
        finish_reason = "tool_calls" if tool_calls else "stop"
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=usage, model=model or "kiro-acp")

    async def _send_prompt(self, prompt_text: str, *, timeout_seconds: float) -> tuple[str, str]:
        await self._ensure_connected(timeout_seconds)

        # Reset text collectors for this turn
        self._client.text_parts.clear()
        self._client.reasoning_parts.clear()

        try:
            await asyncio.wait_for(self._conn.prompt(
                session_id=self._session_id,
                prompt=[text_block(prompt_text)],
            ), timeout=timeout_seconds)
        except Exception:
            # Connection died — reset so next call reconnects
            self._session_id = None
            raise

        return "".join(self._client.text_parts), "".join(self._client.reasoning_parts)


async def _query_kiro_models(command: str, args: list[str], cwd: str, timeout: float = 30.0) -> list[str]:
    """Connect to kiro-cli acp and read available models from session/new response."""
    proc = await asyncio.create_subprocess_exec(
        command, *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=50 * 1024 * 1024,
        start_new_session=True,
    )
    if proc.stdin is None or proc.stdout is None:
        return []

    client = _KiroClient(cwd)
    conn = connect_to_agent(client, proc.stdin, proc.stdout)

    try:
        await asyncio.wait_for(conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(),
            client_info=Implementation(name="hermes-agent", title="Hermes Agent", version="0.0.0"),
        ), timeout=timeout)

        session = await asyncio.wait_for(conn.new_session(mcp_servers=[], cwd=cwd), timeout=timeout)

        if session.models and session.models.available_models:
            return [m.model_id for m in session.models.available_models if m.model_id]
        return []
    except Exception:
        return []
    finally:
        try:
            await conn.close()
        except Exception:
            pass
        _kill_proc_tree(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except asyncio.TimeoutError:
            proc.kill()


def fetch_kiro_models() -> list[str]:
    """Fetch available models from kiro-cli via ACP. Returns model ID list or empty."""
    command = _resolve_command()
    args = _resolve_args()
    cwd = str(Path.cwd())
    try:
        import logging
        logging.disable(logging.CRITICAL)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_query_kiro_models(command, args, cwd))
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            logging.disable(logging.NOTSET)
            # Suppress CPython bpo-41320 BaseSubprocessTransport.__del__ noise
            _saved_fd = os.dup(2)
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
            import gc; gc.collect()
            os.dup2(_saved_fd, 2)
            os.close(_saved_fd)
        return result
    except Exception:
        return []
