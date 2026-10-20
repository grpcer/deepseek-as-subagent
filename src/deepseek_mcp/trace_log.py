"""Per-run visibility into model reasoning and tool activity.

Two independent outputs share the hooks below:

- ``progress_sink``: one-line status updates for the host, sent as MCP progress
  notifications during synchronous delegations. Only the host that issued the
  task receives them.
- The opt-in trace file, disabled unless ``"trace": true`` is set in
  config.json. Unlike ``server.log`` it contains model reasoning, assistant
  text, file paths, search patterns and truncated Bash commands, so it is a
  separate private file that users can ``tail -f`` while a delegation runs.

File contents and Write/Edit payloads are never included in either output;
tool results are reduced to their size and error status.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Callable

from .private_logging import PrivateBoundedLogStream

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 16_000
MAX_ARG_CHARS = 300
MAX_PROGRESS_CHARS = 160

# Only metadata-like arguments from tool_schemas.py; content/old_string/
# new_string/new_source are deliberately excluded.
_SUMMARY_ARGS = ("path", "pattern", "glob", "offset", "limit")

# asyncio.to_thread copies contextvars, so the sink set by the MCP handler is
# visible in the worker thread that runs the agent loop.
progress_sink: ContextVar[Callable[[str], None] | None] = ContextVar(
    "deepseek_progress_sink", default=None
)

_lock = threading.Lock()
_stream: PrivateBoundedLogStream | None = None
_unavailable = False


def trace_path() -> Path:
    return Path.home() / ".deepseek-mcp" / "trace.log"


def _open_stream() -> PrivateBoundedLogStream | None:
    global _stream, _unavailable
    if _stream is not None or _unavailable:
        return _stream
    try:
        _stream = PrivateBoundedLogStream(trace_path(), rotate_on_full=True)
    except Exception as error:
        # Tracing is diagnostic only; a broken trace file must not fail the run.
        _unavailable = True
        logger.warning("DeepSeek trace log disabled: %s", type(error).__name__)
    return _stream


def _enabled(config) -> bool:
    return getattr(config, "trace", False) is True


def _notify(text: str) -> None:
    sink = progress_sink.get()
    if sink is None:
        return
    try:
        sink(_truncate(" ".join(text.split()), MAX_PROGRESS_CHARS))
    except Exception as error:
        logger.warning("DeepSeek progress update failed: %s", type(error).__name__)


def mcp_progress(ctx, loop: asyncio.AbstractEventLoop) -> Callable[[str], None] | None:
    """Build a thread-safe sink that forwards updates as MCP progress notifications."""
    if ctx is None:
        return None
    counter = itertools.count(1)

    def send(text: str) -> None:
        future = asyncio.run_coroutine_threadsafe(
            ctx.report_progress(next(counter), None, text), loop
        )
        # Progress is best effort: never block the agent or surface its failures.
        future.add_done_callback(lambda done: done.cancelled() or done.exception())

    return send


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [+{len(text) - limit} chars]"


def _emit(lines: list[str]) -> None:
    stamp = time.strftime("%H:%M:%S")
    entry = "".join(f"{stamp} {line}\n" for line in lines)
    with _lock:
        stream = _open_stream()
        if stream is None:
            return
        try:
            stream.write(entry)
        except Exception as error:
            logger.warning("DeepSeek trace write failed: %s", type(error).__name__)


def _block(label: str, text: object) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    body = _truncate(text.strip(), MAX_TEXT_CHARS)
    return [f"  {label}:", *(f"    {line}" for line in body.splitlines())]


def run_start(config, task: str) -> None:
    _notify(f"DeepSeek started · {config.model} · effort={config.reasoning_effort}")
    if not _enabled(config):
        return
    _emit([
        f"=== run start · model={config.model} effort={config.reasoning_effort} "
        f"capability={config.delegation_capability} pid={os.getpid()}",
        *_block("task", task),
    ])


def response(config, turn: int, assistant_message: dict) -> None:
    calls = assistant_message.get("tool_calls") or []
    thinking = assistant_message.get("reasoning_content")
    if isinstance(thinking, str) and thinking.strip():
        _notify(f"Turn {turn} · thinking: {thinking.strip()}")
    elif not calls:
        _notify(f"Turn {turn} · writing final answer")
    if not _enabled(config):
        return
    _emit([
        f"--- turn {turn} · {len(calls)} tool call(s)",
        *_block("thinking", assistant_message.get("reasoning_content")),
        *_block("reply", assistant_message.get("content")),
    ])


def describe_tool_args(tool_name: str, args: dict) -> str:
    """Summarize tool arguments without file contents or edit payloads."""
    parts = []
    if tool_name == "Bash":
        command = args.get("command")
        if isinstance(command, str):
            parts.append(_truncate(" ".join(command.split()), MAX_ARG_CHARS))
    for key in _SUMMARY_ARGS:
        value = args.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) or (isinstance(value, str) and value):
            parts.append(f"{key}={_truncate(str(value), MAX_ARG_CHARS)}")
    return " ".join(parts)


def tool(config, tool_call, result: object) -> None:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except (TypeError, ValueError):
        args = None
    if not isinstance(args, dict):
        args = {}
    if isinstance(result, str):
        status = "ERROR" if result.startswith("ERROR") else "ok"
        outcome = f"{status}, {len(result)} chars"
    else:
        outcome = "no result"
    summary = f"{name} {describe_tool_args(name, args)}".strip()
    _notify(f"{summary} [{outcome}]")
    if _enabled(config):
        _emit([f"  → {summary} [{outcome}]"])


def run_result(config, result: dict) -> None:
    summary = (
        f"done · turns={result['turns_used']} "
        f"tool_calls={result['tool_calls']} tokens={result['tokens']['total']}"
    )
    _notify(f"DeepSeek {summary}")
    if _enabled(config):
        _emit([f"=== run end · {summary}"])


def run_error(config, error: BaseException) -> None:
    _notify(f"DeepSeek stopped · {type(error).__name__}")
    if _enabled(config):
        _emit([f"=== run end · stopped · {type(error).__name__}: {error}"])
