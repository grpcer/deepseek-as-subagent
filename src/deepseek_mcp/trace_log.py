"""Opt-in per-run trace of model reasoning and tool activity.

Disabled unless ``"trace": true`` is set in config.json. Unlike ``server.log``,
the trace contains model reasoning, assistant text, file paths, search patterns
and truncated Bash commands, so it is a separate private file that users can
``tail -f`` while a delegation runs. File contents and Write/Edit payloads are
never traced; tool results are reduced to their size and error status.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from .private_logging import PrivateBoundedLogStream

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 16_000
MAX_ARG_CHARS = 300

# Only metadata-like arguments from tool_schemas.py; content/old_string/
# new_string/new_source are deliberately excluded.
_SUMMARY_ARGS = ("path", "pattern", "glob", "offset", "limit")

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
    if not _enabled(config):
        return
    _emit([
        f"=== run start · model={config.model} effort={config.reasoning_effort} "
        f"capability={config.delegation_capability} pid={os.getpid()}",
        *_block("task", task),
    ])


def response(config, turn: int, assistant_message: dict) -> None:
    if not _enabled(config):
        return
    calls = assistant_message.get("tool_calls") or []
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
    if not _enabled(config):
        return
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
    _emit([f"  → {name} {describe_tool_args(name, args)} [{outcome}]"])


def run_result(config, result: dict) -> None:
    if not _enabled(config):
        return
    _emit([
        f"=== run end · done · turns={result['turns_used']} "
        f"tool_calls={result['tool_calls']} tokens={result['tokens']['total']}"
    ])


def run_error(config, error: BaseException) -> None:
    if not _enabled(config):
        return
    _emit([f"=== run end · stopped · {type(error).__name__}: {error}"])
