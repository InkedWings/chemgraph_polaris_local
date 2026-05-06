"""Lightweight JSONL profiling for ChemGraph runs."""

from __future__ import annotations

import datetime as dt
import functools
import json
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable


_WRAPPED_TOOL_IDS: set[int] = set()


def _events_path() -> str | None:
    path = os.environ.get("CHEMGRAPH_PROFILE_EVENTS")
    return path or None


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    return repr(value)


def _preview(value: Any, limit: int = 2000) -> str:
    try:
        text = json.dumps(_jsonable(value), ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _base_event(kind: str, name: str, span_id: str) -> dict[str, Any]:
    return {
        "epoch": time.time(),
        "iso_time": _utc_now(),
        "pid": os.getpid(),
        "kind": kind,
        "name": name,
        "span_id": span_id,
        "worker_id": os.environ.get("CHEMGRAPH_PROFILE_WORKER_ID"),
        "task_id": os.environ.get("CHEMGRAPH_PROFILE_TASK_ID"),
        "log_dir": os.environ.get("CHEMGRAPH_LOG_DIR"),
    }


def write_event(event: dict[str, Any]) -> None:
    path = _events_path()
    if not path:
        return

    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Profiling must never change ChemGraph behavior.
        return


def _token_usage(response: Any) -> dict[str, Any]:
    usage = {}
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        usage["usage_metadata"] = _jsonable(usage_metadata)

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if token_usage:
            usage["token_usage"] = _jsonable(token_usage)
        if response_metadata.get("finish_reason") is not None:
            usage["finish_reason"] = response_metadata.get("finish_reason")
        if response_metadata.get("model_name") is not None:
            usage["model_name"] = response_metadata.get("model_name")
    return usage


def _message_count(messages: Any) -> int | None:
    if isinstance(messages, list):
        return len(messages)
    return None


def profile_llm_invoke(
    runnable: Any,
    messages: Any,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Invoke a LangChain runnable and emit start/end/error events."""
    span_id = str(uuid.uuid4())
    start_perf = time.perf_counter()
    start = _base_event("llm", name, span_id)
    start.update(
        {
            "event": "start",
            "message_count": _message_count(messages),
            "metadata": _jsonable(metadata or {}),
        }
    )
    write_event(start)

    try:
        response = runnable.invoke(messages)
    except Exception as exc:
        event = _base_event("llm", name, span_id)
        event.update(
            {
                "event": "error",
                "duration_seconds": time.perf_counter() - start_perf,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        )
        write_event(event)
        raise

    event = _base_event("llm", name, span_id)
    event.update(
        {
            "event": "end",
            "duration_seconds": time.perf_counter() - start_perf,
            "status": "success",
            "response_type": type(response).__name__,
        }
    )
    event.update(_token_usage(response))
    write_event(event)
    return response


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", type(tool).__name__))


def _profile_tool_call(
    tool_name: str,
    fn: Callable[..., Any],
    tool_input: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    span_id = str(uuid.uuid4())
    start_perf = time.perf_counter()
    start = _base_event("tool", tool_name, span_id)
    start.update(
        {
            "event": "start",
            "input_preview": _preview(tool_input),
        }
    )
    write_event(start)

    try:
        result = fn(tool_input, *args, **kwargs)
    except Exception as exc:
        event = _base_event("tool", tool_name, span_id)
        event.update(
            {
                "event": "error",
                "duration_seconds": time.perf_counter() - start_perf,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        )
        write_event(event)
        raise

    event = _base_event("tool", tool_name, span_id)
    event.update(
        {
            "event": "end",
            "duration_seconds": time.perf_counter() - start_perf,
            "status": "success",
            "result_type": type(result).__name__,
            "result_preview": _preview(result),
        }
    )
    write_event(event)
    return result


def profile_tool(tool: Any) -> Any:
    """Wrap a LangChain tool object's invoke methods once."""
    if id(tool) in _WRAPPED_TOOL_IDS or getattr(tool, "_chemgraph_profile_wrapped", False):
        return tool

    tool_name = _tool_name(tool)
    original_invoke = getattr(tool, "invoke", None)
    if getattr(original_invoke, "_chemgraph_profile_wrapper", False):
        _WRAPPED_TOOL_IDS.add(id(tool))
        return tool
    if callable(original_invoke):

        @functools.wraps(original_invoke)
        def invoke(tool_input: Any, *args: Any, **kwargs: Any) -> Any:
            return _profile_tool_call(tool_name, original_invoke, tool_input, *args, **kwargs)

        setattr(invoke, "_chemgraph_profile_wrapper", True)
        object.__setattr__(tool, "invoke", invoke)

    object.__setattr__(tool, "_chemgraph_profile_wrapped", True)
    _WRAPPED_TOOL_IDS.add(id(tool))
    return tool


def profile_tools(tools: list[Any] | None) -> list[Any] | None:
    if tools is None:
        return None
    return [profile_tool(tool) for tool in tools]
