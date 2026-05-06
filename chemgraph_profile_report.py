#!/usr/bin/env python3
"""Summarize ChemGraph concurrency profiling artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROM_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)$")
VLLM_LOG_LINE = re.compile(
    r"INFO (?P<month>\d{2})-(?P<day>\d{2}) (?P<hms>\d{2}:\d{2}:\d{2}).*"
    r"Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[0-9.]+) tokens/s, "
    r"Running: (?P<running>[0-9]+) reqs, Waiting: (?P<waiting>[0-9]+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%, Prefix cache hit rate: (?P<prefix>[0-9.]+)%"
)
TOOL_CALL_LINE = re.compile(r"^\s{2}([A-Za-z_][A-Za-z0-9_]*) \(chatcmpl-tool", re.MULTILINE)
TOOL_MESSAGE_NAME = re.compile(r"^Name: ([A-Za-z_][A-Za-z0-9_]*)\s*$", re.MULTILINE)
SESSION_ID = re.compile(r"session_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[A-Za-z0-9]+")


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)


def stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "p50": percentile(clean, 0.50),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "max": max(clean),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_labels(label_text: str | None) -> dict[str, str]:
    if not label_text:
        return {}
    body = label_text.strip("{}")
    labels = {}
    for part in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"', body):
        labels[part.group(1)] = part.group(2)
    return labels


def parse_prometheus(path: Path) -> list[tuple[str, dict[str, str], float]]:
    rows = []
    for line in read_text(path).splitlines():
        match = PROM_LINE.match(line.strip())
        if not match:
            continue
        metric, labels, value = match.groups()
        try:
            rows.append((metric, parse_labels(labels), float(value)))
        except ValueError:
            continue
    return rows


def sum_metric(rows: list[tuple[str, dict[str, str], float]], metric: str, labels: dict[str, str] | None = None) -> float:
    total = 0.0
    labels = labels or {}
    for name, row_labels, value in rows:
        if name != metric:
            continue
        if all(row_labels.get(key) == expected for key, expected in labels.items()):
            total += value
    return total


def histogram_buckets(rows: list[tuple[str, dict[str, str], float]], metric: str) -> dict[float, float]:
    buckets: dict[float, float] = defaultdict(float)
    bucket_metric = f"{metric}_bucket"
    for name, labels, value in rows:
        if name != bucket_metric or "le" not in labels:
            continue
        le = labels["le"]
        upper = math.inf if le == "+Inf" else float(le)
        buckets[upper] += value
    return buckets


def histogram_quantile(delta_buckets: dict[float, float], q: float) -> float | None:
    if not delta_buckets:
        return None
    total = delta_buckets.get(math.inf)
    if total is None:
        total = max(delta_buckets.values())
    if total <= 0:
        return None

    rank = q * total
    prev_bound = 0.0
    prev_count = 0.0
    for bound in sorted(delta_buckets):
        count = delta_buckets[bound]
        if count >= rank:
            if math.isinf(bound):
                return prev_bound
            if count <= prev_count:
                return bound
            fraction = (rank - prev_count) / (count - prev_count)
            return prev_bound + (bound - prev_bound) * fraction
        prev_bound = bound
        prev_count = count
    return None


def histogram_delta(before: list[tuple[str, dict[str, str], float]], after: list[tuple[str, dict[str, str], float]], metric: str) -> dict[str, float | int | None]:
    before_buckets = histogram_buckets(before, metric)
    after_buckets = histogram_buckets(after, metric)
    all_bounds = set(before_buckets) | set(after_buckets)
    deltas = {
        bound: max(0.0, after_buckets.get(bound, 0.0) - before_buckets.get(bound, 0.0))
        for bound in all_bounds
    }
    return {
        "count": int(deltas.get(math.inf, max(deltas.values()) if deltas else 0.0)),
        "p50": histogram_quantile(deltas, 0.50),
        "p95": histogram_quantile(deltas, 0.95),
        "p99": histogram_quantile(deltas, 0.99),
    }


def parse_task_log(path: Path) -> dict[str, object]:
    text = read_text(path)
    tool_calls = TOOL_CALL_LINE.findall(text)
    tool_messages = list(TOOL_MESSAGE_NAME.finditer(text))
    failures = 0

    for index, match in enumerate(tool_messages):
        start = match.end()
        end = tool_messages[index + 1].start() if index + 1 < len(tool_messages) else len(text)
        section = text[start:end].lower()
        if (
            section.strip().startswith("error")
            or '"status": "failure"' in section
            or "traceback" in section
            or "calculationfailed" in section
            or "failed with command" in section
            or "mpi_abort" in section
        ):
            failures += 1

    sessions = sorted(set(SESSION_ID.findall(text)))
    return {
        "llm_messages": text.count("================================== Ai Message"),
        "llm_tool_calls": len(tool_calls),
        "tool_execs": len(tool_messages),
        "tool_successes": max(0, len(tool_messages) - failures),
        "tool_failures": failures,
        "invalid_tool_markers": text.count("<tool_call>") + text.count("</tool_call>"),
        "tool_names": ";".join(tool_calls),
        "session_ids": ";".join(sessions),
    }


def parse_event_log(path: Path) -> dict[str, object]:
    llm_time = 0.0
    tool_time = 0.0
    llm_events = 0
    tool_events = 0
    llm_errors = 0
    tool_errors = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    if not path.exists():
        return {
            "event_log": str(path),
            "profile_events_found": 0,
            "llm_calls_profiled": 0,
            "llm_time_seconds": 0.0,
            "llm_error_events": 0,
            "tool_calls_profiled": 0,
            "tool_time_seconds": 0.0,
            "tool_error_events": 0,
            "input_tokens_profiled": 0,
            "output_tokens_profiled": 0,
            "total_tokens_profiled": 0,
        }

    for line in read_text(path).splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") not in {"end", "error"}:
            continue

        duration = as_float(str(event.get("duration_seconds", ""))) or 0.0
        kind = event.get("kind")
        if kind == "llm":
            llm_events += 1
            llm_time += duration
            if event.get("event") == "error":
                llm_errors += 1
            usage = event.get("usage_metadata") or {}
            token_usage = event.get("token_usage") or {}
            input_tokens += int(
                usage.get("input_tokens")
                or token_usage.get("prompt_tokens")
                or 0
            )
            output_tokens += int(
                usage.get("output_tokens")
                or token_usage.get("completion_tokens")
                or 0
            )
            total_tokens += int(
                usage.get("total_tokens")
                or token_usage.get("total_tokens")
                or 0
            )
        elif kind == "tool":
            tool_events += 1
            tool_time += duration
            if event.get("event") == "error":
                tool_errors += 1

    return {
        "event_log": str(path),
        "profile_events_found": int(path.exists()),
        "llm_calls_profiled": llm_events,
        "llm_time_seconds": llm_time,
        "llm_error_events": llm_errors,
        "tool_calls_profiled": tool_events,
        "tool_time_seconds": tool_time,
        "tool_error_events": tool_errors,
        "input_tokens_profiled": input_tokens,
        "output_tokens_profiled": output_tokens,
        "total_tokens_profiled": total_tokens,
    }


def query_from_file(path: Path) -> str:
    text = read_text(path)
    for line in text.splitlines():
        if line.startswith("query: "):
            return line.split("query: ", 1)[1]
    return ""


def load_task_times(smoke_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    times = {}
    for path in smoke_root.glob("worker_*/task_times.csv"):
        worker = path.parent.name
        for row in read_csv(path):
            times[(worker, row.get("id", ""))] = row
    return times


def build_task_profile(run_root: Path) -> list[dict[str, object]]:
    smoke_root = run_root / "smoke"
    combined = read_csv(smoke_root / "combined_summary.csv")
    task_times = load_task_times(smoke_root)
    rows = []
    for row in combined:
        worker = row.get("worker", "")
        exp_id = row.get("id", "")
        log_path = Path(row.get("log", ""))
        query_path = Path(str(log_path) + ".query")
        timing = task_times.get((worker, exp_id), {})
        parsed = parse_task_log(log_path)
        event_path = Path(
            timing.get("event_log") or (str(log_path) + ".events.jsonl")
        )
        events = parse_event_log(event_path)
        task_wall_seconds = as_float(timing.get("end_epoch")) or 0.0
        task_start = as_float(timing.get("start_epoch")) or 0.0
        if task_wall_seconds and task_start:
            task_wall_seconds = task_wall_seconds - task_start
        else:
            task_wall_seconds = as_float(row.get("seconds")) or 0.0
        other_time = max(
            0.0,
            task_wall_seconds
            - float(events["llm_time_seconds"])
            - float(events["tool_time_seconds"]),
        )
        profile_status = row.get("status", "")
        if profile_status == "PASS" and (
            int(parsed["tool_failures"]) > 0
            or int(parsed["invalid_tool_markers"]) > 0
            or int(events["llm_error_events"]) > 0
            or int(events["tool_error_events"]) > 0
        ):
            profile_status = "FAIL"
        out = {
            "worker": worker,
            "id": exp_id,
            "status": row.get("status", ""),
            "profile_status": profile_status,
            "seconds": as_float(row.get("seconds")),
            "workflow": row.get("workflow", ""),
            "recursion_limit": row.get("recursion_limit", ""),
            "start_epoch": timing.get("start_epoch", ""),
            "end_epoch": timing.get("end_epoch", ""),
            "start_iso": timing.get("start_iso", ""),
            "end_iso": timing.get("end_iso", ""),
            "task_wall_seconds": task_wall_seconds,
            "query": query_from_file(query_path),
            "log": str(log_path),
            **parsed,
            **events,
            "other_time_seconds": other_time,
            "llm_time_pct": (
                float(events["llm_time_seconds"]) / task_wall_seconds
                if task_wall_seconds > 0
                else None
            ),
            "tool_time_pct": (
                float(events["tool_time_seconds"]) / task_wall_seconds
                if task_wall_seconds > 0
                else None
            ),
            "other_time_pct": other_time / task_wall_seconds if task_wall_seconds > 0 else None,
        }
        rows.append(out)
    return rows


def parse_vllm_log(run_root: Path) -> list[dict[str, object]]:
    year = None
    metadata = json.loads(read_text(run_root / "profile_metadata.json") or "{}")
    start_iso = str(metadata.get("profile_start_iso", ""))
    if len(start_iso) >= 4 and start_iso[:4].isdigit():
        year = start_iso[:4]

    samples = []
    for line in read_text(run_root / "vllm_backend_incremental.log").splitlines():
        match = VLLM_LOG_LINE.search(line)
        if not match:
            continue
        item = match.groupdict()
        iso = f"{year or '0000'}-{item['month']}-{item['day']}T{item['hms']}Z"
        samples.append(
            {
                "iso_time": iso,
                "prompt_toks_per_s": float(item["prompt"]),
                "generation_toks_per_s": float(item["generation"]),
                "running_reqs": int(item["running"]),
                "waiting_reqs": int(item["waiting"]),
                "gpu_kv_cache_usage_pct": float(item["kv"]),
                "prefix_cache_hit_rate_pct": float(item["prefix"]),
            }
        )
    return samples


def summarize_prometheus(run_root: Path, wall_seconds: float | None) -> dict[str, object]:
    before = parse_prometheus(run_root / "vllm_metrics_before.prom")
    after = parse_prometheus(run_root / "vllm_metrics_after.prom")

    prompt_delta = sum_metric(after, "vllm:prompt_tokens_total") - sum_metric(before, "vllm:prompt_tokens_total")
    generation_delta = sum_metric(after, "vllm:generation_tokens_total") - sum_metric(before, "vllm:generation_tokens_total")
    request_success_delta = sum_metric(after, "vllm:request_success_total") - sum_metric(before, "vllm:request_success_total")

    histograms = {}
    for metric in [
        "vllm:time_to_first_token_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:request_queue_time_seconds",
        "vllm:request_inference_time_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_decode_time_seconds",
        "vllm:request_time_per_output_token_seconds",
    ]:
        histograms[metric] = histogram_delta(before, after, metric)

    return {
        "prompt_tokens_delta": prompt_delta,
        "generation_tokens_delta": generation_delta,
        "request_success_delta": request_success_delta,
        "avg_prompt_toks_per_s_from_counters": prompt_delta / wall_seconds if wall_seconds and wall_seconds > 0 else None,
        "avg_generation_toks_per_s_from_counters": generation_delta / wall_seconds if wall_seconds and wall_seconds > 0 else None,
        "histograms": histograms,
    }


def summarize_resource_csv(path: Path, columns: list[str]) -> dict[str, object]:
    rows = read_csv(path)
    summary = {"rows": len(rows)}
    for column in columns:
        summary[column] = stats([as_float(row.get(column)) for row in rows])
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_num(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    metadata = json.loads(read_text(run_root / "profile_metadata.json") or "{}")
    start_epoch = as_float(str(metadata.get("profile_start_epoch", "")))
    end_epoch = as_float(str(metadata.get("profile_end_epoch", "")))
    wall_seconds = end_epoch - start_epoch if start_epoch is not None and end_epoch is not None else None

    task_rows = build_task_profile(run_root)
    write_csv(run_root / "task_profile.csv", task_rows)

    vllm_samples = parse_vllm_log(run_root)
    write_csv(run_root / "vllm_log_timeseries.csv", vllm_samples)

    vllm_log_summary = {
        "prompt_toks_per_s": stats([row["prompt_toks_per_s"] for row in vllm_samples]),
        "generation_toks_per_s": stats([row["generation_toks_per_s"] for row in vllm_samples]),
        "running_reqs": stats([row["running_reqs"] for row in vllm_samples]),
        "waiting_reqs": stats([row["waiting_reqs"] for row in vllm_samples]),
        "gpu_kv_cache_usage_pct": stats([row["gpu_kv_cache_usage_pct"] for row in vllm_samples]),
    }

    task_seconds = [row["seconds"] for row in task_rows if isinstance(row.get("seconds"), float)]
    task_total = len(task_rows)
    task_pass = sum(1 for row in task_rows if row.get("profile_status") == "PASS")
    tool_execs = sum(int(row.get("tool_execs") or 0) for row in task_rows)
    tool_successes = sum(int(row.get("tool_successes") or 0) for row in task_rows)
    profiled_tool_execs = sum(int(row.get("tool_calls_profiled") or 0) for row in task_rows)
    profiled_tool_errors = sum(int(row.get("tool_error_events") or 0) for row in task_rows)
    profiled_llm_calls = sum(int(row.get("llm_calls_profiled") or 0) for row in task_rows)
    profiled_llm_errors = sum(int(row.get("llm_error_events") or 0) for row in task_rows)
    event_rows = [row for row in task_rows if int(row.get("profile_events_found") or 0)]

    summary = {
        "metadata": metadata,
        "wall_seconds": wall_seconds,
        "tasks": {
            "total": task_total,
            "pass": task_pass,
            "success_rate": task_pass / task_total if task_total else None,
            "smoke_pass": sum(1 for row in task_rows if row.get("status") == "PASS"),
            "duration_seconds": stats(task_seconds),
        },
        "tools": {
            "llm_tool_calls": sum(int(row.get("llm_tool_calls") or 0) for row in task_rows),
            "tool_execs": tool_execs,
            "tool_successes": tool_successes,
            "tool_failures": sum(int(row.get("tool_failures") or 0) for row in task_rows),
            "tool_success_rate": tool_successes / tool_execs if tool_execs else None,
            "invalid_tool_markers": sum(int(row.get("invalid_tool_markers") or 0) for row in task_rows),
            "profiled_llm_calls": profiled_llm_calls,
            "profiled_llm_errors": profiled_llm_errors,
            "profiled_tool_execs": profiled_tool_execs,
            "profiled_tool_errors": profiled_tool_errors,
            "profiled_tool_success_rate": (
                (profiled_tool_execs - profiled_tool_errors) / profiled_tool_execs
                if profiled_tool_execs
                else None
            ),
            "profiled_task_count": len(event_rows),
            "llm_time_seconds": stats([as_float(str(row.get("llm_time_seconds"))) for row in event_rows]),
            "tool_time_seconds": stats([as_float(str(row.get("tool_time_seconds"))) for row in event_rows]),
            "other_time_seconds": stats([as_float(str(row.get("other_time_seconds"))) for row in event_rows]),
            "llm_time_pct": stats([as_float(str(row.get("llm_time_pct"))) for row in event_rows]),
            "tool_time_pct": stats([as_float(str(row.get("tool_time_pct"))) for row in event_rows]),
            "other_time_pct": stats([as_float(str(row.get("other_time_pct"))) for row in event_rows]),
            "input_tokens_profiled": sum(int(row.get("input_tokens_profiled") or 0) for row in task_rows),
            "output_tokens_profiled": sum(int(row.get("output_tokens_profiled") or 0) for row in task_rows),
            "total_tokens_profiled": sum(int(row.get("total_tokens_profiled") or 0) for row in task_rows),
        },
        "vllm_log": vllm_log_summary,
        "vllm_metrics": summarize_prometheus(run_root, wall_seconds),
        "hardware": {
            "vllm_gpu": summarize_resource_csv(
                run_root / "gpu_vllm.csv",
                ["gpu_util_pct", "mem_used_mib", "mem_total_mib", "power_w"],
            ),
            "worker_gpu": summarize_resource_csv(
                run_root / "gpu_worker.csv",
                ["gpu_util_pct", "mem_used_mib", "mem_total_mib", "power_w"],
            ),
            "vllm_cpu": summarize_resource_csv(
                run_root / "cpu_vllm.csv",
                ["cpu_util_pct", "cpu_freq_mhz", "cpu_power_w"],
            ),
            "worker_cpu": summarize_resource_csv(
                run_root / "cpu_worker.csv",
                ["cpu_util_pct", "cpu_freq_mhz", "cpu_power_w"],
            ),
        },
    }

    (run_root / "profile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# ChemGraph Concurrency Profile",
        "",
        f"- Run root: `{run_root}`",
        f"- Wall time: {fmt_num(wall_seconds)} s",
        f"- Task success: {task_pass}/{task_total} ({fmt_num(summary['tasks']['success_rate'], 4)})",
        f"- Smoke-level PASS before profile validation: {summary['tasks']['smoke_pass']}/{task_total}",
        f"- Tool success: {tool_successes}/{tool_execs} ({fmt_num(summary['tools']['tool_success_rate'], 4)})",
        f"- Tasks with internal event logs: {summary['tools']['profiled_task_count']}/{task_total}",
        f"- Profiled LLM calls/errors: {profiled_llm_calls}/{profiled_llm_errors}",
        f"- Profiled tool calls/errors: {profiled_tool_execs}/{profiled_tool_errors}",
        f"- Invalid tool markers: {summary['tools']['invalid_tool_markers']}",
        "",
        "## Latency",
        "",
        f"- Task duration P50/P95/P99: {fmt_num(summary['tasks']['duration_seconds']['p50'])} / {fmt_num(summary['tasks']['duration_seconds']['p95'])} / {fmt_num(summary['tasks']['duration_seconds']['p99'])} s",
        f"- vLLM TTFT P50/P95/P99: {fmt_num(summary['vllm_metrics']['histograms']['vllm:time_to_first_token_seconds']['p50'])} / {fmt_num(summary['vllm_metrics']['histograms']['vllm:time_to_first_token_seconds']['p95'])} / {fmt_num(summary['vllm_metrics']['histograms']['vllm:time_to_first_token_seconds']['p99'])} s",
        f"- vLLM e2e request P50/P95/P99: {fmt_num(summary['vllm_metrics']['histograms']['vllm:e2e_request_latency_seconds']['p50'])} / {fmt_num(summary['vllm_metrics']['histograms']['vllm:e2e_request_latency_seconds']['p95'])} / {fmt_num(summary['vllm_metrics']['histograms']['vllm:e2e_request_latency_seconds']['p99'])} s",
        f"- vLLM prefill P50/P95/P99: {fmt_num(summary['vllm_metrics']['histograms']['vllm:request_prefill_time_seconds']['p50'])} / {fmt_num(summary['vllm_metrics']['histograms']['vllm:request_prefill_time_seconds']['p95'])} / {fmt_num(summary['vllm_metrics']['histograms']['vllm:request_prefill_time_seconds']['p99'])} s",
        f"- vLLM decode P50/P95/P99: {fmt_num(summary['vllm_metrics']['histograms']['vllm:request_decode_time_seconds']['p50'])} / {fmt_num(summary['vllm_metrics']['histograms']['vllm:request_decode_time_seconds']['p95'])} / {fmt_num(summary['vllm_metrics']['histograms']['vllm:request_decode_time_seconds']['p99'])} s",
        f"- Per-task LLM time P50/P95/P99: {fmt_num(summary['tools']['llm_time_seconds']['p50'])} / {fmt_num(summary['tools']['llm_time_seconds']['p95'])} / {fmt_num(summary['tools']['llm_time_seconds']['p99'])} s",
        f"- Per-task tool time P50/P95/P99: {fmt_num(summary['tools']['tool_time_seconds']['p50'])} / {fmt_num(summary['tools']['tool_time_seconds']['p95'])} / {fmt_num(summary['tools']['tool_time_seconds']['p99'])} s",
        f"- Per-task LLM time share P50/P95/P99: {fmt_num(summary['tools']['llm_time_pct']['p50'] * 100 if summary['tools']['llm_time_pct']['p50'] is not None else None)} / {fmt_num(summary['tools']['llm_time_pct']['p95'] * 100 if summary['tools']['llm_time_pct']['p95'] is not None else None)} / {fmt_num(summary['tools']['llm_time_pct']['p99'] * 100 if summary['tools']['llm_time_pct']['p99'] is not None else None)} %",
        f"- Per-task tool time share P50/P95/P99: {fmt_num(summary['tools']['tool_time_pct']['p50'] * 100 if summary['tools']['tool_time_pct']['p50'] is not None else None)} / {fmt_num(summary['tools']['tool_time_pct']['p95'] * 100 if summary['tools']['tool_time_pct']['p95'] is not None else None)} / {fmt_num(summary['tools']['tool_time_pct']['p99'] * 100 if summary['tools']['tool_time_pct']['p99'] is not None else None)} %",
        "",
        "## Throughput",
        "",
        f"- Counter avg prompt throughput: {fmt_num(summary['vllm_metrics']['avg_prompt_toks_per_s_from_counters'])} toks/s",
        f"- Counter avg generation throughput: {fmt_num(summary['vllm_metrics']['avg_generation_toks_per_s_from_counters'])} toks/s",
        f"- Log prompt throughput P50/P95/P99: {fmt_num(summary['vllm_log']['prompt_toks_per_s']['p50'])} / {fmt_num(summary['vllm_log']['prompt_toks_per_s']['p95'])} / {fmt_num(summary['vllm_log']['prompt_toks_per_s']['p99'])} toks/s",
        f"- Log generation throughput P50/P95/P99: {fmt_num(summary['vllm_log']['generation_toks_per_s']['p50'])} / {fmt_num(summary['vllm_log']['generation_toks_per_s']['p95'])} / {fmt_num(summary['vllm_log']['generation_toks_per_s']['p99'])} toks/s",
        "",
        "## Hardware",
        "",
        f"- vLLM GPU util mean/max: {fmt_num(summary['hardware']['vllm_gpu']['gpu_util_pct']['mean'])} / {fmt_num(summary['hardware']['vllm_gpu']['gpu_util_pct']['max'])} %",
        f"- vLLM GPU memory max: {fmt_num(summary['hardware']['vllm_gpu']['mem_used_mib']['max'])} MiB",
        f"- vLLM GPU power mean/max: {fmt_num(summary['hardware']['vllm_gpu']['power_w']['mean'])} / {fmt_num(summary['hardware']['vllm_gpu']['power_w']['max'])} W",
        f"- worker CPU util mean/max: {fmt_num(summary['hardware']['worker_cpu']['cpu_util_pct']['mean'])} / {fmt_num(summary['hardware']['worker_cpu']['cpu_util_pct']['max'])} %",
        f"- vLLM CPU util mean/max: {fmt_num(summary['hardware']['vllm_cpu']['cpu_util_pct']['mean'])} / {fmt_num(summary['hardware']['vllm_cpu']['cpu_util_pct']['max'])} %",
        "",
        "## Notes",
        "",
        "- Task-level LLM/tool counts come from ChemGraph CLI logs.",
        "- Per-task LLM/tool time shares come from ChemGraph JSONL instrumentation when event logs are present.",
    ]
    (run_root / "profile_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(run_root / "profile_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
