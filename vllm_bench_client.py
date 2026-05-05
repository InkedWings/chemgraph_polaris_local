#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request


def percentile(values, pct):
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def mean(values):
    return statistics.fmean(values) if values else None


def load_tokenizer(model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


class PromptFactory:
    def __init__(self, tokenizer, input_len, seed):
        self.tokenizer = tokenizer
        self.input_len = input_len
        self.seed = seed
        self.special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
        self.vocab_size = len(tokenizer)
        self.low_id = min(1000, max(0, self.vocab_size - 1))

    def token_ids(self, index):
        rng = random.Random(self.seed + index)
        ids = []
        while len(ids) < self.input_len:
            token_id = rng.randrange(self.low_id, self.vocab_size)
            if token_id not in self.special_ids:
                ids.append(token_id)
        return ids

    def text_from_ids(self, ids):
        text = self.tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        actual = len(self.tokenizer.encode(text, add_special_tokens=False))
        return text, actual


def request_once(args, tokenizer, prompts, index, force_text=False):
    token_ids = prompts.token_ids(index)
    prompt_mode = "string" if force_text or args.prompt_mode == "string" else "token_ids"
    actual_input_tokens = len(token_ids)
    prompt = token_ids

    if prompt_mode == "string":
        prompt, actual_input_tokens = prompts.text_from_ids(token_ids)

    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.output_len,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://{args.host}:{args.port}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_token = None
    chunks = []

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                choices = obj.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text") or ""
                if text and first_token is None:
                    first_token = time.perf_counter()
                chunks.append(text)
        end = time.perf_counter()
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        if args.prompt_mode == "auto" and prompt_mode == "token_ids":
            return request_once(args, tokenizer, prompts, index, force_text=True)
        return {
            "index": index,
            "ok": False,
            "prompt_mode": prompt_mode,
            "error": f"HTTP {exc.code}: {message[:500]}",
        }
    except Exception as exc:
        return {
            "index": index,
            "ok": False,
            "prompt_mode": prompt_mode,
            "error": repr(exc),
        }

    completion = "".join(chunks)
    output_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
    if first_token is None:
        first_token = end

    ttft = first_token - start
    decode_time = max(end - first_token, 0.0)
    tpot = decode_time / max(output_tokens - 1, 1)

    return {
        "index": index,
        "ok": True,
        "prompt_mode": prompt_mode,
        "input_tokens": actual_input_tokens,
        "output_tokens": output_tokens,
        "latency_s": end - start,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "start_s": start,
        "first_token_s": first_token,
        "end_s": end,
    }


def summarize(args, rows, started, ended):
    ok_rows = [row for row in rows if row.get("ok")]
    failed_rows = [row for row in rows if not row.get("ok")]
    duration = ended - started
    input_tokens = [row["input_tokens"] for row in ok_rows]
    output_tokens = [row["output_tokens"] for row in ok_rows]
    ttft_ms = [row["ttft_s"] * 1000.0 for row in ok_rows]
    tpot_ms = [row["tpot_s"] * 1000.0 for row in ok_rows]
    lat_ms = [row["latency_s"] * 1000.0 for row in ok_rows]
    total_input = sum(input_tokens)
    total_output = sum(output_tokens)

    return {
        "model": args.model,
        "tokenizer_model": args.tokenizer_model,
        "input_len_requested": args.input_len,
        "output_len_requested": args.output_len,
        "num_prompts": args.num_prompts,
        "max_concurrency": args.max_concurrency,
        "prompt_mode": args.prompt_mode,
        "duration_s": duration,
        "success_count": len(ok_rows),
        "failure_count": len(failed_rows),
        "request_throughput": len(ok_rows) / duration if duration > 0 else None,
        "client_input_toks_per_s": total_input / duration if duration > 0 else None,
        "client_output_toks_per_s": total_output / duration if duration > 0 else None,
        "client_total_toks_per_s": (total_input + total_output) / duration if duration > 0 else None,
        "actual_input_tokens_mean": mean(input_tokens),
        "actual_output_tokens_mean": mean(output_tokens),
        "mean_latency_ms": mean(lat_ms),
        "p50_latency_ms": percentile(lat_ms, 50),
        "p99_latency_ms": percentile(lat_ms, 99),
        "mean_ttft_ms": mean(ttft_ms),
        "p50_ttft_ms": percentile(ttft_ms, 50),
        "p99_ttft_ms": percentile(ttft_ms, 99),
        "mean_tpot_ms": mean(tpot_ms),
        "p50_tpot_ms": percentile(tpot_ms, 50),
        "p99_tpot_ms": percentile(tpot_ms, 99),
        "requests": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--prompt-mode", choices=("auto", "token_ids", "string"), default="auto")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()

    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
    os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])

    tokenizer = load_tokenizer(args.tokenizer_model)
    prompts = PromptFactory(tokenizer, args.input_len, args.seed)

    started = time.perf_counter()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        futures = [
            pool.submit(request_once, args, tokenizer, prompts, index)
            for index in range(args.num_prompts)
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    ended = time.perf_counter()

    rows.sort(key=lambda row: row["index"])
    summary = summarize(args, rows, started, ended)

    os.makedirs(os.path.dirname(args.result_json), exist_ok=True)
    with open(args.result_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    printable = {key: value for key, value in summary.items() if key != "requests"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 1 if summary["failure_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
