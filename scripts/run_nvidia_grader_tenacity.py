#!/usr/bin/env python3
"""Grade existing BigFinanceBench traces with Tenacity-managed retries.

This is a standalone variant of `scripts/run_nvidia_grader.py`. It intentionally does
not modify the existing runner or grader implementation. The difference is that each
question is wrapped in an outer Tenacity retry loop, while LiteLLM's internal retry count
defaults to zero unless explicitly set by the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from pathlib import Path

from tenacity import AsyncRetrying, RetryCallState, stop_after_attempt, wait_exponential_jitter

from big_finance_harness import grader as grader_module
from big_finance_harness.resumption import grade_completed_triples
from big_finance_harness.trace import read_traces
from big_finance_harness.types import DatasetItem


JUDGES = {
    "llama": "nvidia:meta/llama-3.3-70b-instruct",
    "qwen": "nvidia:qwen/qwen3-next-80b-a3b-instruct",
    "mistral": "nvidia:mistralai/mistral-medium-3.5-128b",
    "nemotron": "nvidia:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply BFB's grader prompt to released traces with Tenacity retries."
    )
    parser.add_argument("--judge", required=True, choices=JUDGES)
    parser.add_argument("--traces", required=True, type=Path, help="BFB traces JSONL file.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/big_finance_subset.jsonl"),
        help="BFB dataset JSONL containing reference answers and rubrics.",
    )
    parser.add_argument("--output", type=Path, help="Destination grades JSONL file.")
    parser.add_argument("--sample-n", type=int, help="Grade a reproducible trace sample.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-initial-seconds", type=float, default=10.0)
    parser.add_argument("--retry-max-seconds", type=float, default=120.0)
    parser.add_argument(
        "--provider-cap",
        type=int,
        help=(
            "Override the grader module's in-process provider semaphore cap. "
            "Defaults to --concurrency for NVIDIA judges."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_items(path: Path) -> dict[str, DatasetItem]:
    return {
        item.id: item
        for item in (
            DatasetItem.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def default_output_path(traces: Path, judge_alias: str) -> Path:
    label = traces.name.removesuffix(".traces.jsonl").removesuffix(".jsonl")
    return Path("runs/nvidia-grades") / f"{label}.grades.{judge_alias}.jsonl"


def _log_before_sleep(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome is not None else None
    sleep_for = state.next_action.sleep if state.next_action is not None else 0
    print(
        "Retrying judge call "
        f"(attempt {state.attempt_number} failed; sleeping {sleep_for:.1f}s): {exc}",
        flush=True,
    )


async def run(args: argparse.Namespace) -> int:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.sample_n is not None and args.sample_n < 1:
        raise ValueError("--sample-n must be at least 1")
    if args.retry_attempts < 1:
        raise ValueError("--retry-attempts must be at least 1")
    if args.retry_initial_seconds < 0 or args.retry_max_seconds < 0:
        raise ValueError("retry wait values cannot be negative")

    # Avoid nested retry storms by default. Callers can still override this env var.
    os.environ.setdefault("BFB_JUDGE_NUM_RETRIES", "0")

    judge_model_id = JUDGES[args.judge]
    provider = judge_model_id.split(":", 1)[0]
    provider_cap = args.provider_cap
    if provider_cap is None and provider == "nvidia":
        provider_cap = args.concurrency
    if provider_cap is not None:
        if provider_cap < 1:
            raise ValueError("--provider-cap must be at least 1")
        grader_module._JUDGE_CAPS[provider] = provider_cap
        grader_module._JUDGE_SEMAPHORES.pop(judge_model_id, None)

    output = args.output or default_output_path(args.traces, args.judge)
    output.parent.mkdir(parents=True, exist_ok=True)

    items = load_items(args.dataset)
    traces = list(read_traces(args.traces))
    if args.sample_n is not None:
        rng = random.Random(args.sample_seed)
        traces = rng.sample(traces, min(args.sample_n, len(traces)))

    if args.no_resume and output.exists():
        output.unlink()
    completed = grade_completed_triples(output)
    work = [
        trace
        for trace in traces
        if (trace.question_id, trace.trial_idx, judge_model_id) not in completed
    ]

    print(f"Judge: {judge_model_id}")
    print(f"Traces: {len(traces)} ({len(work)} remaining)")
    print(f"Output: {output}")
    print(
        "Tenacity: "
        f"attempts={args.retry_attempts}, "
        f"wait=exponential_jitter({args.retry_initial_seconds:.1f}s.."
        f"{args.retry_max_seconds:.1f}s)"
    )
    if provider_cap is not None:
        print(f"Provider cap override: {provider}={provider_cap}")

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    counters = {"done": 0, "failed": 0}
    started = time.monotonic()

    async def grade_with_retry(trace, item: DatasetItem):
        retrying = AsyncRetrying(
            stop=stop_after_attempt(args.retry_attempts),
            wait=wait_exponential_jitter(
                initial=args.retry_initial_seconds,
                max=args.retry_max_seconds,
            ),
            before_sleep=_log_before_sleep,
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await grader_module.grade(
                    run=trace,
                    item=item,
                    judge_model_id=judge_model_id,
                    max_output_tokens=args.max_output_tokens,
                )
        raise RuntimeError("unreachable retry state")

    async def grade_one(trace) -> None:
        item = items.get(trace.question_id)
        if item is None:
            counters["failed"] += 1
            print(f"Missing dataset item: {trace.question_id}", flush=True)
            return

        async with semaphore:
            try:
                result = await grade_with_retry(trace, item)
            except Exception as exc:
                counters["failed"] += 1
                print(f"Failed {trace.question_id}/t{trace.trial_idx}: {exc}", flush=True)
                return

        async with write_lock:
            with output.open("a", encoding="utf-8") as file:
                file.write(result.model_dump_json() + "\n")
            counters["done"] += 1
            print(f"Graded {counters['done']}/{len(work)}: {trace.question_id}", flush=True)

    await asyncio.gather(*(grade_one(trace) for trace in work))
    elapsed = time.monotonic() - started
    print(f"Finished in {elapsed:.1f}s; {counters['done']} graded, {counters['failed']} failed.")
    return 1 if counters["failed"] else 0


def main() -> int:
    args = parse_args()
    if not args.traces.exists():
        raise FileNotFoundError(args.traces)
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
