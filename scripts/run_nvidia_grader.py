#!/usr/bin/env python3
"""Grade existing BigFinanceBench traces with one NVIDIA-hosted judge."""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from pathlib import Path

from big_finance_harness.grader import grade
from big_finance_harness.resumption import grade_completed_triples
from big_finance_harness.trace import read_traces
from big_finance_harness.types import DatasetItem


JUDGES = {
    "llama": "nvidia:meta/llama-3.3-70b-instruct",
    "qwen": "nvidia:qwen/qwen3-next-80b-a3b-instruct",
    "mistral": "nvidia:mistralai/mistral-medium-3.5-128b",
    "nemotron": "nvidia:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "tinker-gpt-oss-20b": "tinker:openai/gpt-oss-20b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply BFB's original grader prompt to released traces with NVIDIA NIM."
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
    parser.add_argument("--num-retries", type=int, default=20)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument(
        "--question-delay-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait after each attempted question before starting the next.",
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


async def run(args: argparse.Namespace) -> int:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.question_delay_seconds < 0:
        raise ValueError("--question-delay-seconds cannot be negative")
    if args.sample_n is not None and args.sample_n < 1:
        raise ValueError("--sample-n must be at least 1")
    if args.num_retries < 0:
        raise ValueError("--num-retries cannot be negative")
    if args.request_timeout < 1:
        raise ValueError("--request-timeout must be at least 1")

    judge_model_id = JUDGES[args.judge]
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
    if args.question_delay_seconds:
        print(f"Question delay: {args.question_delay_seconds:.1f}s")
        if args.concurrency > 1:
            print("Question delay enabled; processing sequentially despite --concurrency.")

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    counters = {"done": 0, "failed": 0}
    started = time.monotonic()

    async def grade_one(trace) -> None:
        item = items.get(trace.question_id)
        if item is None:
            counters["failed"] += 1
            print(f"Missing dataset item: {trace.question_id}")
            return

        async with semaphore:
            try:
                result = await grade(
                    run=trace,
                    item=item,
                    judge_model_id=judge_model_id,
                    max_output_tokens=args.max_output_tokens,
                    num_retries=args.num_retries,
                    request_timeout=args.request_timeout,
                )
            except Exception as exc:
                counters["failed"] += 1
                print(f"Failed {trace.question_id}/t{trace.trial_idx}: {exc}")
                return

        async with write_lock:
            with output.open("a", encoding="utf-8") as file:
                file.write(result.model_dump_json() + "\n")
            counters["done"] += 1
            print(f"Graded {counters['done']}/{len(work)}: {trace.question_id}")

    if args.question_delay_seconds:
        for idx, trace in enumerate(work, start=1):
            await grade_one(trace)
            if idx < len(work):
                print(f"Waiting {args.question_delay_seconds:.1f}s before next question...")
                await asyncio.sleep(args.question_delay_seconds)
    else:
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
