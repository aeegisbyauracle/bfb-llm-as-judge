#!/usr/bin/env python3
"""Fast NVIDIA NIM grading: all 4 judges in parallel."""

from __future__ import annotations

import argparse
import asyncio
import datetime
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
    "groq-qwen3-32b": "groq:qwen/qwen3-32b",
    "groq-llama-8b": "groq:llama-3.1-8b-instant",
    "groq-llama-70b": "groq:llama-3.3-70b-versatile",
    "cerebras-zai": "cerebras:zai-glm-4.7",
    "cerebras-gemma": "cerebras:gemma-4-31b",
    "tinker-nemotron-ultra": "tinker:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:peft:262144",
    "tinker-nemotron-super": "tinker:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144",
    "tinker-qwen397b": "tinker:Qwen/Qwen3.5-397B-A17B:peft:262144",
    "tinker-qwen35b": "tinker:Qwen/Qwen3.6-35B-A3B",
    "tinker-kimi": "tinker:moonshotai/Kimi-K2.6:peft:131072",
    "vllm-llama": "vllm:meta-llama/Llama-3.3-70B-Instruct",
}

CONCURRENCY_PER_JUDGE = 5
INTER_REQUEST_DELAY = 2.0

_log_file = None
_log_lock = asyncio.Lock()


def _init_log(output_dir: Path, traces_name: str) -> Path:
    global _log_file
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"{traces_name}.run.{ts}.log"
    _log_file = open(log_path, "w", encoding="utf-8")
    return log_path


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast NVIDIA grading with all 4 judges.")
    parser.add_argument("--traces", required=True, type=Path, help="Traces JSONL file.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/big_finance_subset.jsonl"),
        help="BFB dataset JSONL.",
    )
    parser.add_argument(
        "--judge",
        nargs="*",
        choices=list(JUDGES.keys()),
        help="Which judge(s) to run (e.g. --judge mistral nemotron). Defaults to all 4.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/nvidia-grades"))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY_PER_JUDGE)
    parser.add_argument("--delay", type=float, default=INTER_REQUEST_DELAY)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
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


def output_path_for(traces: Path, judge_alias: str, output_dir: Path) -> Path:
    label = traces.name.removesuffix(".traces.jsonl").removesuffix(".jsonl")
    return output_dir / f"{label}.grades.{judge_alias}.jsonl"


async def run_judge(
    judge_alias: str,
    judge_model_id: str,
    traces: list,
    items: dict[str, DatasetItem],
    output: Path,
    concurrency: int,
    delay: float,
    max_output_tokens: int,
    no_resume: bool,
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)

    if no_resume and output.exists():
        output.unlink()
    completed = grade_completed_triples(output)
    work = [
        t for t in traces
        if (t.question_id, t.trial_idx, judge_model_id) not in completed
    ]

    log(f"[{judge_alias}] {len(work)} remaining of {len(traces)} traces")
    if not work:
        return {"done": 0, "failed": 0}

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counters = {"done": 0, "failed": 0}

    async def grade_one(trace) -> None:
        async with semaphore:
            await asyncio.sleep(delay)
            item = items.get(trace.question_id)
            if item is None:
                counters["failed"] += 1
                log(f"[{judge_alias}] Missing dataset item: {trace.question_id}")
                return
            t0 = time.monotonic()
            try:
                result = await grade(
                    run=trace,
                    item=item,
                    judge_model_id=judge_model_id,
                    max_output_tokens=max_output_tokens,
                    num_retries=5,
                    request_timeout=120,
                )
            except Exception as exc:
                elapsed = time.monotonic() - t0
                counters["failed"] += 1
                log(f"[{judge_alias}] FAILED {trace.question_id}/t{trace.trial_idx} ({elapsed:.1f}s): {exc}")
                return

        elapsed = time.monotonic() - t0
        async with write_lock:
            with output.open("a", encoding="utf-8") as f:
                f.write(result.model_dump_json() + "\n")
            counters["done"] += 1
            total = counters["done"] + counters["failed"]
            log(f"[{judge_alias}] {counters['done']}/{len(work)} done ({total}/{len(work)}) [{elapsed:.1f}s]")

    await asyncio.gather(*(grade_one(t) for t in work))
    return counters


async def run(args: argparse.Namespace) -> int:
    judges_to_run = {k: JUDGES[k] for k in args.judge} if args.judge else JUDGES

    items = load_items(args.dataset)
    traces = list(read_traces(args.traces))
    traces_name = args.traces.name.removesuffix(".traces.jsonl").removesuffix(".jsonl")
    log_path = _init_log(args.output_dir, traces_name)
    log(f"Log: {log_path}")
    log(f"Loaded {len(traces)} traces, {len(items)} dataset items")
    log(f"Running {len(judges_to_run)} judge(s): {', '.join(judges_to_run)} | concurrency={args.concurrency}/judge | delay={args.delay}s")

    started = time.monotonic()

    results = await asyncio.gather(*(
        run_judge(
            judge_alias=alias,
            judge_model_id=model_id,
            traces=traces,
            items=items,
            output=output_path_for(args.traces, alias, args.output_dir),
            concurrency=args.concurrency,
            delay=args.delay,
            max_output_tokens=args.max_output_tokens,
            no_resume=args.no_resume,
        )
        for alias, model_id in judges_to_run.items()
    ))

    elapsed = time.monotonic() - started
    log(f"Finished in {elapsed:.1f}s")
    for (alias, _), counters in zip(judges_to_run.items(), results):
        log(f"  {alias}: {counters['done']} graded, {counters['failed']} failed")
    if _log_file:
        _log_file.close()

    return 1 if any(c["failed"] for c in results) else 0


def main() -> int:
    args = parse_args()
    if not args.traces.exists():
        raise FileNotFoundError(args.traces)
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
