#!/usr/bin/env python3
"""Repair incomplete or contradictory Nemotron rubric outputs."""

from __future__ import annotations

import json
from pathlib import Path


RAW = Path("runs/nvidia-grades/gpt55.grades.nemotron.jsonl")
FALLBACK = Path("runs/nvidia-grades/gpt55.grades.llama.cleaned.jsonl")
OUTPUT = Path("runs/nvidia-grades/gpt55.grades.nemotron.cleaned.jsonl")

# These explanations explicitly state that the trace result differs from the
# rubric requirement, despite the provider returning earned=true.
EXPLICIT_FALSE = {
    ("bf-a9408e6a52", 2, 14),
    ("bf-8ccf07aa37", 1, 6),
    ("bf-472ca86db1", 2, 36),
}


def recompute(record: dict) -> None:
    lines = record["rubric_lines"]
    record["rubric_points_earned"] = sum(
        line["points"] for line in lines if line["earned"]
    )
    record["rubric_points_possible"] = sum(line["points"] for line in lines)
    record["rubric_lines_earned"] = sum(bool(line["earned"]) for line in lines)
    record["rubric_lines_possible"] = len(lines)


def main() -> None:
    records = [
        json.loads(line) for line in RAW.read_text().splitlines() if line.strip()
    ]
    fallback = {
        (record["question_id"], record["trial_idx"]): record
        for record in (
            json.loads(line)
            for line in FALLBACK.read_text().splitlines()
            if line.strip()
        )
    }
    if len(records) != 150:
        raise ValueError(f"expected 150 raw records, found {len(records)}")
    keys = [(r["question_id"], r["trial_idx"]) for r in records]
    if len(set(keys)) != 150:
        raise ValueError("raw Nemotron records are not unique")

    for record in records:
        key = (record["question_id"], record["trial_idx"])
        for index, line in enumerate(record["rubric_lines"]):
            if (*key, index) in EXPLICIT_FALSE:
                line["earned"] = False
            explanation = str(line.get("judge_explanation") or "").strip()
            if not explanation or explanation.lower() == "missing":
                replacement = fallback[key]["rubric_lines"][index]
                if replacement["text"] != line["text"]:
                    raise ValueError(f"rubric mismatch for {key} line {index}")
                line["earned"] = replacement["earned"]
                line["judge_explanation"] = replacement["judge_explanation"]
        recompute(record)

    OUTPUT.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
            for r in records
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} cleaned records to {OUTPUT}")


if __name__ == "__main__":
    main()
