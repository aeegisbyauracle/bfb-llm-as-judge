#!/usr/bin/env python3
"""Deduplicate grade JSONL records without modifying the source file."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of",
    "on", "or", "the", "to", "was", "with", "agent", "rubric", "trace",
}


def words(text: str) -> set[str]:
    return {word for word in WORD_RE.findall(text.lower()) if word not in STOPWORDS}


def quality(record: dict[str, object]) -> tuple[int, int, float, int]:
    """Rank duplicates by completeness, consistency, and explanation alignment."""
    lines = record["rubric_lines"]
    assert isinstance(lines, list)
    missing = 0
    conflicts = 0
    alignment = 0.0
    for line in lines:
        assert isinstance(line, dict)
        explanation = str(line["judge_explanation"])
        if explanation == "missing" or not explanation.strip():
            missing += 1
        lower = explanation.lower()
        if bool(line["earned"]) and "not satisfied" in lower:
            conflicts += 1
        rubric_words = words(str(line["text"]))
        if rubric_words:
            alignment += len(rubric_words & words(explanation)) / len(rubric_words)
    completion_tokens = int(record.get("judge_completion_tokens") or 0)
    return (-missing, -conflicts, alignment, completion_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[tuple[object, object, object], list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for position, record in enumerate(records):
        key = (record["question_id"], record["trial_idx"], record["judge"])
        groups[key].append((position, record))

    selected: list[tuple[int, dict[str, object]]] = []
    for variants in groups.values():
        best = max(variants, key=lambda item: quality(item[1]))
        selected.append(best)
    selected.sort(key=lambda item: item[0])

    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for _, record in selected
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"Input rows: {len(records)}")
    print(f"Unique rows written: {len(selected)}")
    print(f"Duplicate rows removed: {len(records) - len(selected)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
