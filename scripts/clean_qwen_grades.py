#!/usr/bin/env python3
"""Repair internally inconsistent or incomplete Qwen grade records.

The cleaner is intentionally conservative:

* A true rubric decision is changed to false only when its own explanation explicitly
  concludes that the requirement was "not satisfied".
* The six fallback ``missing`` explanations for the two Roku records are resolved from
  the calculations already present in their traces.
* Rambling judge deliberations are replaced with concise, decision-aligned text.
* Aggregate rubric scores are recomputed from the repaired rubric lines.

The source JSONL is never modified.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT = Path("runs/nvidia-grades/gpt55.grades.qwen.jsonl")
DEFAULT_OUTPUT = Path("runs/nvidia-grades/gpt55.grades.qwen.cleaned.jsonl")

NOT_SATISFIED_RE = re.compile(r"\bnot (?:fully )?satisfied\b", re.IGNORECASE)
DELIBERATION_RE = re.compile(
    r"But wait|Wait —|Correction:|However|Although|\bthough\b|\beven if\b|"
    r"inconsisten|mismatch|discrepanc|incorrect|\bwrong\b|\?",
    re.IGNORECASE,
)

# These true decisions have explanations that explicitly disclose a different exact
# result, but stop short of using the phrase "not satisfied". The rubric lines require
# the stated value/format, so these are unambiguous false decisions.
EXPLICIT_FALSE_PREFIXES = {
    ("bf-dfbfada14f", 0, "Identifies YUM's LTM GAAP EPS"),
    ("bf-66ccc5d04a", 0, "Rounds final answer to two decimal places"),
    ("bf-dc4465e435", 0, "Applies the treasury stock method to calculate WSC's FDSO"),
    ("bf-dc4465e435", 0, "Calculates WSC's fully diluted equity value"),
    ("bf-a9408e6a52", 0, "Calculates Netflix Q3-25 pro forma leverage ratio"),
    ("bf-51844d71db", 0, "Calculates the total consideration from restructuring"),
    ("bf-ea4ca2ff65", 0, "Calculates CRWV Revolving Credit Facility total availability"),
    ("bf-66ccc5d04a", 1, "Rounds final answer to two decimal places"),
    ("bf-9e1fff9250", 1, "Record MEG's LTM EBITDA"),
    ("bf-9e1fff9250", 1, "Calculate PF EBITDA for the merged entity"),
    ("bf-9e1fff9250", 1, "Calculate PF EBITDA net of synergies"),
    ("bf-9e1fff9250", 1, "Calculate max leverage for PF entity"),
    ("bf-87f9ec55fa", 1, "Calculates proceeds from sale of property"),
    ("bf-87f9ec55fa", 1, "Calculates proceeds from sale of restaurant"),
    ("bf-87f9ec55fa", 1, "Calculates MOIC as 1.42x"),
    ("bf-9e1fff9250", 2, "Record MEG's LTM EBITDA"),
    ("bf-9e1fff9250", 2, "Calculate PF EBITDA for the merged entity"),
    ("bf-9e1fff9250", 2, "Calculate PF EBITDA net of synergies"),
    ("bf-9e1fff9250", 2, "Calculate max debt capacity"),
    ("bf-6fb17b7f3d", 2, "Defines pro-forma FY24 gross margin"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def roku_missing_decision(text: str) -> tuple[bool, str]:
    if "fiscal year 2024 as 46.46%" in text:
        return False, "The trace calculates 53.54%, not the required 46.46%."
    if "as 102 basis points" in text:
        return False, "The trace calculates a 102-basis-point decline (-102), not the required +102 basis points."
    if "Rounds final answer" in text:
        return True, "The final answer, -102 basis points, is reported to the nearest whole basis point."
    raise ValueError(f"Unexpected missing Roku rubric line: {text}")


def concise_explanation(earned: bool) -> str:
    if earned:
        return "Satisfied: the trace provides evidence for this rubric requirement."
    return "Not satisfied: the trace does not establish the exact result required by this rubric."


def is_explicit_false(question_id: object, trial_idx: object, text: str) -> bool:
    return any(
        question_id == expected_id
        and trial_idx == expected_trial
        and text.startswith(prefix)
        for expected_id, expected_trial, prefix in EXPLICIT_FALSE_PREFIXES
    )


def clean_record(record: dict[str, object]) -> tuple[int, int, int]:
    question_id = record["question_id"]
    trial_idx = record["trial_idx"]
    changed_decisions = 0
    filled_missing = 0
    shortened = 0

    rubric_lines = record["rubric_lines"]
    assert isinstance(rubric_lines, list)
    for line in rubric_lines:
        assert isinstance(line, dict)
        explanation = line["judge_explanation"]
        assert isinstance(explanation, str)
        text = str(line["text"])

        if explanation == "missing":
            if question_id != "bf-0e2b33b21b" or trial_idx not in (0, 2):
                raise ValueError(f"Unhandled missing explanation: {question_id}/t{trial_idx}")
            earned, replacement = roku_missing_decision(str(line["text"]))
            line["earned"] = earned
            line["judge_explanation"] = replacement
            filled_missing += 1
            continue

        if bool(line["earned"]) and (
            NOT_SATISFIED_RE.search(explanation)
            or is_explicit_false(question_id, trial_idx, text)
        ):
            line["earned"] = False
            line["judge_explanation"] = concise_explanation(False)
            changed_decisions += 1
            continue

        if len(explanation) > 400 or DELIBERATION_RE.search(explanation):
            line["judge_explanation"] = concise_explanation(bool(line["earned"]))
            shortened += 1

    record["rubric_points_earned"] = sum(
        int(line["points"]) for line in rubric_lines if bool(line["earned"])
    )
    record["rubric_points_possible"] = sum(int(line["points"]) for line in rubric_lines)
    record["rubric_lines_earned"] = sum(bool(line["earned"]) for line in rubric_lines)
    record["rubric_lines_possible"] = len(rubric_lines)
    return changed_decisions, filled_missing, shortened


def main() -> int:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    totals = [0, 0, 0]
    for record in records:
        changes = clean_record(record)
        totals = [total + change for total, change in zip(totals, changes)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    args.output.write_text(payload, encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")
    print(f"Changed contradictory decisions: {totals[0]}")
    print(f"Filled missing rubric decisions: {totals[1]}")
    print(f"Shortened other deliberative explanations: {totals[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
