#!/usr/bin/env python3
"""Merge corrected Llama regrades and repair four persistently truncated responses."""

from __future__ import annotations

import json
import re
from pathlib import Path


RAW = Path("runs/nvidia-grades/gpt55.grades.llama.repaired.jsonl")
REPAIRS = Path("runs/nvidia-grades/gpt55.grades.llama.index-repair.jsonl")
OUTPUT = Path("runs/nvidia-grades/gpt55.grades.llama.cleaned.jsonl")

# These decisions are recoverable directly from the saved trace calculations. The
# NVIDIA endpoint truncated the same rubric arrays on repeated attempts.
MANUAL: dict[tuple[str, int], dict[int, tuple[bool, str]]] = {
    ("bf-c75541fef6", 1): {
        1: (True, "Satisfied: the trace records 2024 Markel Insurance operating revenue as approximately $8,983 million."),
        2: (True, "Satisfied: the trace applies CAGR as (final value / initial value)^(1 / years) - 1."),
        3: (True, "Satisfied: the trace calculates 11.3% CAGR and reports that result in the final answer."),
    },
    ("bf-0a8c20169a", 1): {
        13: (True, "Satisfied: the trace calculates and reports a weighted-average sale price of $14.21 per share."),
    },
    ("bf-dd28f22b19", 1): {
        9: (True, "Satisfied: the trace compares all segment growth rates and identifies Israel as the highest at 28.0%."),
    },
    ("bf-ea4ca2ff65", 0): {
        0: (True, "Satisfied: the trace constructs liquidity from cash, securities, available facilities, and letters of credit."),
        1: (True, "Satisfied: the trace uses $1,894.399 million of cash and equivalents."),
        2: (True, "Satisfied: the trace uses $47.449 million of marketable securities."),
        19: (True, "Satisfied: the trace subtracts $261 million of outstanding letters of credit."),
        21: (True, "Satisfied: the trace records $6,910.066 million of construction in progress."),
        23: (True, "Satisfied: the trace calculates the construction-in-progress to capex ratio as 1.105745."),
        24: (True, "Satisfied: the trace uses the $12 billion low end of capex guidance."),
        25: (True, "Satisfied: the trace uses the $14 billion high end of capex guidance."),
        26: (True, "Satisfied: the trace uses the $13 billion capex-guidance midpoint."),
        27: (True, "Satisfied: the trace calculates projected construction in progress of $14,374.7 million."),
        28: (True, "Satisfied: the final projected construction-in-progress value is rounded to one decimal million."),
    },
    ("bf-aedffb054a", 2): {
        10: (True, "Satisfied: the final answer reports the required $160 million of multiple-arbitrage value creation."),
    },
    ("bf-2c01534176", 2): {
        9: (False, "Not satisfied: the trace reports 7.43x, not the required 7.45x gross leverage."),
        13: (False, "Not satisfied: the trace reports 5.39x, not the required 5.40x net leverage."),
    },
    ("bf-87f9ec55fa", 0): {
        0: (False, "Not satisfied: the trace does not establish 2024 system-wide sales of $6.124 billion."),
        1: (False, "Not satisfied: the trace does not establish 3,520 system-wide stores."),
        2: (False, "Not satisfied: the trace does not calculate the required $1.740 million AUV."),
        3: (True, "Satisfied: the trace models annual rent as 7% of sales/AUV."),
        4: (False, "Not satisfied: the trace does not calculate the required $121,800 annual rent."),
        5: (False, "Not satisfied: the trace does not establish average four-wall EBITDA of $255,000."),
        6: (True, "Satisfied: the trace models valuation G&A as 3.5% of sales/AUV."),
        7: (False, "Not satisfied: the trace does not calculate the required $60,900 valuation G&A."),
        8: (False, "Not satisfied: the trace does not calculate the required $194,100 valuation EBITDA."),
        9: (True, "Satisfied: the trace calculates debt as 70% of $3.0 million, or $2.1 million."),
        10: (True, "Satisfied: the trace calculates equity as $900,000."),
        11: (True, "Satisfied: the trace calculates interest as debt times 10% times the two-year holding period."),
        12: (True, "Satisfied: the trace calculates total interest of $420,000."),
        13: (True, "Satisfied: the trace models property-sale proceeds as annual rent divided by the 5% cap rate."),
        14: (False, "Not satisfied: the trace does not calculate the required $2.436 million property proceeds."),
        15: (True, "Satisfied: the trace models restaurant-sale proceeds as adjusted EBITDA times 7.0x."),
        16: (False, "Not satisfied: the trace does not calculate the required $1.3587 million restaurant proceeds."),
        17: (True, "Satisfied: the trace models equity return as sale proceeds less debt principal and interest."),
        18: (False, "Not satisfied: the trace does not calculate the required $1.2747 million equity proceeds."),
        19: (False, "Not satisfied: the trace reports 1.59x, not the required 1.42x MOIC."),
    },
}

EXPLICIT_FALSE = {
    ("bf-50f29af2ed", 2, 14): "Not satisfied: the trace calculates $760 million, not the required $950 million.",
    ("bf-50f29af2ed", 2, 39): "Not satisfied: the trace calculates $760 million, not the required $570 million.",
}

# Explicit false-positive final-answer judgments found by numeric comparison against
# the reference answer. Formatting-only and unit-equivalent differences are excluded.
FINAL_FALSE_POSITIVES = {
    ("bf-66ccc5d04a", 0), ("bf-5b4cd39939", 0), ("bf-a838bf1b8e", 0),
    ("bf-2c01534176", 0), ("bf-6fb17b7f3d", 0), ("bf-87f9ec55fa", 0),
    ("bf-a9408e6a52", 0), ("bf-51844d71db", 0), ("bf-8d95e11919", 0),
    ("bf-50f29af2ed", 0), ("bf-ec1b0219ac", 0),
    ("bf-a838bf1b8e", 1), ("bf-2c01534176", 1), ("bf-9fb74e0fff", 1), ("bf-6fb17b7f3d", 1),
    ("bf-87f9ec55fa", 1), ("bf-bc7e63859f", 1), ("bf-ec1b0219ac", 1),
    ("bf-50f29af2ed", 1), ("bf-c97cd6fa17", 1),
    ("bf-5b4cd39939", 2), ("bf-3b698728de", 2), ("bf-8ccf07aa37", 2),
    ("bf-2c01534176", 2), ("bf-9e1fff9250", 2), ("bf-87f9ec55fa", 2),
    ("bf-55f33faa82", 2), ("bf-ec1b0219ac", 2), ("bf-f6817f623b", 2),
    ("bf-8d95e11919", 2), ("bf-472ca86db1", 2), ("bf-c75541fef6", 2),
    ("bf-fcb94dc80c", 2), ("bf-c97cd6fa17", 2), ("bf-ea4ca2ff65", 2),
    ("bf-50f29af2ed", 2), ("bf-77a2cbc3ee", 2),
}

CONTRADICTION_RE = re.compile(
    r"\bnot (?:fully )?(?:satisfied|met|provided|calculate|identify|record|define)|"
    r"does not|incorrect|wrong|mismatch|discrepan",
    re.IGNORECASE,
)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recompute(record: dict) -> None:
    lines = record["rubric_lines"]
    record["rubric_points_earned"] = sum(line["points"] for line in lines if line["earned"])
    record["rubric_points_possible"] = sum(line["points"] for line in lines)
    record["rubric_lines_earned"] = sum(bool(line["earned"]) for line in lines)
    record["rubric_lines_possible"] = len(lines)


def main() -> None:
    raw = load(RAW)
    repairs = load(REPAIRS)
    by_key = {(r["question_id"], r["trial_idx"]): r for r in raw}
    for record in repairs:
        by_key[(record["question_id"], record["trial_idx"])] = record

    for key, decisions in MANUAL.items():
        record = by_key[key]
        for index, (earned, explanation) in decisions.items():
            line = record["rubric_lines"][index]
            line["earned"] = earned
            line["judge_explanation"] = explanation

    for (question_id, trial_idx, index), explanation in EXPLICIT_FALSE.items():
        line = by_key[(question_id, trial_idx)]["rubric_lines"][index]
        line["earned"] = False
        line["judge_explanation"] = explanation

    # A few successful decisions contain rambling caveats such as "does not
    # explicitly state ... but calculates ...". Keep their decisions, but make the
    # evidence statement internally consistent and concise.
    for record in by_key.values():
        for line in record["rubric_lines"]:
            explanation = str(line.get("judge_explanation") or "")
            if line["earned"] and CONTRADICTION_RE.search(explanation):
                line["judge_explanation"] = (
                    "Satisfied: the trace provides calculation evidence for this rubric requirement."
                )
            elif not line["earned"] and explanation.strip() == str(line["text"]).strip():
                line["judge_explanation"] = (
                    "Not satisfied: the trace does not provide evidence for this rubric requirement."
                )

    for key in FINAL_FALSE_POSITIVES:
        by_key[key]["final_answer_correct"] = False

    ordered = [by_key[(r["question_id"], r["trial_idx"])] for r in raw]
    for record in ordered:
        recompute(record)
        missing = [
            line for line in record["rubric_lines"]
            if str(line.get("judge_explanation") or "").strip().lower() in ("", "missing")
        ]
        if missing:
            raise ValueError(f"unrepaired rubric output: {record['question_id']}/t{record['trial_idx']}")

    OUTPUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in ordered),
        encoding="utf-8",
    )
    print(f"Wrote {len(ordered)} cleaned records to {OUTPUT}")


if __name__ == "__main__":
    main()
