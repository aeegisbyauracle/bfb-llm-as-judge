"""Verify local judge grade files are complete before analysis.

Completeness criteria:
- one expected JSONL file exists for each judge and trace label
- each file has exactly `--expected-rows` non-empty rows, default 150
- each file has exactly `--expected-rows` unique `(question_id, trial_idx)` keys
- no duplicate keys, missing keys, or invalid JSON rows

Outputs are written to `analysis/results` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_EXPECTED_ROWS = 150
DEFAULT_JUDGE_FILES = {
    "tinker": "tinker-gpt-oss-20b.jsonl",
    "qwen": "nvidia-qwen.jsonl",
    "llama": "vllm-llama.jsonl",
    "nemotron": "nvidia-nemotron.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether candidate judge JSONLs satisfy completeness criteria."
    )
    parser.add_argument("--grade-results-dir", type=Path, default=Path("grade-results"))
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("data/raw/big-finance-benchmark/traces"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results"))
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=DEFAULT_EXPECTED_ROWS,
        help="Required non-empty JSONL row count and unique-key count for each file.",
    )
    return parser.parse_args()


def discover_traces(traces_dir: Path) -> list[str]:
    labels = sorted(path.stem.split(".traces", 1)[0] for path in traces_dir.glob("*.traces.jsonl"))
    if not labels:
        raise FileNotFoundError(f"No trace files found in {traces_dir}")
    return labels


def inspect_file(path: Path, expected_rows: int) -> dict:
    if not path.exists():
        return {
            "exists": False,
            "row_count": 0,
            "unique_key_count": 0,
            "duplicate_key_rows": 0,
            "missing_key_rows": 0,
            "invalid_json_rows": 0,
            "duplicate_keys": "",
            "is_complete": False,
            "issues": f"missing file: {path.name}",
        }

    row_count = 0
    invalid_json_rows = 0
    missing_key_rows = 0
    duplicate_key_rows = 0
    keys_seen = set()
    duplicate_keys = []

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_rows += 1
                continue

            question_id = row.get("question_id")
            trial_idx = row.get("trial_idx")
            if question_id is None or trial_idx is None:
                missing_key_rows += 1
                continue

            key = (str(question_id), int(trial_idx))
            if key in keys_seen:
                duplicate_key_rows += 1
                duplicate_keys.append(f"{key[0]}:{key[1]}")
            keys_seen.add(key)

    issues = []
    if row_count != expected_rows:
        issues.append(f"{row_count} rows, expected {expected_rows}")
    if len(keys_seen) != expected_rows:
        issues.append(f"{len(keys_seen)} unique keys, expected {expected_rows}")
    if duplicate_key_rows:
        issues.append(f"{duplicate_key_rows} duplicate key row(s)")
    if missing_key_rows:
        issues.append(f"{missing_key_rows} missing key row(s)")
    if invalid_json_rows:
        issues.append(f"{invalid_json_rows} invalid JSON row(s)")

    return {
        "exists": True,
        "row_count": row_count,
        "unique_key_count": len(keys_seen),
        "duplicate_key_rows": duplicate_key_rows,
        "missing_key_rows": missing_key_rows,
        "invalid_json_rows": invalid_json_rows,
        "duplicate_keys": ";".join(duplicate_keys),
        "is_complete": not issues,
        "issues": "; ".join(issues),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    trace_labels = discover_traces(args.traces_dir)

    per_file_rows = []
    for judge, file_name in DEFAULT_JUDGE_FILES.items():
        for trace_label in trace_labels:
            path = args.grade_results_dir / trace_label / file_name
            inspection = inspect_file(path, args.expected_rows)
            per_file_rows.append(
                {
                    "judge": judge,
                    "trace_label": trace_label,
                    "expected_file": str(path),
                    **inspection,
                }
            )

    rows_by_judge = defaultdict(list)
    for row in per_file_rows:
        rows_by_judge[row["judge"]].append(row)

    summary_rows = []
    for judge, rows in rows_by_judge.items():
        incomplete_rows = [row for row in rows if not row["is_complete"]]
        summary_rows.append(
            {
                "judge": judge,
                "expected_files": len(trace_labels),
                "complete_files": sum(1 for row in rows if row["is_complete"]),
                "incomplete_files": len(incomplete_rows),
                "is_complete": not incomplete_rows,
                "issues": " | ".join(
                    f"{row['trace_label']}: {row['issues']}" for row in incomplete_rows
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_file_path = args.out_dir / "judge_completeness_by_file.csv"
    summary_path = args.out_dir / "judge_completeness_summary.csv"
    manifest_path = args.out_dir / "judge_completeness_manifest.json"
    write_csv(per_file_path, per_file_rows)
    write_csv(summary_path, summary_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "trace_labels": trace_labels,
                "judge_files": DEFAULT_JUDGE_FILES,
                "expected_rows": args.expected_rows,
                "outputs": {
                    "by_file": str(per_file_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Judge completeness:")
    for row in summary_rows:
        status = "complete" if row["is_complete"] else "incomplete"
        print(
            f"  {row['judge']}: {status} "
            f"({row['complete_files']}/{row['expected_files']} files complete)"
        )
        if row["issues"]:
            print(f"    {row['issues']}")
    print(f"Wrote {per_file_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
