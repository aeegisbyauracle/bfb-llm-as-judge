"""Compare how local LLM judges score each trace model.

Outputs are written to ``analysis/results`` by default:
- ``judge_scores_by_trace.csv``: one row per trace label and judge
- ``judge_score_wide_by_trace.csv``: one row per trace label with judge scores side by side
- ``judge_pairwise_deltas_by_trace.csv``: pairwise score deltas for each trace label
- ``judge_scores_by_item.csv``: one row per trace/question/trial with judge scores side by side
- ``plots/judge_scores_by_trace_heatmap.svg``: average normalized score heatmap
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean


DEFAULT_JUDGE_FILES = {
    "tinker": "tinker-gpt-oss-20b.jsonl",
    "qwen": "nvidia-qwen.jsonl",
    "llama": "vllm-llama.jsonl",
    "nemotron": "nvidia-nemotron.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare score differences between local LLM judges for each trace."
    )
    parser.add_argument("--grade-results-dir", type=Path, default=Path("grade-results"))
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("data/raw/big-finance-benchmark/traces"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results"))
    parser.add_argument(
        "--judges",
        nargs="+",
        choices=sorted(DEFAULT_JUDGE_FILES),
        default=list(DEFAULT_JUDGE_FILES),
        help="Judge aliases to include.",
    )
    return parser.parse_args()


def discover_traces(traces_dir: Path) -> list[str]:
    labels = sorted(path.stem.split(".traces", 1)[0] for path in traces_dir.glob("*.traces.jsonl"))
    if not labels:
        raise FileNotFoundError(f"No traces found in {traces_dir}")
    return labels


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def as_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def as_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def normalized_score(row: dict) -> float | None:
    earned = as_float(row.get("rubric_points_earned"))
    possible = as_float(row.get("rubric_points_possible"))
    if earned is None or not possible:
        return None
    return earned / possible


def row_key(row: dict) -> tuple[str, int] | None:
    question_id = row.get("question_id")
    trial_idx = row.get("trial_idx")
    if question_id is None or trial_idx is None:
        return None
    return (str(question_id), int(trial_idx))


def load_judge_rows(path: Path) -> dict[tuple[str, int], dict]:
    loaded = {}
    for row in read_jsonl(path):
        key = row_key(row)
        if key is not None:
            loaded[key] = row
    return loaded


def fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    return "" if value is None else f"{100 * value:.{digits}f}%"


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


def parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_score_heatmap(path: Path, wide_rows: list[dict], judges: list[str]) -> None:
    trace_labels = [row["trace_label"] for row in wide_rows]
    values = []
    for row in wide_rows:
        values.append(
            [
                parse_float(row.get(f"{judge}_avg_normalized_score")) or math.nan
                for judge in judges
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    left = 118
    top = 76
    cell_w = 104
    cell_h = 32
    legend_w = 170
    width = left + cell_w * len(judges) + legend_w
    height = top + cell_h * len(trace_labels) + 58

    def color_for(value: float) -> str:
        if math.isnan(value):
            return "#f2f2f2"
        value = max(0.0, min(1.0, value))
        start = (247, 252, 245)
        mid = (116, 196, 118)
        end = (0, 90, 50)
        if value < 0.5:
            t = value / 0.5
            rgb = tuple(round(start[i] + t * (mid[i] - start[i])) for i in range(3))
        else:
            t = (value - 0.5) / 0.5
            rgb = tuple(round(mid[i] + t * (end[i] - mid[i])) for i in range(3))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#202124}",
        ".title{font-size:20px;font-weight:700}",
        ".axis{font-size:12px;font-weight:600}",
        ".tick{font-size:11px}",
        ".cell{font-size:11px;font-weight:700}",
        ".muted{fill:#5f6368}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="title" x="24" y="34">Judge Scores by Trace</text>',
        '<text class="muted tick" x="24" y="54">Average normalized rubric score</text>',
    ]

    for col_idx, judge in enumerate(judges):
        x = left + col_idx * cell_w + cell_w / 2
        svg.append(
            f'<text class="axis" x="{x:.1f}" y="{top - 18}" text-anchor="middle">'
            f"{html.escape(judge)}</text>"
        )

    for row_idx, trace_label in enumerate(trace_labels):
        y = top + row_idx * cell_h + cell_h / 2 + 4
        svg.append(
            f'<text class="tick" x="{left - 12}" y="{y:.1f}" text-anchor="end">'
            f"{html.escape(trace_label)}</text>"
        )
        for col_idx, value in enumerate(values[row_idx]):
            x = left + col_idx * cell_w
            y0 = top + row_idx * cell_h
            fill = color_for(value)
            label = "missing" if math.isnan(value) else f"{value:.0%}"
            text_color = "#202124" if math.isnan(value) or value < 0.62 else "#ffffff"
            svg.append(
                f'<rect x="{x}" y="{y0}" width="{cell_w}" height="{cell_h}" '
                f'fill="{fill}" stroke="#ffffff"/>'
            )
            svg.append(
                f'<text class="cell" x="{x + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 4:.1f}" '
                f'text-anchor="middle" fill="{text_color}">{label}</text>'
            )

    legend_x = left + cell_w * len(judges) + 34
    legend_y = top
    svg.append(f'<text class="axis" x="{legend_x}" y="{legend_y - 14}">Score</text>')
    for idx, value in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        y = legend_y + idx * 26
        svg.append(
            f'<rect x="{legend_x}" y="{y}" width="24" height="18" '
            f'fill="{color_for(value)}" stroke="#ffffff"/>'
        )
        svg.append(f'<text class="tick" x="{legend_x + 32}" y="{y + 13}">{value:.0%}</text>')
    svg.append(
        f'<rect x="{legend_x}" y="{legend_y + 140}" width="24" height="18" '
        'fill="#f2f2f2" stroke="#d0d0d0"/>'
    )
    svg.append(f'<text class="tick" x="{legend_x + 32}" y="{legend_y + 153}">missing</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def summarize_trace_judge(trace_label: str, judge: str, rows: dict[tuple[str, int], dict]) -> dict:
    scores = [score for row in rows.values() if (score := normalized_score(row)) is not None]
    correctness = [
        correct for row in rows.values() if (correct := as_bool(row.get("final_answer_correct"))) is not None
    ]
    points_earned = [
        earned for row in rows.values() if (earned := as_float(row.get("rubric_points_earned"))) is not None
    ]
    points_possible = [
        possible for row in rows.values() if (possible := as_float(row.get("rubric_points_possible"))) is not None
    ]
    return {
        "trace_label": trace_label,
        "judge": judge,
        "rows": len(rows),
        "scored_rows": len(scores),
        "avg_normalized_score": fmt(mean(scores) if scores else None),
        "avg_points_earned": fmt(mean(points_earned) if points_earned else None),
        "avg_points_possible": fmt(mean(points_possible) if points_possible else None),
        "final_answer_correct_rate": fmt(mean(correctness) if correctness else None),
    }


def pairwise_delta_rows(
    trace_label: str, by_judge: dict[str, dict[tuple[str, int], dict]]
) -> list[dict]:
    rows = []
    for judge_a, judge_b in combinations(by_judge, 2):
        common_keys = sorted(set(by_judge[judge_a]) & set(by_judge[judge_b]))
        score_deltas = []
        abs_score_deltas = []
        correctness_diffs = []
        for key in common_keys:
            score_a = normalized_score(by_judge[judge_a][key])
            score_b = normalized_score(by_judge[judge_b][key])
            if score_a is not None and score_b is not None:
                delta = score_a - score_b
                score_deltas.append(delta)
                abs_score_deltas.append(abs(delta))

            correct_a = as_bool(by_judge[judge_a][key].get("final_answer_correct"))
            correct_b = as_bool(by_judge[judge_b][key].get("final_answer_correct"))
            if correct_a is not None and correct_b is not None:
                correctness_diffs.append(correct_a != correct_b)

        rows.append(
            {
                "trace_label": trace_label,
                "judge_a": judge_a,
                "judge_b": judge_b,
                "common_rows": len(common_keys),
                "mean_normalized_score_delta_a_minus_b": fmt(
                    mean(score_deltas) if score_deltas else None
                ),
                "mean_abs_normalized_score_delta": fmt(
                    mean(abs_score_deltas) if abs_score_deltas else None
                ),
                "max_abs_normalized_score_delta": fmt(
                    max(abs_score_deltas) if abs_score_deltas else None
                ),
                "final_answer_correct_disagreement_rate": fmt(
                    mean(correctness_diffs) if correctness_diffs else None
                ),
            }
        )
    return rows


def build_item_rows(
    trace_label: str, by_judge: dict[str, dict[tuple[str, int], dict]]
) -> list[dict]:
    all_keys = sorted(set().union(*(set(rows) for rows in by_judge.values())))
    item_rows = []
    for question_id, trial_idx in all_keys:
        row = {
            "trace_label": trace_label,
            "question_id": question_id,
            "trial_idx": trial_idx,
        }
        scores = []
        for judge, rows in by_judge.items():
            judge_row = rows.get((question_id, trial_idx))
            score = normalized_score(judge_row) if judge_row else None
            correct = as_bool(judge_row.get("final_answer_correct")) if judge_row else None
            earned = as_float(judge_row.get("rubric_points_earned")) if judge_row else None
            possible = as_float(judge_row.get("rubric_points_possible")) if judge_row else None
            row[f"{judge}_normalized_score"] = fmt(score)
            row[f"{judge}_points_earned"] = fmt(earned, digits=2)
            row[f"{judge}_points_possible"] = fmt(possible, digits=2)
            row[f"{judge}_final_answer_correct"] = "" if correct is None else str(correct).lower()
            if score is not None:
                scores.append(score)
        row["score_spread"] = fmt(max(scores) - min(scores) if len(scores) >= 2 else None)
        item_rows.append(row)
    return item_rows


def main() -> None:
    args = parse_args()
    judges = {judge: DEFAULT_JUDGE_FILES[judge] for judge in args.judges}
    trace_labels = discover_traces(args.traces_dir)

    loaded_by_trace: dict[str, dict[str, dict[tuple[str, int], dict]]] = defaultdict(dict)
    missing_files = []
    for trace_label in trace_labels:
        for judge, file_name in judges.items():
            path = args.grade_results_dir / trace_label / file_name
            if path.exists():
                loaded_by_trace[trace_label][judge] = load_judge_rows(path)
            else:
                loaded_by_trace[trace_label][judge] = {}
                missing_files.append(str(path))

    summary_rows = []
    wide_rows = []
    pairwise_rows = []
    item_rows = []
    for trace_label in trace_labels:
        by_judge = loaded_by_trace[trace_label]
        summary_rows.extend(
            summarize_trace_judge(trace_label, judge, by_judge[judge]) for judge in judges
        )

        wide_row = {"trace_label": trace_label}
        trace_scores = []
        for judge in judges:
            scores = [
                score
                for row in by_judge[judge].values()
                if (score := normalized_score(row)) is not None
            ]
            avg_score = mean(scores) if scores else None
            wide_row[f"{judge}_avg_normalized_score"] = fmt(avg_score)
            wide_row[f"{judge}_avg_normalized_percent"] = pct(avg_score)
            wide_row[f"{judge}_rows"] = len(by_judge[judge])
            if avg_score is not None:
                trace_scores.append(avg_score)
        wide_row["avg_score_spread"] = fmt(
            max(trace_scores) - min(trace_scores) if len(trace_scores) >= 2 else None
        )
        wide_rows.append(wide_row)

        pairwise_rows.extend(pairwise_delta_rows(trace_label, by_judge))
        item_rows.extend(build_item_rows(trace_label, by_judge))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "judge_scores_by_trace.csv"
    wide_path = args.out_dir / "judge_score_wide_by_trace.csv"
    pairwise_path = args.out_dir / "judge_pairwise_deltas_by_trace.csv"
    item_path = args.out_dir / "judge_scores_by_item.csv"
    plot_path = args.out_dir / "plots" / "judge_scores_by_trace_heatmap.svg"
    manifest_path = args.out_dir / "judge_score_comparison_manifest.json"

    write_csv(summary_path, summary_rows)
    write_csv(wide_path, wide_rows)
    write_csv(pairwise_path, pairwise_rows)
    write_csv(item_path, item_rows)
    write_score_heatmap(plot_path, wide_rows, list(judges))
    manifest_path.write_text(
        json.dumps(
            {
                "trace_labels": trace_labels,
                "judge_files": judges,
                "missing_files": missing_files,
                "outputs": {
                    "by_trace": str(summary_path),
                    "wide_by_trace": str(wide_path),
                    "pairwise_deltas_by_trace": str(pairwise_path),
                    "by_item": str(item_path),
                    "score_heatmap": str(plot_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Average normalized rubric score by trace and judge:")
    for row in wide_rows:
        judge_bits = [
            f"{judge}={row[f'{judge}_avg_normalized_percent'] or 'missing'}"
            for judge in judges
        ]
        print(
            f"  {row['trace_label']}: "
            f"{', '.join(judge_bits)}; spread={pct(float(row['avg_score_spread'])) if row['avg_score_spread'] else ''}"
        )

    if missing_files:
        print("\nMissing judge files:")
        for path in missing_files:
            print(f"  {path}")

    print(f"\nWrote {summary_path}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {pairwise_path}")
    print(f"Wrote {item_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
