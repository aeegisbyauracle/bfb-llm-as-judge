"""Compare local judge grades against the paper Gemini and Opus grades.

Results are written to ``analysis/results`` by default.
Only candidate judges with files for all 10 trace types, exactly 150 rows per
file, and 150 unique `(question_id, trial_idx)` keys per file are included.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bfb-matplotlib-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_JUDGE_FILES = {
    "tinker": "tinker-gpt-oss-20b.jsonl",
    "qwen": "nvidia-qwen.jsonl",
    "llama": "vllm-llama.jsonl",
    "nemotron": "nvidia-nemotron.jsonl",
}
GROUND_TRUTH_JUDGES = ("gemini", "opus")
DEFAULT_EXPECTED_ROWS = 150


@dataclass(frozen=True)
class Grade:
    trace_label: str
    question_id: str
    trial_idx: int
    final_answer_correct: bool | None
    rubric_points_earned: float | None
    rubric_points_possible: float | None
    rubric_line_earned: tuple[bool | None, ...]

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.trace_label, self.question_id, self.trial_idx)

    @property
    def normalized_points(self) -> float | None:
        if self.rubric_points_earned is None or not self.rubric_points_possible:
            return None
        return self.rubric_points_earned / self.rubric_points_possible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grade-results-dir", type=Path, default=Path("grade-results"))
    parser.add_argument(
        "--paper-grades-dir",
        type=Path,
        default=Path("data/raw/big-finance-benchmark/grades"),
    )
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
        help="Required non-empty JSONL row count for each candidate judge trace file.",
    )
    return parser.parse_args()


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


def as_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def as_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def load_grade_file(path: Path, trace_label: str) -> tuple[dict[tuple[str, str, int], Grade], int]:
    out = {}
    duplicates = 0
    for row in read_jsonl(path):
        question_id = row.get("question_id")
        if not question_id:
            continue
        rubric_lines = row.get("rubric_lines") or []
        grade = Grade(
            trace_label=trace_label,
            question_id=str(question_id),
            trial_idx=int(row.get("trial_idx", 0)),
            final_answer_correct=as_bool(row.get("final_answer_correct")),
            rubric_points_earned=as_float(row.get("rubric_points_earned")),
            rubric_points_possible=as_float(row.get("rubric_points_possible")),
            rubric_line_earned=tuple(as_bool(line.get("earned")) for line in rubric_lines),
        )
        if grade.key in out:
            duplicates += 1
        out[grade.key] = grade
    return out, duplicates


def discover_traces(traces_dir: Path) -> list[str]:
    labels = sorted(path.stem.split(".traces", 1)[0] for path in traces_dir.glob("*.traces.jsonl"))
    if not labels:
        raise FileNotFoundError(f"No traces found in {traces_dir}")
    return labels


def complete_candidate_judges(
    grade_results_dir: Path, trace_labels: list[str], expected_rows: int
) -> tuple[dict[str, str], dict[str, list[str]]]:
    included = {}
    excluded = {}
    for judge, file_name in DEFAULT_JUDGE_FILES.items():
        reasons = []
        for label in trace_labels:
            path = grade_results_dir / label / file_name
            if not path.exists():
                reasons.append(f"{label}: missing {file_name}")
                continue
            row_count, unique_count, duplicate_count = inspect_candidate_file(path)
            if row_count != expected_rows:
                reasons.append(f"{label}: {row_count} rows, expected {expected_rows}")
            if unique_count != expected_rows:
                reasons.append(
                    f"{label}: {unique_count} unique keys, expected {expected_rows}"
                )
            if duplicate_count:
                reasons.append(f"{label}: {duplicate_count} duplicate key row(s)")
        if reasons:
            excluded[judge] = reasons
        else:
            included[judge] = file_name
    return included, excluded


def inspect_candidate_file(path: Path) -> tuple[int, int, int]:
    """Return `(non_empty_rows, unique_keys, duplicate_key_rows)` for one grade file."""
    row_count = 0
    duplicate_count = 0
    keys = set()
    for row in read_jsonl(path):
        row_count += 1
        question_id = row.get("question_id")
        trial_idx = row.get("trial_idx", 0)
        key = (question_id, trial_idx)
        if key in keys:
            duplicate_count += 1
        keys.add(key)
    return row_count, len(keys), duplicate_count


def load_judge_set(
    grade_results_dir: Path, trace_labels: list[str], file_name: str
) -> tuple[dict[tuple[str, str, int], Grade], Counter]:
    rows = {}
    duplicates = Counter()
    for label in trace_labels:
        loaded, duplicate_count = load_grade_file(grade_results_dir / label / file_name, label)
        rows.update(loaded)
        duplicates[label] = duplicate_count
    duplicates["total"] = sum(duplicates.values())
    return rows, duplicates


def load_available_judge_set(
    grade_results_dir: Path, trace_labels: list[str], file_name: str
) -> tuple[dict[tuple[str, str, int], Grade], Counter, int]:
    rows = {}
    duplicates = Counter()
    files_loaded = 0
    for label in trace_labels:
        path = grade_results_dir / label / file_name
        if not path.exists():
            continue
        loaded, duplicate_count = load_grade_file(path, label)
        rows.update(loaded)
        duplicates[label] = duplicate_count
        files_loaded += 1
    duplicates["total"] = sum(duplicates.values())
    return rows, duplicates, files_loaded


def load_ground_truth(
    paper_grades_dir: Path, trace_labels: list[str], judge: str
) -> tuple[dict[tuple[str, str, int], Grade], Counter]:
    rows = {}
    duplicates = Counter()
    for label in trace_labels:
        loaded, duplicate_count = load_grade_file(paper_grades_dir / f"{label}.grades.{judge}.jsonl", label)
        rows.update(loaded)
        duplicates[label] = duplicate_count
    duplicates["total"] = sum(duplicates.values())
    return rows, duplicates


def accuracy(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0 or y_den == 0:
        return None
    return numerator / (x_den * y_den)


def compare_pair(candidate: dict[tuple[str, str, int], Grade], truth: dict[tuple[str, str, int], Grade]) -> dict:
    keys = sorted(set(candidate) & set(truth))
    final_compared = final_agree = 0
    tp = fp = tn = fn = 0
    rubric_line_compared = rubric_line_agree = 0
    rubric_vector_compared = rubric_vector_agree = 0
    rubric_length_mismatches = 0
    point_abs_errors = []
    normalized_abs_errors = []
    candidate_points = []
    truth_points = []
    candidate_norms = []
    truth_norms = []

    for key in keys:
        cand = candidate[key]
        gt = truth[key]
        if cand.final_answer_correct is not None and gt.final_answer_correct is not None:
            final_compared += 1
            final_agree += int(cand.final_answer_correct == gt.final_answer_correct)
            if cand.final_answer_correct and gt.final_answer_correct:
                tp += 1
            elif cand.final_answer_correct and not gt.final_answer_correct:
                fp += 1
            elif not cand.final_answer_correct and gt.final_answer_correct:
                fn += 1
            else:
                tn += 1

        if cand.rubric_points_earned is not None and gt.rubric_points_earned is not None:
            point_abs_errors.append(abs(cand.rubric_points_earned - gt.rubric_points_earned))
            candidate_points.append(cand.rubric_points_earned)
            truth_points.append(gt.rubric_points_earned)

        cand_norm = cand.normalized_points
        gt_norm = gt.normalized_points
        if cand_norm is not None and gt_norm is not None:
            normalized_abs_errors.append(abs(cand_norm - gt_norm))
            candidate_norms.append(cand_norm)
            truth_norms.append(gt_norm)

        if len(cand.rubric_line_earned) != len(gt.rubric_line_earned):
            rubric_length_mismatches += 1
        vector_agrees = len(cand.rubric_line_earned) == len(gt.rubric_line_earned)
        for cand_line, gt_line in zip(cand.rubric_line_earned, gt.rubric_line_earned):
            if cand_line is None or gt_line is None:
                vector_agrees = False
                continue
            rubric_line_compared += 1
            rubric_line_agree += int(cand_line == gt_line)
            vector_agrees = vector_agrees and cand_line == gt_line
        if cand.rubric_line_earned:
            rubric_vector_compared += 1
            rubric_vector_agree += int(vector_agrees)

    return {
        "overlap_rows": len(keys),
        "missing_candidate_rows": len(set(truth) - set(candidate)),
        "missing_truth_rows": len(set(candidate) - set(truth)),
        "final_answer_compared": final_compared,
        "final_answer_agreement": accuracy(final_agree, final_compared),
        "final_answer_tp": tp,
        "final_answer_fp": fp,
        "final_answer_tn": tn,
        "final_answer_fn": fn,
        "rubric_line_compared": rubric_line_compared,
        "rubric_line_agreement": accuracy(rubric_line_agree, rubric_line_compared),
        "rubric_vector_compared": rubric_vector_compared,
        "rubric_vector_agreement": accuracy(rubric_vector_agree, rubric_vector_compared),
        "rubric_length_mismatches": rubric_length_mismatches,
        "points_mae": mean(point_abs_errors) if point_abs_errors else None,
        "normalized_points_mae": mean(normalized_abs_errors) if normalized_abs_errors else None,
        "points_pearson": pearson(candidate_points, truth_points),
        "normalized_points_pearson": pearson(candidate_norms, truth_norms),
    }


def compare_by_trace(candidate: dict[tuple[str, str, int], Grade], truth: dict[tuple[str, str, int], Grade], labels: list[str]) -> list[dict]:
    rows = []
    for label in labels:
        candidate_subset = {key: value for key, value in candidate.items() if key[0] == label}
        truth_subset = {key: value for key, value in truth.items() if key[0] == label}
        rows.append({"trace_label": label, **compare_pair(candidate_subset, truth_subset)})
    return rows


def format_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else value


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
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def metric_label(metric: str) -> str:
    return {
        "final_answer_agreement": "Final-Answer Agreement",
        "rubric_line_agreement": "Rubric-Line Agreement",
        "normalized_points_mae": "Normalized Points MAE",
    }.get(metric, metric)


def plot_metric_bars(rows: list[dict], metric: str, out_path: Path) -> bool:
    plot_rows = [
        row
        for row in rows
        if row.get(metric) is not None
        and row.get("comparison_type") != "paper_judge_baseline"
    ]
    if not plot_rows:
        return False

    labels = [f"{row['candidate_judge']} vs {row['ground_truth_judge']}" for row in plot_rows]
    values = [row[metric] for row in plot_rows]
    colors = ["#4c78a8" for _ in plot_rows]

    height = max(3.5, 0.45 * len(plot_rows) + 1.2)
    fig, ax = plt.subplots(figsize=(9, height))
    y_positions = list(range(len(plot_rows)))
    ax.barh(y_positions, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric_label(metric))
    ax.set_title(metric_label(metric) + " by Judge Comparison")

    if metric.endswith("agreement"):
        ax.set_xlim(0, 1)
        for y_pos, value in zip(y_positions, values):
            ax.text(min(value + 0.015, 0.98), y_pos, f"{value:.1%}", va="center")
    else:
        max_value = max(values) if values else 1
        ax.set_xlim(0, max_value * 1.15 if max_value else 1)
        for y_pos, value in zip(y_positions, values):
            ax.text(value + max_value * 0.015, y_pos, f"{value:.3f}", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_trace_heatmap(rows: list[dict], out_path: Path) -> bool:
    plot_rows = [
        row
        for row in rows
        if row.get("final_answer_agreement") is not None
        and row.get("candidate_judge")
        and row.get("ground_truth_judge")
        and row.get("trace_label")
    ]
    if not plot_rows:
        return False

    comparisons = sorted(
        {f"{row['candidate_judge']} vs {row['ground_truth_judge']}" for row in plot_rows}
    )
    trace_labels = sorted({row["trace_label"] for row in plot_rows})
    values = {
        (f"{row['candidate_judge']} vs {row['ground_truth_judge']}", row["trace_label"]): row[
            "final_answer_agreement"
        ]
        for row in plot_rows
    }
    matrix = [
        [values.get((comparison, trace_label), float("nan")) for trace_label in trace_labels]
        for comparison in comparisons
    ]

    fig_width = max(9, 0.72 * len(trace_labels) + 3)
    fig_height = max(3.5, 0.42 * len(comparisons) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(trace_labels)))
    ax.set_xticklabels(trace_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(comparisons)))
    ax.set_yticklabels(comparisons)
    ax.set_title("Final-Answer Agreement by Trace")

    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if math.isnan(value):
                continue
            ax.text(x, y, f"{value:.0%}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Final-Answer Agreement")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_paper_baseline(summary_rows: list[dict], out_path: Path) -> bool:
    baseline = next(
        (row for row in summary_rows if row.get("comparison_type") == "paper_judge_baseline"),
        None,
    )
    if not baseline:
        return False

    metrics = [
        ("Final-answer agreement", baseline.get("final_answer_agreement")),
        ("Rubric-line agreement", baseline.get("rubric_line_agreement")),
        ("Normalized points Pearson", baseline.get("normalized_points_pearson")),
    ]
    metrics = [(label, value) for label, value in metrics if value is not None]
    if not metrics:
        return False

    labels = [label for label, _ in metrics]
    values = [value for _, value in metrics]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x_positions = list(range(len(metrics)))
    ax.bar(x_positions, values, color="#8f8f8f", edgecolor="white", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Paper Judge Agreement Reference: Gemini vs Opus")
    for x_pos, value in zip(x_positions, values):
        ax.text(x_pos, min(value + 0.025, 0.98), f"{value:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def reason_trace_label(reason: str) -> str:
    return reason.split(":", 1)[0]


def plot_candidate_completeness(
    excluded: dict[str, list[str]],
    included: dict[str, str],
    trace_labels: list[str],
    out_path: Path,
) -> bool:
    judges = list(DEFAULT_JUDGE_FILES)
    if not judges:
        return False

    total_files = len(trace_labels)
    complete_counts = []
    incomplete_counts = []
    for judge in judges:
        if judge in included:
            complete_counts.append(total_files)
            incomplete_counts.append(0)
            continue
        incomplete_traces = {reason_trace_label(reason) for reason in excluded.get(judge, [])}
        incomplete_counts.append(len(incomplete_traces))
        complete_counts.append(total_files - len(incomplete_traces))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x_positions = list(range(len(judges)))
    ax.bar(x_positions, complete_counts, color="#59a14f", label="Complete files")
    ax.bar(
        x_positions,
        incomplete_counts,
        bottom=complete_counts,
        color="#e15759",
        label="Incomplete files",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(judges)
    ax.set_ylim(0, total_files)
    ax.set_ylabel("Trace files")
    ax.set_title("Candidate Judge Completeness (10 Files Required)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    for x_pos, complete_count, incomplete_count in zip(
        x_positions, complete_counts, incomplete_counts
    ):
        ax.text(
            x_pos,
            total_files - 0.35,
            f"{complete_count}/{total_files}",
            ha="center",
            va="top",
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_exclusion_issue_counts(excluded: dict[str, list[str]], out_path: Path) -> bool:
    categories = ["missing file", "row count", "unique keys", "duplicate keys"]
    judges = list(DEFAULT_JUDGE_FILES)
    counts = {judge: {category: 0 for category in categories} for judge in judges}

    for judge, reasons in excluded.items():
        for reason in reasons:
            if "missing" in reason:
                counts[judge]["missing file"] += 1
            elif " rows, expected " in reason:
                counts[judge]["row count"] += 1
            elif " unique keys, expected " in reason:
                counts[judge]["unique keys"] += 1
            elif "duplicate key" in reason:
                counts[judge]["duplicate keys"] += 1

    if not any(sum(counts[judge].values()) for judge in judges):
        return False

    colors = {
        "missing file": "#e15759",
        "row count": "#f28e2b",
        "unique keys": "#edc948",
        "duplicate keys": "#b07aa1",
    }
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x_positions = list(range(len(judges)))
    bottoms = [0] * len(judges)
    for category in categories:
        values = [counts[judge][category] for judge in judges]
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            color=colors[category],
            label=category,
            edgecolor="white",
            linewidth=0.8,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_xticks(x_positions)
    ax.set_xticklabels(judges)
    ax.set_ylabel("Completeness issues")
    ax.set_title("Why Candidate Judges Were Excluded")
    ax.legend(frameon=False, loc="upper right")
    for x_pos, total in zip(x_positions, bottoms):
        if total:
            ax.text(
                x_pos,
                total - 0.12,
                str(total),
                ha="center",
                va="top",
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_available_metric_grouped(rows: list[dict], metric: str, out_path: Path) -> bool:
    plot_rows = [
        row
        for row in rows
        if row.get(metric) is not None and row.get("ground_truth_judge") in GROUND_TRUTH_JUDGES
    ]
    if not plot_rows:
        return False

    judges = [judge for judge in DEFAULT_JUDGE_FILES if any(row["candidate_judge"] == judge for row in plot_rows)]
    truth_judges = [truth for truth in GROUND_TRUTH_JUDGES if any(row["ground_truth_judge"] == truth for row in plot_rows)]
    values = {
        (row["candidate_judge"], row["ground_truth_judge"]): row[metric]
        for row in plot_rows
    }
    overlaps = {
        (row["candidate_judge"], row["ground_truth_judge"]): row.get("overlap_rows", 0)
        for row in plot_rows
    }

    fig, ax = plt.subplots(figsize=(9.5, 5))
    x_positions = list(range(len(judges)))
    width = 0.34 if len(truth_judges) > 1 else 0.5
    colors = {"gemini": "#4c78a8", "opus": "#f58518"}

    for i, truth in enumerate(truth_judges):
        offset = (i - (len(truth_judges) - 1) / 2) * width
        bar_positions = [x + offset for x in x_positions]
        bar_values = [values.get((judge, truth), float("nan")) for judge in judges]
        bars = ax.bar(
            bar_positions,
            bar_values,
            width=width,
            label=f"vs {truth}",
            color=colors.get(truth, "#888888"),
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, judge, value in zip(bars, judges, bar_values):
            if math.isnan(value):
                continue
            overlap = overlaps.get((judge, truth), 0)
            if metric.endswith("agreement") or metric.endswith("pearson"):
                label = f"{value:.1%}\nn={overlap}"
            else:
                label = f"{value:.3f}\nn={overlap}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(judges)
    ax.set_ylabel(metric_label(metric))
    ax.set_title(metric_label(metric) + " vs Gemini and Opus")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    if metric.endswith("agreement") or metric.endswith("pearson"):
        ax.set_ylim(0, 1.08)
    else:
        finite_values = [value for value in values.values() if value is not None]
        max_value = max(finite_values) if finite_values else 1
        ax.set_ylim(0, max_value * 1.25 if max_value else 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def write_plots(
    out_dir: Path,
    summary_rows: list[dict],
    trace_rows: list[dict],
    available_summary_rows: list[dict],
    included: dict[str, str],
    excluded: dict[str, list[str]],
    trace_labels: list[str],
) -> dict[str, str]:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    outputs: dict[str, str] = {}
    known_plot_files = [
        "final_answer_agreement.png",
        "rubric_line_agreement.png",
        "normalized_points_mae.png",
        "final_answer_agreement_by_trace.png",
        "paper_judge_baseline.png",
        "candidate_completeness.png",
        "candidate_exclusion_issues.png",
        "available_final_answer_agreement_by_judge.png",
        "available_rubric_line_agreement_by_judge.png",
        "available_normalized_points_mae_by_judge.png",
    ]
    for file_name in known_plot_files:
        path = plots_dir / file_name
        if path.exists():
            path.unlink()

    plot_specs = [
        ("final_answer_agreement", "final_answer_agreement.png"),
        ("rubric_line_agreement", "rubric_line_agreement.png"),
        ("normalized_points_mae", "normalized_points_mae.png"),
    ]
    for metric, file_name in plot_specs:
        path = plots_dir / file_name
        if plot_metric_bars(summary_rows, metric, path):
            outputs[metric] = str(path)

    heatmap_path = plots_dir / "final_answer_agreement_by_trace.png"
    if plot_trace_heatmap(trace_rows, heatmap_path):
        outputs["final_answer_agreement_by_trace"] = str(heatmap_path)

    baseline_path = plots_dir / "paper_judge_baseline.png"
    if plot_paper_baseline(summary_rows, baseline_path):
        outputs["paper_judge_baseline"] = str(baseline_path)

    completeness_path = plots_dir / "candidate_completeness.png"
    if plot_candidate_completeness(excluded, included, trace_labels, completeness_path):
        outputs["candidate_completeness"] = str(completeness_path)

    exclusions_path = plots_dir / "candidate_exclusion_issues.png"
    if plot_exclusion_issue_counts(excluded, exclusions_path):
        outputs["candidate_exclusion_issues"] = str(exclusions_path)

    available_plot_specs = [
        ("final_answer_agreement", "available_final_answer_agreement_by_judge.png"),
        ("rubric_line_agreement", "available_rubric_line_agreement_by_judge.png"),
        ("normalized_points_mae", "available_normalized_points_mae_by_judge.png"),
    ]
    for metric, file_name in available_plot_specs:
        path = plots_dir / file_name
        if plot_available_metric_grouped(available_summary_rows, metric, path):
            outputs[f"available_{metric}_by_judge"] = str(path)

    return outputs


def main() -> None:
    args = parse_args()
    trace_labels = discover_traces(args.traces_dir)
    included, excluded = complete_candidate_judges(
        args.grade_results_dir, trace_labels, args.expected_rows
    )

    candidate_rows = {}
    duplicate_summary = {}
    for judge, file_name in included.items():
        rows, duplicates = load_judge_set(args.grade_results_dir, trace_labels, file_name)
        candidate_rows[judge] = rows
        duplicate_summary[f"candidate:{judge}"] = dict(duplicates)

    ground_truth_rows = {}
    for judge in GROUND_TRUTH_JUDGES:
        rows, duplicates = load_ground_truth(args.paper_grades_dir, trace_labels, judge)
        ground_truth_rows[judge] = rows
        duplicate_summary[f"ground_truth:{judge}"] = dict(duplicates)

    available_candidate_rows = {}
    available_file_counts = {}
    for judge, file_name in DEFAULT_JUDGE_FILES.items():
        rows, duplicates, files_loaded = load_available_judge_set(
            args.grade_results_dir, trace_labels, file_name
        )
        if not rows:
            continue
        available_candidate_rows[judge] = rows
        available_file_counts[judge] = files_loaded
        duplicate_summary[f"available_candidate:{judge}"] = dict(duplicates)

    summary_rows = []
    trace_rows = []
    for candidate_name, rows in candidate_rows.items():
        for truth_name, truth in ground_truth_rows.items():
            summary_rows.append(
                {
                    "candidate_judge": candidate_name,
                    "ground_truth_judge": truth_name,
                    **compare_pair(rows, truth),
                }
            )
            for trace_row in compare_by_trace(rows, truth, trace_labels):
                trace_rows.append(
                    {
                        "candidate_judge": candidate_name,
                        "ground_truth_judge": truth_name,
                        **trace_row,
                    }
                )

    summary_rows.append(
        {
            "candidate_judge": "gemini",
            "ground_truth_judge": "opus",
            "comparison_type": "paper_judge_baseline",
            **compare_pair(ground_truth_rows["gemini"], ground_truth_rows["opus"]),
        }
    )

    available_summary_rows = []
    for candidate_name, rows in available_candidate_rows.items():
        for truth_name, truth in ground_truth_rows.items():
            available_summary_rows.append(
                {
                    "candidate_judge": candidate_name,
                    "ground_truth_judge": truth_name,
                    "comparison_type": "available_rows_diagnostic",
                    "candidate_files_loaded": available_file_counts[candidate_name],
                    "candidate_files_expected": len(trace_labels),
                    **compare_pair(rows, truth),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "summary_by_judge.csv", summary_rows)
    write_csv(args.out_dir / "summary_by_trace.csv", trace_rows)
    write_csv(args.out_dir / "available_summary_by_judge.csv", available_summary_rows)
    plot_outputs = write_plots(
        args.out_dir,
        summary_rows,
        trace_rows,
        available_summary_rows,
        included,
        excluded,
        trace_labels,
    )
    manifest = {
        "trace_labels": trace_labels,
        "expected_rows_per_candidate_file": args.expected_rows,
        "included_candidate_judges": included,
        "excluded_candidate_judges": excluded,
        "duplicate_rows_last_row_kept": duplicate_summary,
        "outputs": {
            "summary_by_judge": str(args.out_dir / "summary_by_judge.csv"),
            "summary_by_trace": str(args.out_dir / "summary_by_trace.csv"),
            "available_summary_by_judge": str(args.out_dir / "available_summary_by_judge.csv"),
            "plots": plot_outputs,
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Included judges: {', '.join(included) if included else '(none)'}")
    for judge, reasons in excluded.items():
        print(f"Excluded {judge}: {len(reasons)} completeness issue(s): {', '.join(reasons)}")
    print(f"Wrote {args.out_dir / 'summary_by_judge.csv'}")
    print(f"Wrote {args.out_dir / 'summary_by_trace.csv'}")
    print(f"Wrote {args.out_dir / 'available_summary_by_judge.csv'}")
    for plot_path in plot_outputs.values():
        print(f"Wrote {plot_path}")
    print(f"Wrote {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
