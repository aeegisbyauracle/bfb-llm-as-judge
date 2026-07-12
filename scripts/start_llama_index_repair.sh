#!/bin/zsh
set -eu

root=/Users/anshu/Desktop/bfb-llm-as-judge
cd "$root"

source_traces=data/raw/big-finance-benchmark/traces/gpt55.traces.jsonl
raw_grades=runs/nvidia-grades/gpt55.grades.llama.repaired.jsonl
repair_traces=runs/nvidia-grades/gpt55.traces.llama.index-repair.jsonl
repair_grades=runs/nvidia-grades/gpt55.grades.llama.index-repair.jsonl
log=runs/nvidia-grades/llama-index-repair.log
status_log=runs/nvidia-grades/llama-index-repair-status.log

if pgrep -f "run_nvidia_grader.py.*gpt55.grades.llama.index-repair.jsonl" >/dev/null; then
  print "index repair already running; refusing duplicate" > "$status_log"
  exit 0
fi

SOURCE_TRACES="$source_traces" RAW_GRADES="$raw_grades" REPAIR_TRACES="$repair_traces" \
  .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["SOURCE_TRACES"])
raw = Path(os.environ["RAW_GRADES"])
dest = Path(os.environ["REPAIR_TRACES"])

affected = set()
for line in raw.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    if any(
        not str(rubric.get("judge_explanation") or "").strip()
        or str(rubric.get("judge_explanation")).strip().lower() == "missing"
        for rubric in record.get("rubric_lines", [])
    ):
        affected.add((record["question_id"], record["trial_idx"]))

selected = []
for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    if (record["question_id"], record["trial_idx"]) in affected:
        selected.append(line)

if len(selected) != len(affected):
    raise SystemExit(f"trace mismatch: selected={len(selected)} affected={len(affected)}")
dest.write_text("\n".join(selected) + "\n", encoding="utf-8")
print(len(selected))
PY

count=$(wc -l < "$repair_traces" | tr -d ' ')
print "$(date '+%F %T') prepared $count affected traces" > "$status_log"

nohup env BFB_JUDGE_REQUEST_TIMEOUT=600 BFB_JUDGE_NUM_RETRIES=2 \
  .venv/bin/python scripts/run_nvidia_grader.py \
  --judge llama \
  --traces "$repair_traces" \
  --sample-n "$count" \
  --output "$repair_grades" \
  > "$log" 2>&1 &
pid=$!
print "grader PID $pid" >> "$status_log"

caffeinate -i -w "$pid" >/dev/null 2>&1 &
print "caffeinate attached" >> "$status_log"
