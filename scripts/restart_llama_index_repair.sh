#!/bin/zsh
set -eu

root=/Users/anshu/Desktop/bfb-llm-as-judge
cd "$root"

source_traces=runs/nvidia-grades/gpt55.traces.llama.index-repair.jsonl
output=runs/nvidia-grades/gpt55.grades.llama.index-repair.jsonl
pending=runs/nvidia-grades/gpt55.traces.llama.index-repair.pending.jsonl
log=runs/nvidia-grades/llama-index-repair.log
status_log=runs/nvidia-grades/llama-index-repair-status.log

live=($(pgrep -f "run_nvidia_grader.py.*gpt55.grades.llama.index-repair.jsonl" || true))
if (( ${#live[@]} > 1 )); then
  print "multiple index-repair graders found: ${live[*]}" > "$status_log"
  exit 1
fi
if (( ${#live[@]} == 1 )); then
  kill "$live[1]"
  sleep 1
fi

SOURCE_TRACES="$source_traces" OUTPUT="$output" PENDING="$pending" \
  .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["SOURCE_TRACES"])
output = Path(os.environ["OUTPUT"])
pending = Path(os.environ["PENDING"])

clean = {}
if output.exists():
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        complete = all(
            str(item.get("judge_explanation") or "").strip().lower() not in ("", "missing")
            for item in record.get("rubric_lines", [])
        )
        if complete:
            clean[(record["question_id"], record["trial_idx"])] = record

payload = "".join(
    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    for record in clean.values()
)
output.write_text(payload, encoding="utf-8")

missing = []
for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    if (record["question_id"], record["trial_idx"]) not in clean:
        missing.append(line)
pending.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
print(f"clean={len(clean)} pending={len(missing)}")
PY

count=$(wc -l < "$pending" | tr -d ' ')
print "$(date '+%F %T') preserved clean records; pending $count" > "$status_log"
if (( count == 0 )); then
  exit 0
fi

nohup env BFB_JUDGE_REQUEST_TIMEOUT=600 BFB_JUDGE_NUM_RETRIES=2 \
  .venv/bin/python scripts/run_nvidia_grader.py \
  --judge llama \
  --traces "$pending" \
  --sample-n "$count" \
  --output "$output" \
  > "$log" 2>&1 &
pid=$!
print "grader PID $pid" >> "$status_log"
caffeinate -i -w "$pid" >/dev/null 2>&1 &
print "caffeinate attached" >> "$status_log"
