#!/bin/zsh
set -eu

old_pid=${1:?usage: restart_llama_grader.sh OLD_PID}
root=/Users/anshu/Desktop/bfb-llm-as-judge
status_log=$root/runs/nvidia-grades/llama-restart-status.log
seed=${2:-$(date +%s)}
source_traces=$root/data/raw/big-finance-benchmark/traces/gpt55.traces.jsonl
output_file=$root/runs/nvidia-grades/gpt55.grades.llama.repaired.jsonl
missing_traces=$root/runs/nvidia-grades/gpt55.traces.llama.missing.jsonl

# A previously approved invocation may carry an expired PID. Resolve the one live
# repaired-output Llama process, but refuse to proceed if overlapping graders exist.
if ! kill -0 "$old_pid" 2>/dev/null; then
  active_pids=(${(f)"$(pgrep -f 'run_nvidia_grader.py.*--judge llama.*gpt55.grades.llama.repaired.jsonl' || true)"})
  if (( ${#active_pids} > 1 )); then
    print -u2 "Refusing restart: multiple repaired-output Llama graders are active"
    exit 1
  fi
  if (( ${#active_pids} == 1 )); then
    old_pid=$active_pids[1]
  fi
fi
print "$(date '+%Y-%m-%d %H:%M:%S') restart requested for PID $old_pid with seed $seed" > "$status_log"

# Recover the already-exported credential from the user's existing grader process
# without printing or persisting it.
key=$(ps eww -p "$old_pid" -o command= 2>/dev/null | tr ' ' '\n' | sed -n 's/^NVIDIA_API_KEY=//p' | head -n 1)
if [[ -z "$key" ]]; then
  # The old grader may already have exited. Fall back to another same-user shell
  # where the user previously exported the credential (for example VS Code's shell).
  key=$(ps eww -ax -o command= | tr ' ' '\n' | sed -n 's/^NVIDIA_API_KEY=//p' | head -n 1)
fi
if [[ -z "$key" ]]; then
  print "credential inheritance failed" >> "$status_log"
  print -u2 "Could not inherit NVIDIA_API_KEY from PID $old_pid"
  exit 1
fi
print "credential inherited" >> "$status_log"
export NVIDIA_API_KEY="$key"
export BFB_JUDGE_REQUEST_TIMEOUT=600
unset key

kill "$old_pid" 2>/dev/null || true
cd "$root"
MISSING_TRACES="$missing_traces" SOURCE_TRACES="$source_traces" OUTPUT_FILE="$output_file" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["SOURCE_TRACES"])
output = Path(os.environ["OUTPUT_FILE"])
dest = Path(os.environ["MISSING_TRACES"])

have = set()
if output.exists():
    for line in output.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            have.add((obj["question_id"], obj["trial_idx"]))

missing_lines = []
for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    if (obj["question_id"], obj["trial_idx"]) not in have:
        missing_lines.append(line)

dest.write_text("\n".join(missing_lines) + ("\n" if missing_lines else ""), encoding="utf-8")
print(f"missing traces written: {len(missing_lines)}")
PY
missing_count=$(wc -l < "$missing_traces" | tr -d ' ')
print "missing trace count $missing_count" >> "$status_log"

nohup .venv/bin/python scripts/run_nvidia_grader.py \
  --judge llama \
  --traces "$missing_traces" \
  --sample-n "$missing_count" \
  --sample-seed "$seed" \
  --output "$output_file" \
  > runs/nvidia-grades/llama-repair.log 2>&1 &
pid=$!
disown
print "LLAMA_RESTART_PID=$pid"
print "grader launched as PID $pid" >> "$status_log"
nohup caffeinate -i -w "$pid" >/dev/null 2>&1 &
disown
print "caffeinate attached" >> "$status_log"
