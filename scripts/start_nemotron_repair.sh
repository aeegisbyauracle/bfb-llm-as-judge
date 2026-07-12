#!/bin/zsh
set -eu

root=/Users/anshu/Desktop/bfb-llm-as-judge
cd "$root"

output=runs/nvidia-grades/gpt55.grades.nemotron.repair.jsonl
log=runs/nvidia-grades/nemotron-repair.log
status_log=runs/nvidia-grades/nemotron-repair-status.log

if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
  print "NVIDIA_API_KEY is not exported in this Terminal session" > "$status_log"
  exit 1
fi

live=($(pgrep -f "run_nvidia_grader.py.*gpt55.grades.nemotron.repair.jsonl" || true))
if (( ${#live[@]} )); then
  print "Nemotron repair already running; refusing duplicate" > "$status_log"
  exit 0
fi

nohup env BFB_JUDGE_REQUEST_TIMEOUT=180 BFB_JUDGE_NUM_RETRIES=1 \
  BFB_JUDGE_HARD_TIMEOUT=420 \
  .venv/bin/python scripts/run_nvidia_grader.py \
  --judge nemotron \
  --traces runs/nvidia-grades/gpt55.traces.nemotron.repair.jsonl \
  --sample-n 5 \
  --output "$output" \
  > "$log" 2>&1 &
pid=$!
print "$(date '+%F %T') repair PID $pid" > "$status_log"
caffeinate -i -w "$pid" >/dev/null 2>&1 &
print "caffeinate attached" >> "$status_log"
