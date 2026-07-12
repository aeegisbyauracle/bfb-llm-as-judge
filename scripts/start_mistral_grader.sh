#!/bin/zsh
set -eu

root=/Users/anshu/Desktop/bfb-llm-as-judge
cd "$root"

output=runs/nvidia-grades/gpt55.grades.mistral.jsonl
log=runs/nvidia-grades/mistral.log
status_log=runs/nvidia-grades/mistral-status.log

live=($(pgrep -f "run_nvidia_grader.py.*gpt55.grades.mistral.jsonl" || true))
if (( ${#live[@]} )); then
  now=$(date +%s)
  modified=$(stat -f %m "$output" 2>/dev/null || print "$now")
  age=$((now - modified))
  if (( age < 3600 )); then
    print "Mistral grader already running; refusing duplicate" > "$status_log"
    exit 0
  fi

  print "$(date '+%F %T') stopping stale grader PID(s): ${live[*]}" > "$status_log"
  kill -TERM "${live[@]}"
  for _ in {1..20}; do
    sleep 1
    remaining=($(pgrep -f "run_nvidia_grader.py.*gpt55.grades.mistral.jsonl" || true))
    (( ${#remaining[@]} == 0 )) && break
  done
  if (( ${#remaining[@]} )); then
    print "stale grader did not exit; refusing overlapping restart" >> "$status_log"
    exit 1
  fi
fi

nohup env BFB_JUDGE_REQUEST_TIMEOUT=120 BFB_JUDGE_NUM_RETRIES=0 \
  BFB_JUDGE_HARD_TIMEOUT=180 \
  .venv/bin/python scripts/run_nvidia_grader.py \
  --judge mistral \
  --traces data/raw/big-finance-benchmark/traces/gpt55.traces.jsonl \
  --sample-n 150 \
  --sample-seed 31115 \
  --concurrency 1 \
  --question-delay-seconds 600 \
  --output "$output" \
  > "$log" 2>&1 &
pid=$!
print "$(date '+%F %T') grader PID $pid" > "$status_log"
caffeinate -i -w "$pid" >/dev/null 2>&1 &
print "caffeinate attached" >> "$status_log"
