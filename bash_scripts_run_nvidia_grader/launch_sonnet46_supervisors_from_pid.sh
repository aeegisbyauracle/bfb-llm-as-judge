#!/usr/bin/env bash
set -euo pipefail

pid="${1:?usage: launch_sonnet46_supervisors_from_pid.sh PID_WITH_NVIDIA_API_KEY}"
root="${BFB_ROOT:-/Users/anshu/Desktop/bfb-llm-as-judge}"
cd "$root"

for judge in llama qwen mistral; do
  log="runs/nvidia-grades/sonnet46.${judge}-supervisor-wrapper.log"
  nohup bash_scripts_run_nvidia_grader/supervise_trace_judge_tenacity_from_pid.sh \
    "$pid" sonnet46 "$judge" > "$log" 2>&1 &
  supervisor_pid=$!
  echo "$(date '+%F %T') ${judge} supervisor PID ${supervisor_pid}" \
    > "runs/nvidia-grades/sonnet46.${judge}-supervisor-wrapper-status.log"
  caffeinate -i -w "$supervisor_pid" >/dev/null 2>&1 &
done

echo "launched sonnet46 supervisors"
