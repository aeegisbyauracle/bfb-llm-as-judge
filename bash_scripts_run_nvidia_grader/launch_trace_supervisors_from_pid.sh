#!/usr/bin/env bash
set -euo pipefail

pid="${1:?usage: launch_trace_supervisors_from_pid.sh PID_WITH_NVIDIA_API_KEY TRACE_LABEL [JUDGE...]}"
trace_label="${2:?usage: launch_trace_supervisors_from_pid.sh PID_WITH_NVIDIA_API_KEY TRACE_LABEL [JUDGE...]}"
shift 2

root="${BFB_ROOT:-/Users/anshu/Desktop/bfb-llm-as-judge}"
cd "$root"

if (( $# )); then
  judges=("$@")
else
  judges=(llama qwen nemotron mistral)
fi

for judge in "${judges[@]}"; do
  log="runs/nvidia-grades/${trace_label}.${judge}-supervisor-wrapper.log"
  nohup bash_scripts_run_nvidia_grader/supervise_trace_judge_tenacity_from_pid.sh \
    "$pid" "$trace_label" "$judge" > "$log" 2>&1 &
  supervisor_pid=$!
  echo "$(date '+%F %T') ${trace_label}/${judge} supervisor PID ${supervisor_pid}" \
    > "runs/nvidia-grades/${trace_label}.${judge}-supervisor-wrapper-status.log"
  caffeinate -i -w "$supervisor_pid" >/dev/null 2>&1 &
done

echo "launched ${trace_label} supervisors for: ${judges[*]}"
