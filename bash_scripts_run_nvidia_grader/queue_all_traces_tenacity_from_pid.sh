#!/usr/bin/env bash
set -euo pipefail

pid="${1:?usage: queue_all_traces_tenacity_from_pid.sh PID_WITH_NVIDIA_API_KEY [TRACE_LABEL...]}"
shift

root="${BFB_ROOT:-/Users/anshu/Desktop/bfb-llm-as-judge}"
cd "$root"

if (( $# )); then
  traces=("$@")
else
  traces=()
  while IFS= read -r path; do
    file="$(basename "$path")"
    traces+=("${file%.traces.jsonl}")
  done < <(find data/raw/big-finance-benchmark/traces -maxdepth 1 -name '*.traces.jsonl' | sort)
fi

is_trace_running() {
  local trace_label="$1"
  pgrep -f "run_nvidia_grader_tenacity.py.*${trace_label}\\.traces\\.jsonl" >/dev/null
}

for trace_label in "${traces[@]}"; do
  echo "$(date '+%F %T') waiting for any active ${trace_label} jobs to finish"
  while is_trace_running "$trace_label"; do
    sleep 300
  done

  echo "$(date '+%F %T') launching ${trace_label}"
  bash_scripts_run_nvidia_grader/launch_trace_all_tenacity_from_pid.sh "$pid" "$trace_label"

  echo "$(date '+%F %T') waiting for ${trace_label} jobs to finish"
  while is_trace_running "$trace_label"; do
    sleep 300
  done
done

echo "$(date '+%F %T') queued traces complete"
