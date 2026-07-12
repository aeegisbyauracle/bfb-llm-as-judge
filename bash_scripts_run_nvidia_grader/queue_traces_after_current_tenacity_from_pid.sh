#!/usr/bin/env bash
set -euo pipefail

pid="${1:?usage: queue_traces_after_current_tenacity_from_pid.sh PID_WITH_NVIDIA_API_KEY CURRENT_TRACE [NEXT_TRACE...]}"
current_trace="${2:?usage: queue_traces_after_current_tenacity_from_pid.sh PID_WITH_NVIDIA_API_KEY CURRENT_TRACE [NEXT_TRACE...]}"
shift 2

root="${BFB_ROOT:-/Users/anshu/Desktop/bfb-llm-as-judge}"
cd "$root"

is_trace_running() {
  local trace_label="$1"
  pgrep -f "run_nvidia_grader_tenacity.py.*${trace_label}\\.traces\\.jsonl" >/dev/null
}

echo "$(date '+%F %T') waiting for active ${current_trace} jobs to finish"
while is_trace_running "$current_trace"; do
  sleep 300
done

if (( $# )); then
  exec bash_scripts_run_nvidia_grader/queue_all_traces_tenacity_from_pid.sh "$pid" "$@"
fi

remaining=()
while IFS= read -r path; do
  file="$(basename "$path")"
  trace_label="${file%.traces.jsonl}"
  [[ "$trace_label" == "$current_trace" ]] && continue
  remaining+=("$trace_label")
done < <(find data/raw/big-finance-benchmark/traces -maxdepth 1 -name '*.traces.jsonl' | sort)

exec bash_scripts_run_nvidia_grader/queue_all_traces_tenacity_from_pid.sh "$pid" "${remaining[@]}"
