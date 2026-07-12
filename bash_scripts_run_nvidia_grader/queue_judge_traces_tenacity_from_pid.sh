#!/usr/bin/env bash
set -euo pipefail

pid="${1:?usage: queue_judge_traces_tenacity_from_pid.sh PID_WITH_NVIDIA_API_KEY JUDGE TRACE...}"
judge="${2:?usage: queue_judge_traces_tenacity_from_pid.sh PID_WITH_NVIDIA_API_KEY JUDGE TRACE...}"
shift 2

root="${BFB_ROOT:-/Users/anshu/Desktop/bfb-llm-as-judge}"
cd "$root"

case "$judge" in
  llama|qwen|mistral|nemotron) ;;
  *)
    echo "Unknown judge '$judge'. Expected one of: llama, qwen, mistral, nemotron" >&2
    exit 2
    ;;
esac

if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
  key="$(
    (ps eww -p "$pid" -o command= 2>/dev/null || true) \
      | tr ' ' '\n' \
      | sed -n 's/^NVIDIA_API_KEY=//p' \
      | head -n 1
  )"
  if [[ -z "$key" ]]; then
    echo "Could not inherit NVIDIA_API_KEY from PID $pid" >&2
    exit 1
  fi
  export NVIDIA_API_KEY="$key"
  unset key
fi

if (( $# )); then
  traces=("$@")
else
  traces=()
  while IFS= read -r path; do
    file="$(basename "$path")"
    traces+=("${file%.traces.jsonl}")
  done < <(find data/raw/big-finance-benchmark/traces -maxdepth 1 -name '*.traces.jsonl' | sort)
fi

log="runs/nvidia-grades/${judge}-lane-queue.log"
status_log="runs/nvidia-grades/${judge}-lane-queue-status.log"
echo "$(date '+%F %T') queue ${judge}: ${traces[*]}" > "$status_log"

for trace_label in "${traces[@]}"; do
  echo "$(date '+%F %T') starting ${trace_label}/${judge}" >> "$status_log"
  bash_scripts_run_nvidia_grader/supervise_trace_judge_tenacity_from_pid.sh \
    0 "$trace_label" "$judge" >> "$log" 2>&1
  echo "$(date '+%F %T') finished ${trace_label}/${judge}" >> "$status_log"
done

echo "$(date '+%F %T') ${judge} queue complete" >> "$status_log"
