#!/usr/bin/env bash
set -euo pipefail

pid="${1:?usage: queue_judge_after_active_tenacity_from_pid.sh PID_WITH_NVIDIA_API_KEY JUDGE TRACE...}"
judge="${2:?usage: queue_judge_after_active_tenacity_from_pid.sh PID_WITH_NVIDIA_API_KEY JUDGE TRACE...}"
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

status_log="runs/nvidia-grades/${judge}-after-active-queue-status.log"
echo "$(date '+%F %T') waiting for active ${judge} lane" > "$status_log"

while pgrep -f "supervise_trace_judge_tenacity_from_pid.sh .* ${judge}$|run_nvidia_grader_tenacity.py --judge ${judge} " >/dev/null; do
  sleep 300
done

echo "$(date '+%F %T') ${judge} lane free; launching queued traces: $*" >> "$status_log"
exec bash_scripts_run_nvidia_grader/queue_judge_traces_tenacity_from_pid.sh 0 "$judge" "$@"
