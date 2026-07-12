#!/usr/bin/env bash
set -euo pipefail

pid="${1:?usage: supervise_trace_judge_tenacity_from_pid.sh PID_WITH_NVIDIA_API_KEY TRACE_LABEL JUDGE_ALIAS}"
trace_label="${2:?usage: supervise_trace_judge_tenacity_from_pid.sh PID_WITH_NVIDIA_API_KEY TRACE_LABEL JUDGE_ALIAS}"
judge="${3:?usage: supervise_trace_judge_tenacity_from_pid.sh PID_WITH_NVIDIA_API_KEY TRACE_LABEL JUDGE_ALIAS}"

root="${BFB_ROOT:-/Users/anshu/Desktop/bfb-llm-as-judge}"
cd "$root"

case "$judge" in
  llama|qwen|mistral|nemotron) ;;
  *)
    echo "Unknown judge '$judge'. Expected one of: llama, qwen, mistral, nemotron" >&2
    exit 2
    ;;
esac

key="${NVIDIA_API_KEY:-}"

if [[ -z "$key" ]]; then
  key="$(
    (ps eww -p "$pid" -o command= 2>/dev/null || true) \
      | tr ' ' '\n' \
      | sed -n 's/^NVIDIA_API_KEY=//p' \
      | head -n 1
  )"
fi

if [[ -z "$key" ]]; then
  key_file="${NVIDIA_API_KEY_FILE:-/private/tmp/bfb_nvidia_key_for_sonnet_supervisor}"
  if [[ -f "$key_file" ]]; then
    key="$(<"$key_file")"
  fi
fi

if [[ -z "$key" ]]; then
  echo "Could not inherit NVIDIA_API_KEY from PID $pid or key file" >&2
  exit 1
fi

export NVIDIA_API_KEY="$key"
unset key

target="${TARGET_GRADES:-150}"
sample_n="${SAMPLE_N:-150}"
sample_seed="${SAMPLE_SEED:-0}"
traces="data/raw/big-finance-benchmark/traces/${trace_label}.traces.jsonl"
output="runs/nvidia-grades/${trace_label}.grades.${judge}.jsonl"
log="runs/nvidia-grades/${trace_label}.${judge}-supervisor.log"
status_log="runs/nvidia-grades/${trace_label}.${judge}-supervisor-status.log"

case "$judge" in
  mistral)
    concurrency="${CONCURRENCY:-1}"
    retry_attempts="${TENACITY_RETRY_ATTEMPTS:-30}"
    retry_initial="${TENACITY_RETRY_INITIAL_SECONDS:-120}"
    retry_max="${TENACITY_RETRY_MAX_SECONDS:-900}"
    request_timeout="${BFB_JUDGE_REQUEST_TIMEOUT:-600}"
    hard_timeout="${BFB_JUDGE_HARD_TIMEOUT:-900}"
    ;;
  qwen)
    concurrency="${CONCURRENCY:-2}"
    retry_attempts="${TENACITY_RETRY_ATTEMPTS:-20}"
    retry_initial="${TENACITY_RETRY_INITIAL_SECONDS:-60}"
    retry_max="${TENACITY_RETRY_MAX_SECONDS:-600}"
    request_timeout="${BFB_JUDGE_REQUEST_TIMEOUT:-600}"
    hard_timeout="${BFB_JUDGE_HARD_TIMEOUT:-900}"
    ;;
  *)
    concurrency="${CONCURRENCY:-2}"
    retry_attempts="${TENACITY_RETRY_ATTEMPTS:-20}"
    retry_initial="${TENACITY_RETRY_INITIAL_SECONDS:-60}"
    retry_max="${TENACITY_RETRY_MAX_SECONDS:-600}"
    request_timeout="${BFB_JUDGE_REQUEST_TIMEOUT:-600}"
    hard_timeout="${BFB_JUDGE_HARD_TIMEOUT:-900}"
    ;;
esac

provider_cap="${PROVIDER_CAP:-$concurrency}"
mkdir -p "$(dirname "$output")"

count_grades() {
  if [[ -f "$output" ]]; then
    wc -l < "$output" | tr -d ' '
  else
    echo 0
  fi
}

echo "$(date '+%F %T') supervising ${trace_label}/${judge}" > "$status_log"
echo "output=${output}" >> "$status_log"
echo "target=${target} sample_n=${sample_n} sample_seed=${sample_seed}" >> "$status_log"
echo "concurrency=${concurrency} provider_cap=${provider_cap}" >> "$status_log"
echo "request_timeout=${request_timeout} hard_timeout=${hard_timeout}" >> "$status_log"
echo "retry_attempts=${retry_attempts} retry_initial=${retry_initial} retry_max=${retry_max}" >> "$status_log"

while true; do
  have="$(count_grades)"
  echo "$(date '+%F %T') ${trace_label}/${judge}: ${have}/${target}" >> "$status_log"
  if (( have >= target )); then
    echo "$(date '+%F %T') complete" >> "$status_log"
    exit 0
  fi

  {
    echo
    echo "===== $(date '+%F %T') starting pass; have ${have}/${target} ====="
  } >> "$log"

  set +e
  PYTHONUNBUFFERED=1 \
    BFB_JUDGE_REQUEST_TIMEOUT="$request_timeout" \
    BFB_JUDGE_NUM_RETRIES=0 \
    BFB_JUDGE_HARD_TIMEOUT="$hard_timeout" \
    .venv/bin/python scripts/run_nvidia_grader_tenacity.py \
      --judge "$judge" \
      --traces "$traces" \
      --sample-n "$sample_n" \
      --sample-seed "$sample_seed" \
      --concurrency "$concurrency" \
      --provider-cap "$provider_cap" \
      --retry-attempts "$retry_attempts" \
      --retry-initial-seconds "$retry_initial" \
      --retry-max-seconds "$retry_max" \
      --output "$output" \
      >> "$log" 2>&1
  rc=$?
  set -e

  echo "$(date '+%F %T') pass exited rc=${rc}" >> "$status_log"
  sleep "${SUPERVISOR_SLEEP_SECONDS:-60}"
done
