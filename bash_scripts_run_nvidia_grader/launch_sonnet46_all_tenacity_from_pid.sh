#!/usr/bin/env bash
set -euo pipefail

pid="${1:?usage: launch_sonnet46_all_tenacity_from_pid.sh PID_WITH_NVIDIA_API_KEY}"
root="${BFB_ROOT:-/Users/anshu/Desktop/bfb-llm-as-judge}"
cd "$root"

key="$(
  ps eww -p "$pid" -o command= 2>/dev/null \
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

common_env=(
  PYTHONUNBUFFERED=1
  FORCE_RESTART=1
  BFB_JUDGE_REQUEST_TIMEOUT=180
  BFB_JUDGE_HARD_TIMEOUT=300
)

env "${common_env[@]}" \
  CONCURRENCY=5 \
  PROVIDER_CAP=5 \
  TENACITY_RETRY_ATTEMPTS=5 \
  TENACITY_RETRY_INITIAL_SECONDS=30 \
  TENACITY_RETRY_MAX_SECONDS=300 \
  bash_scripts_run_nvidia_grader/start_llama_tenacity_parallel.sh sonnet46

env "${common_env[@]}" \
  CONCURRENCY=5 \
  PROVIDER_CAP=5 \
  TENACITY_RETRY_ATTEMPTS=5 \
  TENACITY_RETRY_INITIAL_SECONDS=30 \
  TENACITY_RETRY_MAX_SECONDS=300 \
  bash_scripts_run_nvidia_grader/start_qwen_tenacity_parallel.sh sonnet46

env "${common_env[@]}" \
  CONCURRENCY=5 \
  PROVIDER_CAP=5 \
  TENACITY_RETRY_ATTEMPTS=5 \
  TENACITY_RETRY_INITIAL_SECONDS=30 \
  TENACITY_RETRY_MAX_SECONDS=300 \
  bash_scripts_run_nvidia_grader/start_nemotron_tenacity_parallel.sh sonnet46

env "${common_env[@]}" \
  CONCURRENCY=1 \
  PROVIDER_CAP=1 \
  TENACITY_RETRY_ATTEMPTS=10 \
  TENACITY_RETRY_INITIAL_SECONDS=120 \
  TENACITY_RETRY_MAX_SECONDS=900 \
  bash_scripts_run_nvidia_grader/start_mistral_tenacity_parallel.sh sonnet46

echo "launched sonnet46 tenacity graders"
