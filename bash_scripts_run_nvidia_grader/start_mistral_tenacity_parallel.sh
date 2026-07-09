#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trace_arg="${1:-${TRACE_LABEL:-${TRACE:-gpt55}}}"
if [[ -z "${SAMPLE_SEED:-}" && "$trace_arg" == "gpt55" ]]; then
  export SAMPLE_SEED=31115
fi
exec "$script_dir/start_judge_tenacity_parallel.sh" mistral "$@"
