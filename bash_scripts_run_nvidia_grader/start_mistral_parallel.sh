#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SAMPLE_SEED="${SAMPLE_SEED:-31115}"
exec "$script_dir/start_judge_parallel.sh" mistral "$@"
