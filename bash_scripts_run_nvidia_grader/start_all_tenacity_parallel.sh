#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/start_llama_tenacity_parallel.sh" "$@"
"$script_dir/start_qwen_tenacity_parallel.sh" "$@"
"$script_dir/start_mistral_tenacity_parallel.sh" "$@"
"$script_dir/start_nemotron_tenacity_parallel.sh" "$@"
