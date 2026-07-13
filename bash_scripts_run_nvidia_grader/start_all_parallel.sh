#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/start_llama_parallel.sh" "$@"
"$script_dir/start_qwen_parallel.sh" "$@"
"$script_dir/start_mistral_parallel.sh" "$@"
"$script_dir/start_nemotron_parallel.sh" "$@"
