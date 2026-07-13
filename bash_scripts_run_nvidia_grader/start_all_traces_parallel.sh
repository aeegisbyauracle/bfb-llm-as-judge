#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${BFB_ROOT:-/Users/anshu/Desktop/bfb-llm-as-judge}"

if (( $# )); then
  trace_labels=("$@")
else
  trace_labels=()
  while IFS= read -r path; do
    file="$(basename "$path")"
    trace_labels+=("${file%.traces.jsonl}")
  done < <(find "$root/data/raw/big-finance-benchmark/traces" -maxdepth 1 -name '*.traces.jsonl' | sort)
fi

for trace_label in "${trace_labels[@]}"; do
  "$script_dir/start_all_parallel.sh" "$trace_label"
done
