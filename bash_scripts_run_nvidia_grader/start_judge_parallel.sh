#!/usr/bin/env bash
set -euo pipefail

judge="${1:?usage: start_judge_parallel.sh JUDGE_ALIAS [TRACE_LABEL|TRACE_PATH]}"
trace_arg="${2:-${TRACE_LABEL:-${TRACE:-gpt55}}}"

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
  mkdir -p runs/nvidia-grades
  echo "NVIDIA_API_KEY is not exported in this Terminal session" > "runs/nvidia-grades/${judge}-parallel-status.log"
  exit 1
fi

if [[ "$trace_arg" == */* || "$trace_arg" == *.jsonl ]]; then
  traces="$trace_arg"
  trace_file="$(basename "$traces")"
  trace_label="${TRACE_LABEL:-${trace_file%.traces.jsonl}}"
  trace_label="${trace_label%.jsonl}"
else
  trace_label="$trace_arg"
  traces="data/raw/big-finance-benchmark/traces/${trace_label}.traces.jsonl"
fi

if [[ ! -f "$traces" ]]; then
  mkdir -p runs/nvidia-grades
  echo "Trace file not found: ${traces}" > "runs/nvidia-grades/${judge}-${trace_label}-parallel-status.log"
  exit 1
fi

sample_n="${SAMPLE_N:-150}"
sample_seed="${SAMPLE_SEED:-0}"
concurrency="${CONCURRENCY:-50}"
request_timeout="${BFB_JUDGE_REQUEST_TIMEOUT:-600}"
num_retries="${BFB_JUDGE_NUM_RETRIES:-1}"
hard_timeout="${BFB_JUDGE_HARD_TIMEOUT:-900}"

output="${OUTPUT:-runs/nvidia-grades/${trace_label}.grades.${judge}.jsonl}"
log="${LOG:-runs/nvidia-grades/${trace_label}.${judge}-parallel.log}"
status_log="${STATUS_LOG:-runs/nvidia-grades/${trace_label}.${judge}-parallel-status.log}"

mkdir -p "$(dirname "$output")"

live=()
while IFS= read -r pid; do
  [[ -n "$pid" ]] && live+=("$pid")
done < <(pgrep -f "run_nvidia_grader.py.*--judge ${judge}.*${output}" || true)

if (( ${#live[@]} )); then
  if [[ "${FORCE_RESTART:-0}" == "1" ]]; then
    echo "$(date '+%F %T') stopping existing ${judge} grader PID(s): ${live[*]}" > "$status_log"
    kill -TERM "${live[@]}" 2>/dev/null || true
    for _ in {1..20}; do
      sleep 1
      remaining=()
      while IFS= read -r pid; do
        [[ -n "$pid" ]] && remaining+=("$pid")
      done < <(pgrep -f "run_nvidia_grader.py.*--judge ${judge}.*${output}" || true)
      (( ${#remaining[@]} == 0 )) && break
    done
    if (( ${#remaining[@]} )); then
      echo "existing ${judge} grader did not exit; refusing overlapping restart" >> "$status_log"
      exit 1
    fi
  else
    echo "${judge} grader already running for ${output}; set FORCE_RESTART=1 to replace it" > "$status_log"
    exit 0
  fi
fi

nohup env \
  BFB_JUDGE_REQUEST_TIMEOUT="$request_timeout" \
  BFB_JUDGE_NUM_RETRIES="$num_retries" \
  BFB_JUDGE_HARD_TIMEOUT="$hard_timeout" \
  .venv/bin/python scripts/run_nvidia_grader.py \
  --judge "$judge" \
  --traces "$traces" \
  --sample-n "$sample_n" \
  --sample-seed "$sample_seed" \
  --concurrency "$concurrency" \
  --output "$output" \
  > "$log" 2>&1 &

pid=$!
echo "$(date '+%F %T') ${judge} grader PID ${pid}" > "$status_log"
echo "trace_label=${trace_label}" >> "$status_log"
echo "traces=${traces}" >> "$status_log"
echo "output=${output}" >> "$status_log"
echo "sample_n=${sample_n} sample_seed=${sample_seed} concurrency=${concurrency}" >> "$status_log"

caffeinate -i -w "$pid" >/dev/null 2>&1 &
echo "caffeinate attached" >> "$status_log"
