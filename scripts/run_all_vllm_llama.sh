#!/usr/bin/env bash
set -e

export VLLM_API_KEY="test123"

TRACES=(
    gem3flash
    gemma4-31b
    glm-51
    gpt54mini
    gpt55
    kimi-k26
    opus47
    qwen36-27b
    sonnet46
)

for t in "${TRACES[@]}"; do
    echo "========================================"
    echo "Starting: $t"
    echo "========================================"
    python3 scripts/nvidia_grade_concurrent.py \
        --traces "data/raw/traces/${t}.traces.jsonl" \
        --judge vllm-llama \
        --concurrency 1 \
        --delay 1.0
    echo "Finished: $t"
    echo ""
done

echo "All done."
