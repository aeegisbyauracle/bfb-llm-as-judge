# bfb-llm-as-judge

## Raw data

Raw datasets are not tracked in git. The full Hugging Face dataset should live at:

```text
data/raw/big-finance-benchmark/
```

To download it locally, run this from the repository root:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path

out = Path("data/raw/big-finance-benchmark")
out.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="RogoAI/big-finance-benchmark",
    repo_type="dataset",
    revision="main",
    local_dir=str(out),
)
PY
```

If `huggingface_hub` is not installed:

```bash
python3 -m pip install huggingface_hub
```

The expected download currently contains 34 files and is about 245 MB.

## NVIDIA judges

This repository includes the official BigFinanceBench harness with an additional
`nvidia:<model-id>` provider. NVIDIA judges receive BFB's unchanged system prompt,
question, reference answer, unweighted rubric, final answer, complete trace, and strict
JSON response schema.

Install the harness and export the one key shared by all NVIDIA NIM models:

```bash
python3 -m pip install -r requirements.txt
export NVIDIA_API_KEY="nvapi-your-key"
# Only needed for Tinker-hosted judge aliases:
export TINKER_API_KEY="tinker-your-key"
```

Available judge aliases:

| Alias | NVIDIA model |
| --- | --- |
| `llama` | `meta/llama-3.3-70b-instruct` |
| `qwen` | `qwen/qwen3-next-80b-a3b-instruct` |
| `mistral` | `mistralai/mistral-medium-3.5-128b` |
| `nemotron` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |
| `tinker-gpt-oss-20b` | `tinker:openai/gpt-oss-20b` |

Smoke-test one released trace with Llama:

```bash
python3 scripts/run_nvidia_grader.py \
  --judge llama \
  --traces data/raw/big-finance-benchmark/traces/gpt55.traces.jsonl \
  --sample-n 1
```

Run another judge by replacing `llama` with `qwen`, `mistral`, or `nemotron`. Results
are written under `runs/nvidia-grades/`. Runs resume automatically and use concurrency
`1` by default because NVIDIA's free endpoints can have long queues.

The original orchestrator also accepts NVIDIA judges directly:

```bash
python3 scripts/run_eval_set.py \
  --dataset data/big_finance_subset.jsonl \
  --run-id nvidia-judge-run \
  --sample-n 1 \
  --judge nvidia:meta/llama-3.3-70b-instruct
```
