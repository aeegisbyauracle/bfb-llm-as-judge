# Parallel NVIDIA grader launchers

These scripts start `scripts/run_nvidia_grader.py` with `CONCURRENCY=50` by default.
They resume from the existing grade JSONL files under `runs/nvidia-grades/`.
They can grade any trace file under `data/raw/big-finance-benchmark/traces/`.

Run one judge:

```bash
export NVIDIA_API_KEY="nvapi-your-key"
bash_scripts_run_nvidia_grader/start_mistral_parallel.sh
```

Run one judge for a specific base-model trace:

```bash
bash_scripts_run_nvidia_grader/start_mistral_parallel.sh sonnet46
```

That reads:

```text
data/raw/big-finance-benchmark/traces/sonnet46.traces.jsonl
```

and writes:

```text
runs/nvidia-grades/sonnet46.grades.mistral.jsonl
```

Run all four judges for one base-model trace:

```bash
export NVIDIA_API_KEY="nvapi-your-key"
bash_scripts_run_nvidia_grader/start_all_parallel.sh gpt55
```

Run all four judges for every available trace file:

```bash
export NVIDIA_API_KEY="nvapi-your-key"
bash_scripts_run_nvidia_grader/start_all_traces_parallel.sh
```

Useful overrides:

```bash
CONCURRENCY=25 bash_scripts_run_nvidia_grader/start_mistral_parallel.sh sonnet46
SAMPLE_SEED=31115 bash_scripts_run_nvidia_grader/start_mistral_parallel.sh gpt55
FORCE_RESTART=1 bash_scripts_run_nvidia_grader/start_mistral_parallel.sh gpt55
TRACES=/path/to/custom.traces.jsonl bash_scripts_run_nvidia_grader/start_mistral_parallel.sh
```

Logs and status files include the trace label, for example
`runs/nvidia-grades/sonnet46.mistral-parallel.log` and
`runs/nvidia-grades/sonnet46.mistral-parallel-status.log`.

## Tenacity retry launchers

There is also a standalone Python runner at:

```text
scripts/run_nvidia_grader_tenacity.py
```

It does not modify the original runner or grader files. It wraps each judge call in
Tenacity retries, defaults LiteLLM internal retries to `0`, and overrides the in-process
NVIDIA semaphore cap for that run so `CONCURRENCY=50` can actually fan out.

Restart Mistral on `gpt55` with the Tenacity runner:

```bash
FORCE_RESTART=1 CONCURRENCY=50 bash_scripts_run_nvidia_grader/start_mistral_tenacity_parallel.sh gpt55
```

Run Nemotron on `sonnet46` with Tenacity:

```bash
bash_scripts_run_nvidia_grader/start_nemotron_tenacity_parallel.sh sonnet46
```

Useful Tenacity overrides:

```bash
BFB_JUDGE_HARD_TIMEOUT=180 TENACITY_RETRY_ATTEMPTS=4 bash_scripts_run_nvidia_grader/start_mistral_tenacity_parallel.sh gpt55
PROVIDER_CAP=25 CONCURRENCY=25 bash_scripts_run_nvidia_grader/start_mistral_tenacity_parallel.sh gpt55
```
