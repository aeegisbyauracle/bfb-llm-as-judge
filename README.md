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
