#!/usr/bin/env python3
"""Run Qwen3 Next 80B through NVIDIA's free API endpoint."""

import argparse
import os
import sys

from openai import OpenAI


MODEL = "qwen/qwen3-next-80b-a3b-instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Run {MODEL} on NVIDIA NIM.")
    parser.add_argument("--prompt", default="Write a short poem.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("Set NVIDIA_API_KEY before running this script.", file=sys.stderr)
        return 1

    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=180,
            max_retries=1,
        )
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": args.prompt}],
            temperature=0.6,
            top_p=0.7,
            max_tokens=args.max_tokens,
            stream=False,
        )
        print(completion.choices[0].message.content or "")
        return 0
    except Exception as exc:
        print(f"NVIDIA request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
