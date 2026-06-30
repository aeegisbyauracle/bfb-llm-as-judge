#!/usr/bin/env python3
"""Get an inference response from NVIDIA's OpenAI-compatible API."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
DEFAULT_PROMPT = "How many r's are in the word 'strawberry'?"
DEFAULT_MAX_TOKENS = 65536
DEFAULT_REASONING_BUDGET = 16384
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95


def build_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Set NVIDIA_API_KEY before running this script.")

    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def request_completion(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    enable_thinking: bool,
    reasoning_budget: int,
) -> Any:
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "reasoning_budget": reasoning_budget,
    }
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=extra_body,
        stream=False,
    )
    return response.choices[0].message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call NVIDIA's OpenAI-compatible inference API."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NVIDIA model id.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Top-p.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum output tokens.",
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=DEFAULT_REASONING_BUDGET,
        help="Reasoning token budget sent in extra_body.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable NVIDIA thinking mode in chat_template_kwargs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        client = build_client()
        message = request_completion(
            client=client,
            model=args.model,
            prompt=args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            enable_thinking=not args.disable_thinking,
            reasoning_budget=args.reasoning_budget,
        )

        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            print("Reasoning:")
            print(reasoning)
            print()

        print("Response:")
        print(message.content)
        return 0
    except Exception as exc:
        print(f"NVIDIA request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
