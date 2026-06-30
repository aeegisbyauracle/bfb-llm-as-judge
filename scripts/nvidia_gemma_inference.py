#!/usr/bin/env python3
"""Get a Gemma inference response from NVIDIA's build API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it"
DEFAULT_PROMPT = "How many r's are in the word 'strawberry'?"
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TIMEOUT = 120


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": args.stream,
        "chat_template_kwargs": {"enable_thinking": not args.disable_thinking},
    }


def print_stream(response: requests.Response) -> None:
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue

        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break

        chunk = json.loads(data)
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if content:
            print(content, end="", flush=True)
    print()


def print_completion(response: requests.Response) -> None:
    body = response.json()
    message = body["choices"][0]["message"]

    reasoning = message.get("reasoning_content")
    if reasoning:
        print("Reasoning:")
        print(reasoning)
        print()

    print("Response:")
    print(message.get("content", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Gemma 4 through NVIDIA's inference API."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NVIDIA model id.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum output tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Top-p.")
    parser.add_argument(
        "--stream", action="store_true", help="Stream response text as it arrives."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Maximum seconds to wait between response bytes.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable thinking through chat_template_kwargs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("Set NVIDIA_API_KEY before running this script.", file=sys.stderr)
        return 1

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream" if args.stream else "application/json",
    }

    try:
        response = requests.post(
            INVOKE_URL,
            headers=headers,
            json=build_payload(args),
            stream=args.stream,
            timeout=(30, args.timeout),
        )
        response.raise_for_status()

        if args.stream:
            print_stream(response)
        else:
            print_completion(response)
        return 0
    except requests.ReadTimeout:
        print(
            "NVIDIA returned no response data before the timeout. "
            "Retry with fewer tokens or --disable-thinking.",
            file=sys.stderr,
        )
        return 1
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"NVIDIA request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
