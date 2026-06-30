#!/usr/bin/env python3
"""Get an inference response from OpenRouter with optional reasoning carryover."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "openai/gpt-oss-20b:free"
DEFAULT_PROMPT = "How many r's are in the word 'strawberry'?"
DEFAULT_FOLLOW_UP = "Are you sure? Think carefully."


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before running this script.")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def message_to_dict(message: Any) -> dict[str, Any]:
    """Convert an OpenAI SDK message object into an OpenRouter-compatible dict."""
    output = {
        "role": "assistant",
        "content": message.content,
    }

    reasoning_details = getattr(message, "reasoning_details", None)
    if reasoning_details is not None:
        output["reasoning_details"] = reasoning_details

    return output


def request_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    reasoning_enabled: bool,
) -> Any:
    extra_body = {"reasoning": {"enabled": True}} if reasoning_enabled else None
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body=extra_body,
    )
    return response.choices[0].message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call OpenRouter using the OpenAI Python SDK."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Initial user prompt.")
    parser.add_argument(
        "--follow-up",
        default=DEFAULT_FOLLOW_UP,
        help="Optional follow-up prompt. Pass an empty string to skip it.",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable OpenRouter reasoning in extra_body.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reasoning_enabled = not args.no_reasoning

    try:
        client = build_client()
        first_messages = [{"role": "user", "content": args.prompt}]
        first_response = request_completion(
            client=client,
            model=args.model,
            messages=first_messages,
            reasoning_enabled=reasoning_enabled,
        )

        print("First response:")
        print(first_response.content)

        if not args.follow_up:
            return 0

        follow_up_messages = [
            {"role": "user", "content": args.prompt},
            message_to_dict(first_response),
            {"role": "user", "content": args.follow_up},
        ]
        second_response = request_completion(
            client=client,
            model=args.model,
            messages=follow_up_messages,
            reasoning_enabled=reasoning_enabled,
        )

        print()
        print("Follow-up response:")
        print(second_response.content)
        return 0
    except Exception as exc:
        print(f"OpenRouter request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
