import json

from big_finance_harness.types import (
    Message,
    RunRecord,
    StepRecord,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def test_run_record_accepts_public_release_without_new_metadata_fields():
    payload = {
        "question_id": "bf-legacy",
        "question": "Question",
        "reference_answer": "Answer",
        "model": "gpt-5.5",
        "harness_version": "0.1.0",
        "steps": [],
        "final_answer": "Answer",
        "stop_reason": "final_answer",
        "total_prompt_tokens": 1,
        "total_completion_tokens": 1,
        "total_wallclock_seconds": 1.0,
    }

    record = RunRecord.model_validate(payload)

    assert record.thinking == "off"
    assert record.max_steps == 0
    assert record.started_at == ""
    assert record.completed_at == ""


def test_message_block_discriminator_roundtrip():
    msg = Message(
        role="assistant",
        content=[
            TextBlock(text="hi"),
            ToolUseBlock(id="t1", name="web_search", input={"query": "x"}),
        ],
    )
    serialized = msg.model_dump_json()
    restored = Message.model_validate_json(serialized)
    assert isinstance(restored.content[0], TextBlock)
    assert isinstance(restored.content[1], ToolUseBlock)
    assert restored.content[1].name == "web_search"


def test_tool_result_block_roundtrip():
    msg = Message(
        role="tool",
        content=[ToolResultBlock(tool_use_id="t1", content="ok", is_error=False)],
    )
    raw = json.loads(msg.model_dump_json())
    assert raw["content"][0]["type"] == "tool_result"
    Message.model_validate(raw)


def test_run_record_serializable():
    record = RunRecord(
        question_id="q1",
        question="What?",
        reference_answer="42",
        model="anthropic:claude-opus-4-7-20260416",
        harness_version="0.1.0",
        thinking="off",
        temperature=0.0,
        max_steps=30,
        steps=[
            StepRecord(
                step=0,
                assistant_text="thinking…",
                tool_calls=[ToolUseBlock(id="c1", name="final_answer", input={"answer": "42"})],
                tool_results=[ToolResultBlock(tool_use_id="c1", content="42")],
                prompt_tokens=10,
                completion_tokens=5,
                wallclock_seconds=0.5,
            )
        ],
        final_answer="42",
        stop_reason="final_answer",
        total_prompt_tokens=10,
        total_completion_tokens=5,
        total_wallclock_seconds=0.5,
        cost_usd=0.0,
        started_at="2026-04-30T00:00:00+00:00",
        completed_at="2026-04-30T00:00:01+00:00",
    )
    raw = record.model_dump_json()
    restored = RunRecord.model_validate_json(raw)
    assert restored.final_answer == "42"
