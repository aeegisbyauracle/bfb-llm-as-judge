import pytest

from big_finance_harness.models.base import _nvidia_call_kwargs, _to_litellm_model, parse_model_id


def test_accepts_dated_anthropic_snapshot():
    assert parse_model_id("anthropic:claude-opus-4-7-20260416") == (
        "anthropic",
        "claude-opus-4-7-20260416",
    )


def test_accepts_dated_openai_snapshot():
    assert parse_model_id("openai:gpt-5.2-2026-01-15") == ("openai", "gpt-5.2-2026-01-15")


def test_warns_on_floating_alias():
    from big_finance_harness.models.base import FloatingAliasWarning

    with pytest.warns(FloatingAliasWarning, match="no date suffix"):
        provider, snapshot = parse_model_id("anthropic:claude-opus-4-7")
    assert provider == "anthropic"
    assert snapshot == "claude-opus-4-7"


def test_accepts_preview_alias_with_warning():
    from big_finance_harness.models.base import FloatingAliasWarning

    with pytest.warns(FloatingAliasWarning):
        provider, snapshot = parse_model_id("google:gemini-3.1-pro-preview")
    assert snapshot == "gemini-3.1-pro-preview"


def test_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unsupported provider"):
        parse_model_id("cohere:command-r-2026-01-01")


def test_rejects_missing_colon():
    with pytest.raises(ValueError, match="provider:snapshot"):
        parse_model_id("claude-opus-4-7-20260416")


def test_accepts_nvidia_model(monkeypatch):
    with pytest.warns(UserWarning, match="no date suffix"):
        provider, snapshot = parse_model_id("nvidia:meta/llama-3.3-70b-instruct")
    assert provider == "nvidia"
    assert snapshot == "meta/llama-3.3-70b-instruct"
    assert _to_litellm_model(provider, snapshot) == "nvidia_nim/meta/llama-3.3-70b-instruct"


def test_accepts_tinker_model():
    with pytest.warns(UserWarning, match="no date suffix"):
        provider, snapshot = parse_model_id("tinker:openai/gpt-oss-20b")
    assert provider == "tinker"
    assert snapshot == "openai/gpt-oss-20b"
    assert _to_litellm_model(provider, snapshot) == "openai/openai/gpt-oss-20b"


def test_nvidia_credentials_use_existing_environment_name(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "  test-key\n")
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    assert _nvidia_call_kwargs() == {
        "api_key": "test-key",
        "api_base": "https://integrate.api.nvidia.com/v1",
    }
