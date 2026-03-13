"""Tests for LiteLLM wrapper."""

from unittest.mock import MagicMock, patch

from src.shared.llm import LLMClient


def test_llm_client_complete():
    with patch("src.shared.llm.litellm.completion") as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = MagicMock(total_tokens=50)
        mock_completion.return_value = mock_response

        client = LLMClient(model="gpt-4o-mini")
        result = client.complete(system_prompt="You are helpful.", user_prompt="Hello")

    assert result == "Test response"
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert len(call_kwargs["messages"]) == 2


def test_llm_client_uses_configured_model():
    with patch("src.shared.llm.litellm.completion") as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_response.usage = MagicMock(total_tokens=10)
        mock_completion.return_value = mock_response

        client = LLMClient(model="claude-sonnet-4-20250514")
        client.complete(system_prompt="sys", user_prompt="usr")

    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"


def test_llm_client_default_model():
    client = LLMClient()
    assert client.model == "gpt-4o-mini"
