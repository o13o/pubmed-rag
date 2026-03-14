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


def test_llm_client_complete_stream():
    with patch("src.shared.llm.litellm.completion") as mock_completion:
        # Simulate streaming chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]
        chunk2.usage = None

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock(delta=MagicMock(content=None))]
        chunk3.usage = MagicMock(total_tokens=25)

        mock_completion.return_value = iter([chunk1, chunk2, chunk3])

        client = LLMClient(model="gpt-4o-mini")
        chunks = list(client.complete_stream(
            system_prompt="You are helpful.",
            user_prompt="Hello",
        ))

    assert chunks == ["Hello", " world"]
    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["stream"] is True
    assert call_kwargs["stream_options"] == {"include_usage": True}


def test_llm_client_complete_stream_handles_empty_choices():
    with patch("src.shared.llm.litellm.completion") as mock_completion:
        chunk_empty = MagicMock()
        chunk_empty.choices = []
        chunk_empty.usage = None

        chunk_normal = MagicMock()
        chunk_normal.choices = [MagicMock(delta=MagicMock(content="ok"))]
        chunk_normal.usage = None

        mock_completion.return_value = iter([chunk_empty, chunk_normal])

        client = LLMClient()
        chunks = list(client.complete_stream(
            system_prompt="sys", user_prompt="usr",
        ))

    assert chunks == ["ok"]
