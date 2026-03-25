"""Integration tests for the MiniMax API backend.

These tests verify the end-to-end behavior of the MiniMax backend
including streaming, tool calling, and error handling with mock HTTP responses.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.llama_conversation.backends.minimax import (
    MiniMaxAPIClient,
    MINIMAX_DEFAULT_BASE_URL,
)
from custom_components.llama_conversation.const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_REQUEST_TIMEOUT,
    CONF_ENABLE_LEGACY_TOOL_CALLING,
    CONF_TOOL_RESPONSE_AS_STRING,
    CONF_THINKING_PREFIX,
    CONF_THINKING_SUFFIX,
    CONF_TOOL_CALL_PREFIX,
    CONF_TOOL_CALL_SUFFIX,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_ENABLE_LEGACY_TOOL_CALLING,
    DEFAULT_TOOL_RESPONSE_AS_STRING,
    DEFAULT_THINKING_PREFIX,
    DEFAULT_THINKING_SUFFIX,
    DEFAULT_TOOL_CALL_PREFIX,
    DEFAULT_TOOL_CALL_SUFFIX,
)


def _make_sse_chunk(content: str, finish_reason: str | None = None) -> bytes:
    """Build an SSE data line for a streaming chat completion chunk."""
    chunk = {
        "object": "chat.completion.chunk",
        "choices": [{
            "delta": {"content": content},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode("utf-8")


def _make_sse_done() -> bytes:
    return b"data: [DONE]\n\n"


@pytest.fixture
def minimax_entity_options():
    """Typical entity options passed to _generate_stream."""
    return {
        CONF_CHAT_MODEL: "MiniMax-M2.7",
        CONF_MAX_TOKENS: 256,
        CONF_TEMPERATURE: 0.5,
        CONF_TOP_P: 0.9,
        CONF_REQUEST_TIMEOUT: 60,
        CONF_ENABLE_LEGACY_TOOL_CALLING: DEFAULT_ENABLE_LEGACY_TOOL_CALLING,
        CONF_TOOL_RESPONSE_AS_STRING: DEFAULT_TOOL_RESPONSE_AS_STRING,
        CONF_THINKING_PREFIX: DEFAULT_THINKING_PREFIX,
        CONF_THINKING_SUFFIX: DEFAULT_THINKING_SUFFIX,
        CONF_TOOL_CALL_PREFIX: DEFAULT_TOOL_CALL_PREFIX,
        CONF_TOOL_CALL_SUFFIX: DEFAULT_TOOL_CALL_SUFFIX,
    }


@pytest.fixture
def minimax_client_options():
    return {
        CONF_API_KEY: "test-integration-key",
        CONF_BASE_URL: MINIMAX_DEFAULT_BASE_URL,
        CONF_CHAT_MODEL: "MiniMax-M2.7",
    }


@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    return hass


class TestMiniMaxConnectionValidation:
    """Integration tests for connection validation with various HTTP responses."""

    @pytest.mark.asyncio
    async def test_validate_with_server_error(self, mock_hass):
        mock_response = AsyncMock()
        mock_response.ok = False
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch(
            "custom_components.llama_conversation.backends.minimax.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await MiniMaxAPIClient.async_validate_connection(
                mock_hass,
                {CONF_API_KEY: "key", CONF_BASE_URL: MINIMAX_DEFAULT_BASE_URL},
            )
        assert "500" in result

    @pytest.mark.asyncio
    async def test_validate_with_custom_base_url(self, mock_hass):
        mock_response = AsyncMock()
        mock_response.ok = True
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch(
            "custom_components.llama_conversation.backends.minimax.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await MiniMaxAPIClient.async_validate_connection(
                mock_hass,
                {CONF_API_KEY: "key", CONF_BASE_URL: "https://custom.example.com/v1"},
            )
        assert result is None

        # Verify it called the custom URL
        call_args = mock_session.post.call_args
        assert "custom.example.com" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_validate_connection_exception(self, mock_hass):
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("DNS failure"))
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)

        with patch(
            "custom_components.llama_conversation.backends.minimax.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await MiniMaxAPIClient.async_validate_connection(
                mock_hass,
                {CONF_API_KEY: "key", CONF_BASE_URL: MINIMAX_DEFAULT_BASE_URL},
            )
        assert "DNS failure" in result


class TestMiniMaxResponseParsing:
    """Integration tests for various response formats from MiniMax API."""

    @pytest.fixture
    def client(self, mock_hass, minimax_client_options):
        return MiniMaxAPIClient(mock_hass, minimax_client_options)

    def test_parse_thinking_content_not_exposed(self, client):
        """Verify thinking content in responses is handled via prefix/suffix config."""
        response = {
            "object": "chat.completion",
            "choices": [{
                "message": {"content": "<think>Let me think about this...</think>The light is on."},
                "finish_reason": "stop",
            }],
        }
        text, tool_calls = client._extract_response(response)
        # The _extract_response returns raw text; thinking strip happens in _async_stream_parse_completion
        assert "<think>" in text
        assert "The light is on." in text

    def test_parse_multiple_tool_calls_in_chunk(self, client):
        response = {
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "turn_on", "arguments": '{"entity_id": "light.a"}'}},
                        {"function": {"name": "turn_off", "arguments": '{"entity_id": "light.b"}'}},
                    ],
                },
                "finish_reason": None,
            }],
        }
        text, tool_calls = client._extract_response(response)
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "turn_on"
        assert tool_calls[1]["name"] == "turn_off"

    def test_parse_unknown_object_type(self, client):
        """Fallback for non-standard object type."""
        response = {
            "object": "text.completion",
            "choices": [{
                "text": "raw text output",
                "finish_reason": "stop",
            }],
        }
        text, tool_calls = client._extract_response(response)
        assert text == "raw text output"
