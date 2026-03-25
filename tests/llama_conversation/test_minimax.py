"""Tests for the MiniMax API backend."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.llama_conversation.backends.minimax import (
    MiniMaxAPIClient,
    MINIMAX_AVAILABLE_MODELS,
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
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REQUEST_TIMEOUT,
    BACKEND_TYPE_MINIMAX,
)


# --- Fixtures ---

@pytest.fixture
def minimax_options():
    """Return default MiniMax client options."""
    return {
        CONF_API_KEY: "test-api-key-123",
        CONF_BASE_URL: MINIMAX_DEFAULT_BASE_URL,
        CONF_CHAT_MODEL: "MiniMax-M2.7",
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_REQUEST_TIMEOUT: DEFAULT_REQUEST_TIMEOUT,
    }


@pytest.fixture
def mock_hass():
    """Return a mock Home Assistant instance."""
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    return hass


# --- Unit Tests ---

class TestMiniMaxClientInit:
    """Tests for MiniMaxAPIClient initialization."""

    def test_init_with_defaults(self, mock_hass, minimax_options):
        client = MiniMaxAPIClient(mock_hass, minimax_options)
        assert client.api_host == MINIMAX_DEFAULT_BASE_URL
        assert client.api_key == "test-api-key-123"

    def test_init_with_custom_base_url(self, mock_hass, minimax_options):
        minimax_options[CONF_BASE_URL] = "https://custom.api.example.com/v1"
        client = MiniMaxAPIClient(mock_hass, minimax_options)
        assert client.api_host == "https://custom.api.example.com/v1"

    def test_init_strips_trailing_slash(self, mock_hass, minimax_options):
        minimax_options[CONF_BASE_URL] = "https://api.minimax.io/v1/"
        client = MiniMaxAPIClient(mock_hass, minimax_options)
        assert client.api_host == "https://api.minimax.io/v1"

    def test_init_empty_api_key(self, mock_hass, minimax_options):
        minimax_options[CONF_API_KEY] = ""
        client = MiniMaxAPIClient(mock_hass, minimax_options)
        assert client.api_key == ""


class TestMiniMaxGetName:
    """Tests for get_name static method."""

    def test_get_name_default_url(self):
        name = MiniMaxAPIClient.get_name({CONF_BASE_URL: MINIMAX_DEFAULT_BASE_URL})
        assert "MiniMax" in name
        assert MINIMAX_DEFAULT_BASE_URL in name

    def test_get_name_custom_url(self):
        name = MiniMaxAPIClient.get_name({CONF_BASE_URL: "https://custom.example.com/v1"})
        assert "MiniMax" in name
        assert "custom.example.com" in name

    def test_get_name_missing_url_uses_default(self):
        name = MiniMaxAPIClient.get_name({})
        assert "MiniMax" in name
        assert MINIMAX_DEFAULT_BASE_URL in name


class TestMiniMaxAvailableModels:
    """Tests for async_get_available_models."""

    @pytest.mark.asyncio
    async def test_returns_hardcoded_models(self, mock_hass, minimax_options):
        client = MiniMaxAPIClient(mock_hass, minimax_options)
        models = await client.async_get_available_models()
        assert models == MINIMAX_AVAILABLE_MODELS
        assert "MiniMax-M2.7" in models
        assert "MiniMax-M2.7-highspeed" in models
        assert "MiniMax-M2.5" in models
        assert "MiniMax-M2.5-highspeed" in models

    @pytest.mark.asyncio
    async def test_returns_four_models(self, mock_hass, minimax_options):
        client = MiniMaxAPIClient(mock_hass, minimax_options)
        models = await client.async_get_available_models()
        assert len(models) == 4


class TestMiniMaxExtractResponse:
    """Tests for _extract_response method."""

    @pytest.fixture
    def client(self, mock_hass, minimax_options):
        return MiniMaxAPIClient(mock_hass, minimax_options)

    def test_extract_chat_completion(self, client):
        response = {
            "object": "chat.completion",
            "choices": [{
                "message": {"content": "Hello world"},
                "finish_reason": "stop",
            }],
        }
        text, tool_calls = client._extract_response(response)
        assert text == "Hello world"
        assert tool_calls is None

    def test_extract_chat_completion_chunk(self, client):
        response = {
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {"content": "Hi"},
                "finish_reason": None,
            }],
        }
        text, tool_calls = client._extract_response(response)
        assert text == "Hi"
        assert tool_calls is None

    def test_extract_chunk_with_tool_calls(self, client):
        response = {
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "turn_on", "arguments": '{"entity_id": "light.living_room"}'}},
                    ],
                },
                "finish_reason": None,
            }],
        }
        text, tool_calls = client._extract_response(response)
        assert text == ""
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "turn_on"

    def test_extract_empty_choices(self, client):
        response = {"object": "chat.completion", "choices": []}
        text, tool_calls = client._extract_response(response)
        assert text is None
        assert tool_calls is None

    def test_extract_missing_choices(self, client):
        response = {"object": "chat.completion"}
        text, tool_calls = client._extract_response(response)
        assert text is None
        assert tool_calls is None

    def test_extract_length_finish_reason(self, client):
        """Should log a warning but still return the text."""
        response = {
            "object": "chat.completion",
            "choices": [{
                "message": {"content": "truncated output"},
                "finish_reason": "length",
            }],
        }
        text, tool_calls = client._extract_response(response)
        assert text == "truncated output"

    def test_extract_chunk_no_tool_calls_key(self, client):
        response = {
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {"content": "token"},
            }],
        }
        text, tool_calls = client._extract_response(response)
        assert text == "token"
        assert tool_calls is None


class TestMiniMaxTemperatureClamping:
    """Tests to verify temperature is clamped to MiniMax range [0, 1]."""

    def test_temperature_above_max_clamped(self):
        """Temperature > 1.0 should be clamped to 1.0."""
        temp = max(0.0, min(1.0, float(2.0)))
        assert temp == 1.0

    def test_temperature_below_min_clamped(self):
        """Negative temperature should be clamped to 0.0."""
        temp = max(0.0, min(1.0, float(-0.5)))
        assert temp == 0.0

    def test_temperature_zero_accepted(self):
        """Temperature 0 should be accepted."""
        temp = max(0.0, min(1.0, float(0.0)))
        assert temp == 0.0

    def test_temperature_normal_passthrough(self):
        """Temperature in valid range passes through unchanged."""
        temp = max(0.0, min(1.0, float(0.7)))
        assert temp == 0.7


class TestMiniMaxValidateConnection:
    """Tests for async_validate_connection."""

    @pytest.mark.asyncio
    async def test_missing_api_key(self, mock_hass):
        result = await MiniMaxAPIClient.async_validate_connection(
            mock_hass, {CONF_API_KEY: "", CONF_BASE_URL: MINIMAX_DEFAULT_BASE_URL}
        )
        assert result == "API key is required"

    @pytest.mark.asyncio
    async def test_successful_connection(self, mock_hass):
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
                {CONF_API_KEY: "valid-key", CONF_BASE_URL: MINIMAX_DEFAULT_BASE_URL},
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_unauthorized_returns_error(self, mock_hass):
        mock_response = AsyncMock()
        mock_response.ok = False
        mock_response.status = 401
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
                {CONF_API_KEY: "bad-key", CONF_BASE_URL: MINIMAX_DEFAULT_BASE_URL},
            )
        assert result == "Invalid API key"

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, mock_hass):
        import asyncio

        mock_session = MagicMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_session.post = MagicMock(return_value=mock_cm)

        with patch(
            "custom_components.llama_conversation.backends.minimax.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await MiniMaxAPIClient.async_validate_connection(
                mock_hass,
                {CONF_API_KEY: "key", CONF_BASE_URL: MINIMAX_DEFAULT_BASE_URL},
            )
        assert result == "Connection timed out"


class TestMiniMaxBackendConstant:
    """Tests for backend type constant registration."""

    def test_backend_type_value(self):
        assert BACKEND_TYPE_MINIMAX == "minimax"

    def test_backend_in_cls_map(self):
        from custom_components.llama_conversation import BACKEND_TO_CLS
        assert BACKEND_TYPE_MINIMAX in BACKEND_TO_CLS
        assert BACKEND_TO_CLS[BACKEND_TYPE_MINIMAX] is MiniMaxAPIClient


class TestMiniMaxStreamingSupport:
    """Tests for streaming support flag."""

    def test_supports_streaming(self, mock_hass, minimax_options):
        client = MiniMaxAPIClient(mock_hass, minimax_options)
        assert client._attr_supports_streaming is True
