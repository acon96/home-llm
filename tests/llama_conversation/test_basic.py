"""Lightweight smoke tests for backend helpers.

These avoid backend calls and only cover helper utilities to keep the suite green
while the integration evolves. No integration code is modified.
"""

import pytest
from openai import OpenAIError

from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.exceptions import ConfigEntryError

from custom_components.llama_conversation.backends.llamacpp import LlamaCppClient, snapshot_settings
from custom_components.llama_conversation.backends.ollama import OllamaAPIClient, _normalize_path
from custom_components.llama_conversation.backends.generic_openai import GenericOpenAIAPIClient
from custom_components.llama_conversation.const import (
    CONF_API_KEY,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_LENGTH,
    CONF_LLAMACPP_BATCH_SIZE,
    CONF_LLAMACPP_BATCH_THREAD_COUNT,
    CONF_LLAMACPP_THREAD_COUNT,
    CONF_LLAMACPP_ENABLE_FLASH_ATTENTION,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_PROMPT_CACHING_ENABLED,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_LLAMACPP_BATCH_SIZE,
    DEFAULT_LLAMACPP_BATCH_THREAD_COUNT,
    DEFAULT_LLAMACPP_THREAD_COUNT,
    DEFAULT_LLAMACPP_ENABLE_FLASH_ATTENTION,
    DEFAULT_GBNF_GRAMMAR_FILE,
    DEFAULT_PROMPT_CACHING_ENABLED,
    CONF_API_PATH,
    CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    RECOMMENDED_CHAT_MODELS,
)
from custom_components.llama_conversation.utils import LlamaCppPythonInstallError
from custom_components.llama_conversation.utils import LlamaCppPythonInstallError, strip_thinking_blocks
from custom_components.llama_conversation.const import DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX

class TestStripThinkingBlocks:
    """Tests for the thinking-block sanitizer that prevents reasoning leakage into TTS speech."""

    def test_no_blocks_pass_through(self):
        content = "Hello, I can help you with that."
        result = strip_thinking_blocks(content, DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX)
        assert result == content

    def test_single_closed_block_stripped(self):
        content = "<think>Let me think about this...</think> Hello, the light is on."
        result = strip_thinking_blocks(content, DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX)
        assert result == "Hello, the light is on."

    def test_multiple_closed_blocks_stripped(self):
        content = "<think>First thought...</think> Some text <think>Second thought...</think> Final answer."
        result = strip_thinking_blocks(content, DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX)
        assert result == "Some text  Final answer."

    def test_unclosed_block_truncates(self):
        """If a thinking block opens but never closes, everything after is dropped."""
        content = "<think>This reasoning should not leak..."
        result = strip_thinking_blocks(content, DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX)
        assert result == ""

    def test_unclosed_block_after_text(self):
        content = "Here is the answer. <think>Hidden reasoning follows"
        result = strip_thinking_blocks(content, DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX)
        assert result == "Here is the answer."

    def test_empty_content_returns_empty(self):
        result = strip_thinking_blocks("", DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX)
        assert result == ""

    def test_none_content_returns_none(self):
        result = strip_thinking_blocks(None, DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX)
        assert result is None

    def test_empty_prefix_returns_original(self):
        content = "<think>hidden</think> visible"
        result = strip_thinking_blocks(content, "", DEFAULT_THINKING_SUFFIX)
        assert result == content

    def test_empty_suffix_returns_original(self):
        content = "<think>hidden</think> visible"
        result = strip_thinking_blocks(content, DEFAULT_THINKING_PREFIX, "")
        assert result == content

    def test_whitespace_stripped_from_result(self):
        content = "  \n<think>reasoning...</think>  \n"
        result = strip_thinking_blocks(content, DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX)
        assert result == ""

    def test_custom_prefix_suffix(self):
        result = strip_thinking_blocks(
            "<think>hidden</think> visible",
            "<think>",
            "</think>",
        )
        assert result == "visible"

    def test_nested_prefix_in_text(self):
        """First prefix greedily matches the first suffix; content between is stripped."""
        content = "I have <think> marks in my notes. <think>real block</think> done"
        result = strip_thinking_blocks(content, DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX)
        assert result == "I have  done"

    def test_only_block_content_becomes_empty(self):
        content = "<think>all reasoning</think>"
        result = strip_thinking_blocks(content, DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX)
        assert result == ""

    def test_text_before_and_after_blocks(self):
        content = "intro <think>middle</think> outro"
        result = strip_thinking_blocks(content, DEFAULT_THINKING_PREFIX, DEFAULT_THINKING_SUFFIX)
        assert result == "intro  outro"


@pytest.fixture
async def hass_defaults(hass):
    return hass


def test_snapshot_settings_defaults():
    options = {CONF_CHAT_MODEL: "test-model"}
    snap = snapshot_settings(options)
    assert snap[CONF_CONTEXT_LENGTH] == DEFAULT_CONTEXT_LENGTH
    assert snap[CONF_LLAMACPP_BATCH_SIZE] == DEFAULT_LLAMACPP_BATCH_SIZE
    assert snap[CONF_LLAMACPP_THREAD_COUNT] == DEFAULT_LLAMACPP_THREAD_COUNT
    assert snap[CONF_LLAMACPP_BATCH_THREAD_COUNT] == DEFAULT_LLAMACPP_BATCH_THREAD_COUNT
    assert snap[CONF_LLAMACPP_ENABLE_FLASH_ATTENTION] == DEFAULT_LLAMACPP_ENABLE_FLASH_ATTENTION
    assert snap[CONF_GBNF_GRAMMAR_FILE] == DEFAULT_GBNF_GRAMMAR_FILE
    assert snap[CONF_PROMPT_CACHING_ENABLED] == DEFAULT_PROMPT_CACHING_ENABLED


def test_snapshot_settings_overrides():
    options = {
        CONF_CONTEXT_LENGTH: 4096,
        CONF_LLAMACPP_BATCH_SIZE: 64,
        CONF_LLAMACPP_THREAD_COUNT: 6,
        CONF_LLAMACPP_BATCH_THREAD_COUNT: 3,
        CONF_LLAMACPP_ENABLE_FLASH_ATTENTION: True,
        CONF_GBNF_GRAMMAR_FILE: "custom.gbnf",
        CONF_PROMPT_CACHING_ENABLED: True,
    }
    snap = snapshot_settings(options)
    assert snap[CONF_CONTEXT_LENGTH] == 4096
    assert snap[CONF_LLAMACPP_BATCH_SIZE] == 64
    assert snap[CONF_LLAMACPP_THREAD_COUNT] == 6
    assert snap[CONF_LLAMACPP_BATCH_THREAD_COUNT] == 3
    assert snap[CONF_LLAMACPP_ENABLE_FLASH_ATTENTION] is True
    assert snap[CONF_GBNF_GRAMMAR_FILE] == "custom.gbnf"
    assert snap[CONF_PROMPT_CACHING_ENABLED] is True


def test_ollama_keep_alive_formatting():
    assert OllamaAPIClient._format_keep_alive("0") == 0
    assert OllamaAPIClient._format_keep_alive("0.0") == 0
    assert OllamaAPIClient._format_keep_alive(5) == "5m"
    assert OllamaAPIClient._format_keep_alive("15") == "15m"


def test_generic_openai_name_and_path(hass_defaults):
    client = GenericOpenAIAPIClient(
        hass_defaults,
        {
            CONF_HOST: "localhost",
            CONF_PORT: "8080",
            CONF_SSL: False,
            CONF_API_PATH: "v1",
            CONF_CHAT_MODEL: "demo",
        },
    )
    name = client.get_name(
        {
            CONF_HOST: "localhost",
            CONF_PORT: "8080",
            CONF_SSL: False,
            CONF_API_PATH: "v1",
        }
    )
    assert "Generic OpenAI" in name
    assert "localhost" in name


def test_normalize_path_helper():
    assert _normalize_path(None) == ""
    assert _normalize_path("") == ""
    assert _normalize_path("/v1/") == "/v1"
    assert _normalize_path("v2") == "/v2"


@pytest.mark.asyncio
async def test_llama_cpp_startup_validation_surfaces_install_error(monkeypatch, hass):
    client = LlamaCppClient(hass, {CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: False})

    monkeypatch.setattr(
        "custom_components.llama_conversation.backends.llamacpp.validate_llama_cpp_python_installation",
        lambda: None,
    )
    monkeypatch.setattr(
        "custom_components.llama_conversation.backends.llamacpp.importlib.util.find_spec",
        lambda _module: None,
    )

    def raise_install_error(*_args, **_kwargs):
        raise LlamaCppPythonInstallError("Unable to install package wheel: unexpected BufError")

    monkeypatch.setattr(
        "custom_components.llama_conversation.backends.llamacpp.install_llama_cpp_python",
        raise_install_error,
    )

    with pytest.raises(ConfigEntryError, match="unexpected BufError"):
        await client.async_validate_startup()


@pytest.mark.asyncio
async def test_generic_openai_validate_connection_uses_formatted_base_url(monkeypatch, hass):
    captured: dict[str, str] = {}

    class FakeListResult:
        def __await__(self):
            async def _done():
                return self
            return _done().__await__()

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class FakeModels:
        def list(self):
            return FakeListResult()

    class FakeClient:
        def __init__(self, *, api_key, base_url, timeout=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["timeout"] = timeout
            self.models = FakeModels()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "custom_components.llama_conversation.backends.generic_openai.AsyncOpenAI",
        FakeClient,
    )

    err = await GenericOpenAIAPIClient.async_validate_connection(
        hass,
        {
            CONF_HOST: "localhost",
            CONF_PORT: "11434",
            CONF_SSL: False,
            CONF_API_PATH: "v1",
            CONF_API_KEY: "token",
        },
    )

    assert err is None
    assert captured["api_key"] == "token"
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["timeout"] == 5


@pytest.mark.asyncio
async def test_generic_openai_get_available_models_falls_back_on_openai_error(monkeypatch, hass_defaults):
    class FailingModelsAPI:
        def list(self):
            raise OpenAIError("boom")

    class FakeClient:
        def __init__(self, **_kwargs):
            self.models = FailingModelsAPI()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "custom_components.llama_conversation.backends.generic_openai.AsyncOpenAI",
        FakeClient,
    )

    client = GenericOpenAIAPIClient(
        hass_defaults,
        {
            CONF_HOST: "localhost",
            CONF_PORT: "11434",
            CONF_SSL: False,
            CONF_API_PATH: "v1",
            CONF_CHAT_MODEL: "demo",
        },
    )

    models = await client.async_get_available_models()

    assert models == RECOMMENDED_CHAT_MODELS


def test_ollama_client_supports_streaming():
    """OllamaAPIClient must declare streaming support like the other backends."""
    assert getattr(OllamaAPIClient, '_attr_supports_streaming', False) is True


def test_generic_openai_client_supports_streaming():
    """GenericOpenAIAPIClient must declare streaming support."""
    assert getattr(GenericOpenAIAPIClient, '_attr_supports_streaming', False) is True
