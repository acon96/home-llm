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
    class ExplodingModels:
        def list(self):
            raise OpenAIError("boom")

    class FakeClient:
        def __init__(self, **_kwargs):
            self.models = ExplodingModels()

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
