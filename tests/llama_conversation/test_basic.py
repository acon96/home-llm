"""Lightweight smoke tests for backend helpers.

These avoid backend calls and only cover helper utilities to keep the suite green
while the integration evolves. No integration code is modified.
"""

import pytest

from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL

from custom_components.llama_conversation.backends.llamacpp import snapshot_settings
from custom_components.llama_conversation.backends.ollama import OllamaAPIClient, _normalize_path
from custom_components.llama_conversation.backends.generic_openai import GenericOpenAIAPIClient
from custom_components.llama_conversation.const import (
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
    CONF_GENERIC_OPENAI_PATH,
)


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
            CONF_GENERIC_OPENAI_PATH: "v1",
            CONF_CHAT_MODEL: "demo",
        },
    )
    name = client.get_name(
        {
            CONF_HOST: "localhost",
            CONF_PORT: "8080",
            CONF_SSL: False,
            CONF_GENERIC_OPENAI_PATH: "v1",
        }
    )
    assert "Generic OpenAI" in name
    assert "localhost" in name


def test_normalize_path_helper():
    assert _normalize_path(None) == ""
    assert _normalize_path("") == ""
    assert _normalize_path("/v1/") == "/v1"
    assert _normalize_path("v2") == "/v2"
