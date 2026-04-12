"""Config flow option schema tests to ensure options are wired per-backend."""

import asyncio

import pytest

from pytest_homeassistant_custom_component.common import MockConfigEntry

from homeassistant.const import CONF_HOST, CONF_LLM_HASS_API, CONF_PORT, CONF_SSL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.data_entry_flow import FlowResultType

from custom_components.llama_conversation.config_flow import ConfigFlow, local_llama_config_option_schema
from custom_components.llama_conversation.const import (
    BACKEND_TYPE_LLAMA_CPP,
    BACKEND_TYPE_TEXT_GEN_WEBUI,
    BACKEND_TYPE_GENERIC_OPENAI,
    CONF_BACKEND_TYPE,
    CONF_SELECTED_LANGUAGE,
    BACKEND_TYPE_LLAMA_CPP_SERVER,
    BACKEND_TYPE_OLLAMA,
    CONF_CONTEXT_LENGTH,
    CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_LLAMACPP_BATCH_SIZE,
    CONF_LLAMACPP_BATCH_THREAD_COUNT,
    CONF_LLAMACPP_ENABLE_FLASH_ATTENTION,
    CONF_LLAMACPP_THREAD_COUNT,
    CONF_MAX_TOKENS,
    CONF_MIN_P,
    CONF_NUM_IN_CONTEXT_EXAMPLES,
    CONF_OLLAMA_JSON_MODE,
    CONF_OLLAMA_KEEP_ALIVE_MIN,
    CONF_PROMPT,
    CONF_PROMPT_CACHING_ENABLED,
    CONF_PROMPT_CACHING_INTERVAL,
    CONF_REQUEST_TIMEOUT,
    CONF_TEXT_GEN_WEBUI_CHAT_MODE,
    CONF_TEXT_GEN_WEBUI_PRESET,
    CONF_THINKING_PREFIX,
    CONF_TOOL_CALL_PREFIX,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_TYPICAL_P,
    CONF_TEMPERATURE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_LLAMACPP_BATCH_SIZE,
    DEFAULT_LLAMACPP_BATCH_THREAD_COUNT,
    DEFAULT_LLAMACPP_ENABLE_FLASH_ATTENTION,
    DEFAULT_LLAMACPP_THREAD_COUNT,
    DEFAULT_NUM_IN_CONTEXT_EXAMPLES,
    DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
    DEFAULT_OLLAMA_JSON_MODE,
    DEFAULT_PROMPT,
    DEFAULT_PROMPT_CACHING_INTERVAL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE,
    DEFAULT_THINKING_PREFIX,
    DEFAULT_TOOL_CALL_PREFIX,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_TYPICAL_P,
    DOMAIN,
)


def _build_flow(hass: HomeAssistant) -> ConfigFlow:
    flow = ConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}
    return flow


def _schema(hass: HomeAssistant, backend: str, options: dict | None = None):
    return local_llama_config_option_schema(
        hass=hass,
        language="en",
        options=options or {},
        backend_type=backend,
        subentry_type="conversation",
    )


def _get_default(schema: dict, key_name: str):
    for key in schema:
        if getattr(key, "schema", None) == key_name:
            default = getattr(key, "default", None)
            return default() if callable(default) else default
    raise AssertionError(f"Key {key_name} not found in schema")


def _get_suggested(schema: dict, key_name: str):
    for key in schema:
        if getattr(key, "schema", None) == key_name:
            return (getattr(key, "description", {}) or {}).get("suggested_value")
    raise AssertionError(f"Key {key_name} not found in schema")


def test_schema_llama_cpp_defaults_and_overrides(hass: HomeAssistant):
    overrides = {
        CONF_CONTEXT_LENGTH: 4096,
        CONF_LLAMACPP_BATCH_SIZE: 8,
        CONF_LLAMACPP_THREAD_COUNT: 6,
        CONF_LLAMACPP_BATCH_THREAD_COUNT: 3,
        CONF_LLAMACPP_ENABLE_FLASH_ATTENTION: True,
        CONF_PROMPT_CACHING_INTERVAL: 15,
        CONF_TOP_K: 12,
        CONF_TOOL_CALL_PREFIX: "<tc>",
    }

    schema = _schema(hass, BACKEND_TYPE_LLAMA_CPP, overrides)

    expected_keys = {
        CONF_MAX_TOKENS,
        CONF_CONTEXT_LENGTH,
        CONF_TOP_K,
        CONF_TOP_P,
        CONF_MIN_P,
        CONF_TYPICAL_P,
        CONF_PROMPT_CACHING_ENABLED,
        CONF_PROMPT_CACHING_INTERVAL,
        CONF_GBNF_GRAMMAR_FILE,
        CONF_LLAMACPP_BATCH_SIZE,
        CONF_LLAMACPP_THREAD_COUNT,
        CONF_LLAMACPP_BATCH_THREAD_COUNT,
        CONF_LLAMACPP_ENABLE_FLASH_ATTENTION,
    }
    assert expected_keys.issubset({getattr(k, "schema", None) for k in schema})

    assert _get_default(schema, CONF_CONTEXT_LENGTH) == DEFAULT_CONTEXT_LENGTH
    assert _get_default(schema, CONF_LLAMACPP_BATCH_SIZE) == DEFAULT_LLAMACPP_BATCH_SIZE
    assert _get_default(schema, CONF_LLAMACPP_THREAD_COUNT) == DEFAULT_LLAMACPP_THREAD_COUNT
    assert _get_default(schema, CONF_LLAMACPP_BATCH_THREAD_COUNT) == DEFAULT_LLAMACPP_BATCH_THREAD_COUNT
    assert _get_default(schema, CONF_LLAMACPP_ENABLE_FLASH_ATTENTION) is DEFAULT_LLAMACPP_ENABLE_FLASH_ATTENTION
    assert _get_default(schema, CONF_PROMPT_CACHING_INTERVAL) == DEFAULT_PROMPT_CACHING_INTERVAL
    # suggested values should reflect overrides
    assert _get_suggested(schema, CONF_CONTEXT_LENGTH) == 4096
    assert _get_suggested(schema, CONF_LLAMACPP_BATCH_SIZE) == 8
    assert _get_suggested(schema, CONF_LLAMACPP_THREAD_COUNT) == 6
    assert _get_suggested(schema, CONF_LLAMACPP_BATCH_THREAD_COUNT) == 3
    assert _get_suggested(schema, CONF_LLAMACPP_ENABLE_FLASH_ATTENTION) is True
    assert _get_suggested(schema, CONF_PROMPT_CACHING_INTERVAL) == 15
    assert _get_suggested(schema, CONF_TOP_K) == 12
    assert _get_suggested(schema, CONF_TOOL_CALL_PREFIX) == "<tc>"


def test_schema_text_gen_webui_options_preserved(hass: HomeAssistant):
    overrides = {
        CONF_REQUEST_TIMEOUT: 123,
        CONF_TEXT_GEN_WEBUI_PRESET: "custom-preset",
        CONF_TEXT_GEN_WEBUI_CHAT_MODE: DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE,
        CONF_CONTEXT_LENGTH: 2048,
    }

    schema = _schema(hass, BACKEND_TYPE_TEXT_GEN_WEBUI, overrides)

    expected = {CONF_TEXT_GEN_WEBUI_CHAT_MODE, CONF_TEXT_GEN_WEBUI_PRESET, CONF_REQUEST_TIMEOUT, CONF_CONTEXT_LENGTH}
    assert expected.issubset({getattr(k, "schema", None) for k in schema})
    assert _get_default(schema, CONF_REQUEST_TIMEOUT) == DEFAULT_REQUEST_TIMEOUT
    assert _get_default(schema, CONF_CONTEXT_LENGTH) == DEFAULT_CONTEXT_LENGTH
    assert _get_suggested(schema, CONF_REQUEST_TIMEOUT) == 123
    assert _get_suggested(schema, CONF_TEXT_GEN_WEBUI_PRESET) == "custom-preset"
    assert _get_suggested(schema, CONF_CONTEXT_LENGTH) == 2048


def test_schema_generic_openai_options_preserved(hass: HomeAssistant):
    overrides = {CONF_TOP_P: 0.25, CONF_REQUEST_TIMEOUT: 321}

    schema = _schema(hass, BACKEND_TYPE_GENERIC_OPENAI, overrides)

    assert {CONF_TOP_P, CONF_REQUEST_TIMEOUT}.issubset({getattr(k, "schema", None) for k in schema})
    assert _get_default(schema, CONF_TOP_P) == DEFAULT_TOP_P
    assert _get_default(schema, CONF_REQUEST_TIMEOUT) == DEFAULT_REQUEST_TIMEOUT
    assert _get_suggested(schema, CONF_TOP_P) == 0.25
    assert _get_suggested(schema, CONF_REQUEST_TIMEOUT) == 321
    # Base prompt options still present
    prompt_default = _get_default(schema, CONF_PROMPT)
    assert prompt_default is not None and "You are 'Al'" in prompt_default
    assert _get_default(schema, CONF_NUM_IN_CONTEXT_EXAMPLES) == DEFAULT_NUM_IN_CONTEXT_EXAMPLES


def test_schema_llama_cpp_server_includes_gbnf(hass: HomeAssistant):
    schema = _schema(hass, BACKEND_TYPE_LLAMA_CPP_SERVER)
    keys = {getattr(k, "schema", None) for k in schema}

    assert {CONF_MAX_TOKENS, CONF_TOP_K, CONF_GBNF_GRAMMAR_FILE}.issubset(keys)
    assert _get_default(schema, CONF_GBNF_GRAMMAR_FILE) == "output.gbnf"


def test_schema_ollama_defaults_and_overrides(hass: HomeAssistant):
    overrides = {CONF_OLLAMA_KEEP_ALIVE_MIN: 5, CONF_CONTEXT_LENGTH: 1024, CONF_TOP_K: 7}
    schema = _schema(hass, BACKEND_TYPE_OLLAMA, overrides)

    assert {CONF_MAX_TOKENS, CONF_CONTEXT_LENGTH, CONF_OLLAMA_KEEP_ALIVE_MIN, CONF_OLLAMA_JSON_MODE}.issubset(
        {getattr(k, "schema", None) for k in schema}
    )
    assert _get_default(schema, CONF_OLLAMA_KEEP_ALIVE_MIN) == DEFAULT_OLLAMA_KEEP_ALIVE_MIN
    assert _get_default(schema, CONF_OLLAMA_JSON_MODE) is DEFAULT_OLLAMA_JSON_MODE
    assert _get_default(schema, CONF_CONTEXT_LENGTH) == DEFAULT_CONTEXT_LENGTH
    assert _get_default(schema, CONF_TOP_K) == DEFAULT_TOP_K
    assert _get_suggested(schema, CONF_OLLAMA_KEEP_ALIVE_MIN) == 5
    assert _get_suggested(schema, CONF_CONTEXT_LENGTH) == 1024
    assert _get_suggested(schema, CONF_TOP_K) == 7


def test_schema_includes_llm_api_selector(monkeypatch, hass: HomeAssistant):
    monkeypatch.setattr(
        "custom_components.llama_conversation.config_flow.llm.async_get_apis",
        lambda _hass: [type("API", (), {"id": "dummy", "name": "Dummy API", "tools": []})()],
    )
    schema = _schema(hass, BACKEND_TYPE_LLAMA_CPP)

    assert _get_default(schema, CONF_LLM_HASS_API) is None
    # Base prompt and thinking prefixes use defaults when not overridden
    prompt_default = _get_default(schema, CONF_PROMPT)
    assert prompt_default is not None and "You are 'Al'" in prompt_default
    assert _get_default(schema, CONF_THINKING_PREFIX) == DEFAULT_THINKING_PREFIX
    assert _get_default(schema, CONF_TOOL_CALL_PREFIX) == DEFAULT_TOOL_CALL_PREFIX


@pytest.mark.asyncio
async def test_configure_connection_rejects_invalid_hostname(hass: HomeAssistant):
    flow = _build_flow(hass)
    flow.internal_step = "configure_connection"
    flow.client_config = {
        CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI,
        CONF_SELECTED_LANGUAGE: "en",
    }

    result = await flow.async_step_user(
        {
            CONF_HOST: "bad host name",
            CONF_PORT: "8080",
            CONF_SSL: False,
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "invalid_hostname"


@pytest.mark.asyncio
async def test_configure_connection_surfaces_backend_validation_error(monkeypatch, hass: HomeAssistant):
    flow = _build_flow(hass)
    flow.internal_step = "configure_connection"
    flow.client_config = {
        CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI,
        CONF_SELECTED_LANGUAGE: "en",
    }

    async def fake_validate_connection(_hass, _config):
        return RuntimeError("connection refused")

    monkeypatch.setattr(
        "custom_components.llama_conversation.config_flow.BACKEND_TO_CLS",
        {BACKEND_TYPE_GENERIC_OPENAI: type("Backend", (), {"async_validate_connection": staticmethod(fake_validate_connection)})},
    )

    result = await flow.async_step_user(
        {
            CONF_HOST: "localhost",
            CONF_PORT: "8080",
            CONF_SSL: False,
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "failed_to_connect"
    assert result["description_placeholders"]["exception"] == "connection refused"


@pytest.mark.asyncio
async def test_local_backend_wheel_install_failure_returns_to_backend_picker(monkeypatch, hass: HomeAssistant):
    flow = _build_flow(hass)
    flow.internal_step = "pick_backend"
    flow.client_config = {}

    monkeypatch.setattr(
        "custom_components.llama_conversation.config_flow.get_llama_cpp_python_version",
        lambda: None,
    )
    monkeypatch.setattr(
        "custom_components.llama_conversation.config_flow.install_llama_cpp_python",
        lambda _config_dir: False,
    )
    monkeypatch.setattr(
        hass,
        "async_create_background_task",
        lambda coro, *, name=None: asyncio.create_task(coro, name=name),
    )

    initial = await flow.async_step_user(
        {
            CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_CPP,
            CONF_SELECTED_LANGUAGE: "en",
        }
    )

    assert initial["type"] == FlowResultType.SHOW_PROGRESS
    await flow.install_wheel_task

    progress_done = await flow.async_step_user()

    assert progress_done["type"] == FlowResultType.SHOW_PROGRESS_DONE
    assert flow.internal_step == "pick_backend"

    retry = await flow.async_step_user()

    assert retry["type"] == FlowResultType.FORM
    assert retry["errors"]["base"] == "pip_wheel_error"


@pytest.mark.asyncio
async def test_async_step_finish_rejects_duplicate_llama_cpp_entry(hass: HomeAssistant):
    existing = MockConfigEntry(domain=DOMAIN, data={CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_CPP})
    existing.add_to_hass(hass)

    flow = _build_flow(hass)
    flow.client_config = {
        CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_CPP,
        CONF_SELECTED_LANGUAGE: "en",
    }

    result = await flow.async_step_finish()

    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "duplicate_client"
