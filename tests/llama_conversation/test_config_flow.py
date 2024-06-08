import pytest
from unittest.mock import patch, MagicMock

from homeassistant import config_entries, setup
from homeassistant.core import HomeAssistant
from homeassistant.const import (
    CONF_HOST,
    CONF_PORT,
    CONF_SSL,
    CONF_LLM_HASS_API,
)
from homeassistant.data_entry_flow import FlowResultType

from custom_components.llama_conversation.config_flow import local_llama_config_option_schema, ConfigFlow
from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_MIN_P,
    CONF_TYPICAL_P,
    CONF_REQUEST_TIMEOUT,
    CONF_BACKEND_TYPE,
    CONF_DOWNLOADED_MODEL_FILE,
    CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
    CONF_PROMPT_TEMPLATE,
    CONF_TOOL_FORMAT,
    CONF_TOOL_MULTI_TURN_CHAT,
    CONF_ENABLE_FLASH_ATTENTION,
    CONF_USE_GBNF_GRAMMAR,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    CONF_IN_CONTEXT_EXAMPLES_FILE,
    CONF_NUM_IN_CONTEXT_EXAMPLES,
    CONF_TEXT_GEN_WEBUI_PRESET,
    CONF_OPENAI_API_KEY,
    CONF_TEXT_GEN_WEBUI_ADMIN_KEY,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_PROMPT_CACHING_ENABLED,
    CONF_PROMPT_CACHING_INTERVAL,
    CONF_SERVICE_CALL_REGEX,
    CONF_REMOTE_USE_CHAT_ENDPOINT,
    CONF_TEXT_GEN_WEBUI_CHAT_MODE,
    CONF_OLLAMA_KEEP_ALIVE_MIN,
    CONF_OLLAMA_JSON_MODE,
    CONF_CONTEXT_LENGTH,
    CONF_BATCH_SIZE,
    CONF_THREAD_COUNT,
    CONF_BATCH_THREAD_COUNT,
    BACKEND_TYPE_LLAMA_HF,
    BACKEND_TYPE_LLAMA_EXISTING,
    BACKEND_TYPE_TEXT_GEN_WEBUI,
    BACKEND_TYPE_GENERIC_OPENAI,
    BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER,
    BACKEND_TYPE_OLLAMA,
    DEFAULT_CHAT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_MIN_P,
    DEFAULT_TYPICAL_P,
    DEFAULT_BACKEND_TYPE,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_ENABLE_FLASH_ATTENTION,
    DEFAULT_USE_GBNF_GRAMMAR,
    DEFAULT_GBNF_GRAMMAR_FILE,
    DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    DEFAULT_IN_CONTEXT_EXAMPLES_FILE,
    DEFAULT_NUM_IN_CONTEXT_EXAMPLES,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_NUM_INTERACTIONS,
    DEFAULT_PROMPT_CACHING_ENABLED,
    DEFAULT_PROMPT_CACHING_INTERVAL,
    DEFAULT_SERVICE_CALL_REGEX,
    DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
    DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE,
    DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
    DEFAULT_OLLAMA_JSON_MODE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_THREAD_COUNT,
    DEFAULT_BATCH_THREAD_COUNT,
    DOMAIN,
)

# async def test_validate_config_flow_llama_hf(hass: HomeAssistant):
#     result = await hass.config_entries.flow.async_init(
#         DOMAIN, context={"source": config_entries.SOURCE_USER}
#     )
#     assert result["type"] == FlowResultType.FORM
#     assert result["errors"] is None

#     result2 = await hass.config_entries.flow.async_configure(
#         result["flow_id"], { CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_HF },
#     )
#     assert result2["type"] == FlowResultType.FORM

#     with patch("custom_components.llama_conversation.async_setup_entry", return_value=True) as mock_setup_entry:
#         result3 = await hass.config_entries.flow.async_configure(
#             result2["flow_id"],
#             TEST_DATA,
#         )
#         await hass.async_block_till_done()

#     assert result3["type"] == "create_entry"
#     assert result3["title"] == ""
#     assert result3["data"] == {
#         # ACCOUNT_ID: TEST_DATA["account_id"],
#         # CONF_PASSWORD: TEST_DATA["password"],
#         # CONNECTION_TYPE: CLOUD,
#     }
#     assert result3["options"] == {}
#     assert len(mock_setup_entry.mock_calls) == 1

@pytest.fixture
def validate_connections_mock():
    validate_mock = MagicMock()
    with patch.object(ConfigFlow, '_validate_text_generation_webui', new=validate_mock), \
         patch.object(ConfigFlow, '_validate_ollama', new=validate_mock):
        yield validate_mock

@pytest.fixture
def mock_setup_entry():
    with patch("custom_components.llama_conversation.async_setup_entry", return_value=True) as mock_setup_entry, \
        patch("custom_components.llama_conversation.async_unload_entry", return_value=True):
        yield mock_setup_entry

async def test_validate_config_flow_generic_openai(mock_setup_entry, hass: HomeAssistant, enable_custom_integrations):    
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {}
    assert result["step_id"] == "pick_backend"

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"], { CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI },
    )

    assert result2["type"] == FlowResultType.FORM
    assert result2["errors"] == {}
    assert result2["step_id"] == "remote_model"

    result3 = await hass.config_entries.flow.async_configure(
        result2["flow_id"],
        {
            CONF_HOST: "localhost",
            CONF_PORT: "5000",
            CONF_SSL: False,
            CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        },
    )

    assert result3["type"] == FlowResultType.FORM
    assert result3["errors"] == {}
    assert result3["step_id"] == "model_parameters"

    options_dict = {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_REQUEST_TIMEOUT: DEFAULT_REQUEST_TIMEOUT,
        CONF_PROMPT_TEMPLATE: DEFAULT_PROMPT_TEMPLATE,
        CONF_EXTRA_ATTRIBUTES_TO_EXPOSE: DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
        CONF_REFRESH_SYSTEM_PROMPT: DEFAULT_REFRESH_SYSTEM_PROMPT,
        CONF_REMEMBER_CONVERSATION: DEFAULT_REMEMBER_CONVERSATION,
        CONF_REMEMBER_NUM_INTERACTIONS: DEFAULT_REMEMBER_NUM_INTERACTIONS,
        CONF_SERVICE_CALL_REGEX: DEFAULT_SERVICE_CALL_REGEX,
        CONF_REMOTE_USE_CHAT_ENDPOINT: DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
        CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES,
        CONF_IN_CONTEXT_EXAMPLES_FILE: DEFAULT_IN_CONTEXT_EXAMPLES_FILE,
        CONF_NUM_IN_CONTEXT_EXAMPLES: DEFAULT_NUM_IN_CONTEXT_EXAMPLES,
    }

    result4 = await hass.config_entries.flow.async_configure(
        result2["flow_id"], options_dict
    )
    await hass.async_block_till_done()

    assert result4["type"] == "create_entry"
    assert result4["title"] == f"LLM Model '{DEFAULT_CHAT_MODEL}' (remote)"
    assert result4["data"] == {
        CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI,
        CONF_HOST: "localhost",
        CONF_PORT: "5000",
        CONF_SSL: False,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
    }
    assert result4["options"] == options_dict
    assert len(mock_setup_entry.mock_calls) == 1

async def test_validate_config_flow_ollama(mock_setup_entry, hass: HomeAssistant, enable_custom_integrations, validate_connections_mock):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {}
    assert result["step_id"] == "pick_backend"

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"], { CONF_BACKEND_TYPE: BACKEND_TYPE_OLLAMA },
    )

    assert result2["type"] == FlowResultType.FORM
    assert result2["errors"] == {}
    assert result2["step_id"] == "remote_model"

    # simulate incorrect settings on first try
    validate_connections_mock.side_effect = [
        ("failed_to_connect", Exception("ConnectionError"), []),
        (None, None, [])
    ]

    result3 = await hass.config_entries.flow.async_configure(
        result2["flow_id"],
        {
            CONF_HOST: "localhost",
            CONF_PORT: "5000",
            CONF_SSL: False,
            CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        },
    )

    assert result3["type"] == FlowResultType.FORM
    assert len(result3["errors"]) == 1
    assert "base" in result3["errors"]
    assert result3["step_id"] == "remote_model"

    # retry
    result3 = await hass.config_entries.flow.async_configure(
        result2["flow_id"],
        {
            CONF_HOST: "localhost",
            CONF_PORT: "5001",
            CONF_SSL: False,
            CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        },
    )

    assert result3["type"] == FlowResultType.FORM
    assert result3["errors"] == {}
    assert result3["step_id"] == "model_parameters"

    options_dict = {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TOP_K: DEFAULT_TOP_K,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_TYPICAL_P: DEFAULT_MIN_P,
        CONF_REQUEST_TIMEOUT: DEFAULT_REQUEST_TIMEOUT,
        CONF_PROMPT_TEMPLATE: DEFAULT_PROMPT_TEMPLATE,
        CONF_EXTRA_ATTRIBUTES_TO_EXPOSE: DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
        CONF_REFRESH_SYSTEM_PROMPT: DEFAULT_REFRESH_SYSTEM_PROMPT,
        CONF_REMEMBER_CONVERSATION: DEFAULT_REMEMBER_CONVERSATION,
        CONF_REMEMBER_NUM_INTERACTIONS: DEFAULT_REMEMBER_NUM_INTERACTIONS,
        CONF_SERVICE_CALL_REGEX: DEFAULT_SERVICE_CALL_REGEX,
        CONF_REMOTE_USE_CHAT_ENDPOINT: DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
        CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES,
        CONF_IN_CONTEXT_EXAMPLES_FILE: DEFAULT_IN_CONTEXT_EXAMPLES_FILE,
        CONF_NUM_IN_CONTEXT_EXAMPLES: DEFAULT_NUM_IN_CONTEXT_EXAMPLES,
        CONF_CONTEXT_LENGTH: DEFAULT_CONTEXT_LENGTH,
        CONF_OLLAMA_KEEP_ALIVE_MIN: DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
        CONF_OLLAMA_JSON_MODE: DEFAULT_OLLAMA_JSON_MODE,
    }

    result4 = await hass.config_entries.flow.async_configure(
        result2["flow_id"], options_dict
    )
    await hass.async_block_till_done()

    assert result4["type"] == "create_entry"
    assert result4["title"] == f"LLM Model '{DEFAULT_CHAT_MODEL}' (remote)"
    assert result4["data"] == {
        CONF_BACKEND_TYPE: BACKEND_TYPE_OLLAMA,
        CONF_HOST: "localhost",
        CONF_PORT: "5001",
        CONF_SSL: False,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
    }
    assert result4["options"] == options_dict
    mock_setup_entry.assert_called_once()

# TODO: write tests for configflow setup for llama.cpp (both versions) + text-generation-webui

def test_validate_options_schema(hass: HomeAssistant):

    universal_options = [
        CONF_LLM_HASS_API, CONF_PROMPT, CONF_PROMPT_TEMPLATE, CONF_TOOL_FORMAT, CONF_TOOL_MULTI_TURN_CHAT,
        CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES, CONF_IN_CONTEXT_EXAMPLES_FILE, CONF_NUM_IN_CONTEXT_EXAMPLES,
        CONF_MAX_TOKENS, CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
        CONF_SERVICE_CALL_REGEX, CONF_REFRESH_SYSTEM_PROMPT, CONF_REMEMBER_CONVERSATION, CONF_REMEMBER_NUM_INTERACTIONS,
    ]

    options_llama_hf = local_llama_config_option_schema(hass, None, BACKEND_TYPE_LLAMA_HF)
    assert set(options_llama_hf.keys()) == set(universal_options + [
        CONF_TOP_K, CONF_TEMPERATURE, CONF_TOP_P, CONF_MIN_P, CONF_TYPICAL_P, # supports all sampling parameters
        CONF_BATCH_SIZE, CONF_THREAD_COUNT, CONF_BATCH_THREAD_COUNT, CONF_ENABLE_FLASH_ATTENTION, # llama.cpp specific
        CONF_CONTEXT_LENGTH, # supports context length
        CONF_USE_GBNF_GRAMMAR, CONF_GBNF_GRAMMAR_FILE, # supports GBNF
        CONF_PROMPT_CACHING_ENABLED, CONF_PROMPT_CACHING_INTERVAL # supports prompt caching
    ])

    options_llama_existing = local_llama_config_option_schema(hass, None, BACKEND_TYPE_LLAMA_EXISTING)
    assert set(options_llama_existing.keys()) == set(universal_options + [
        CONF_TOP_K, CONF_TEMPERATURE, CONF_TOP_P, CONF_MIN_P, CONF_TYPICAL_P, # supports all sampling parameters
        CONF_BATCH_SIZE, CONF_THREAD_COUNT, CONF_BATCH_THREAD_COUNT, CONF_ENABLE_FLASH_ATTENTION, # llama.cpp specific
        CONF_CONTEXT_LENGTH, # supports context length
        CONF_USE_GBNF_GRAMMAR, CONF_GBNF_GRAMMAR_FILE, # supports GBNF
        CONF_PROMPT_CACHING_ENABLED, CONF_PROMPT_CACHING_INTERVAL # supports prompt caching
    ])

    options_ollama = local_llama_config_option_schema(hass, None, BACKEND_TYPE_OLLAMA)
    assert set(options_ollama.keys()) == set(universal_options + [
        CONF_TOP_K, CONF_TEMPERATURE, CONF_TOP_P, CONF_TYPICAL_P, # supports top_k temperature, top_p and typical_p samplers
        CONF_OLLAMA_KEEP_ALIVE_MIN, CONF_OLLAMA_JSON_MODE, # ollama specific
        CONF_CONTEXT_LENGTH, # supports context length
        CONF_REMOTE_USE_CHAT_ENDPOINT, CONF_REQUEST_TIMEOUT, # is a remote backend
    ])

    options_text_gen_webui = local_llama_config_option_schema(hass, None, BACKEND_TYPE_TEXT_GEN_WEBUI)
    assert set(options_text_gen_webui.keys()) == set(universal_options + [
        CONF_TOP_K, CONF_TEMPERATURE, CONF_TOP_P, CONF_MIN_P, CONF_TYPICAL_P, # supports all sampling parameters
        CONF_TEXT_GEN_WEBUI_CHAT_MODE, CONF_TEXT_GEN_WEBUI_PRESET, # text-gen-webui specific
        CONF_CONTEXT_LENGTH, # supports context length
        CONF_REMOTE_USE_CHAT_ENDPOINT, CONF_REQUEST_TIMEOUT, # is a remote backend
    ])

    options_generic_openai = local_llama_config_option_schema(hass, None, BACKEND_TYPE_GENERIC_OPENAI)
    assert set(options_generic_openai.keys()) == set(universal_options + [
        CONF_TEMPERATURE, CONF_TOP_P, # only supports top_p and temperature sampling
        CONF_REMOTE_USE_CHAT_ENDPOINT, CONF_REQUEST_TIMEOUT, # is a remote backend
    ])

    options_llama_cpp_python_server = local_llama_config_option_schema(hass, None, BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER)
    assert set(options_llama_cpp_python_server.keys()) == set(universal_options + [
        CONF_TOP_K, CONF_TEMPERATURE, CONF_TOP_P, # supports top_k, temperature, and top p sampling
        CONF_USE_GBNF_GRAMMAR, CONF_GBNF_GRAMMAR_FILE, # supports GBNF
        CONF_REMOTE_USE_CHAT_ENDPOINT, CONF_REQUEST_TIMEOUT, # is a remote backend
    ])