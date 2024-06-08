import json
import logging
import pytest
import jinja2
from unittest.mock import patch, MagicMock, PropertyMock, AsyncMock, ANY

from custom_components.llama_conversation.agent import LlamaCppAgent, OllamaAPIAgent, TextGenerationWebuiAgent, GenericOpenAIAPIAgent
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
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT_BASE,
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
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT,
    TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT,
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT,
    DOMAIN,
    PROMPT_TEMPLATE_DESCRIPTIONS,
    DEFAULT_OPTIONS,
)

from homeassistant.components.conversation import ConversationInput
from homeassistant.const import (
    CONF_HOST,
    CONF_PORT,
    CONF_SSL,
    CONF_LLM_HASS_API
)
from homeassistant.helpers.llm import LLM_API_ASSIST, APIInstance

_LOGGER = logging.getLogger(__name__)

class WarnDict(dict):
    def get(self, _key, _default=None):
        if _key in self:
            return self[_key]
        
        _LOGGER.warning(f"attempting to get unset dictionary key {_key}")

        return _default

class MockConfigEntry:
    def __init__(self, entry_id='test_entry_id', data={}, options={}):
        self.entry_id = entry_id
        self.data = WarnDict(data)
        self.options = WarnDict(options)
        

@pytest.fixture
def config_entry():
    yield MockConfigEntry(
        data={
            CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
            CONF_BACKEND_TYPE: DEFAULT_BACKEND_TYPE,
            CONF_DOWNLOADED_MODEL_FILE: "/config/models/some-model.q4_k_m.gguf",
            CONF_HOST: "localhost",
            CONF_PORT: "5000",
            CONF_SSL: False,
            CONF_OPENAI_API_KEY: "OpenAI-API-Key",
            CONF_TEXT_GEN_WEBUI_ADMIN_KEY: "Text-Gen-Webui-Admin-Key"
        },
        options={
            **DEFAULT_OPTIONS,
            CONF_LLM_HASS_API: LLM_API_ASSIST,
            CONF_PROMPT: DEFAULT_PROMPT_BASE,
            CONF_SERVICE_CALL_REGEX: r"({[\S \t]*})"
        }
    )

@pytest.fixture
def local_llama_agent_fixture(config_entry, hass, enable_custom_integrations):
    with patch.object(LlamaCppAgent, '_load_icl_examples') as load_icl_examples_mock, \
         patch.object(LlamaCppAgent, '_load_grammar') as load_grammar_mock, \
         patch.object(LlamaCppAgent, 'entry', new_callable=PropertyMock) as entry_mock, \
         patch.object(LlamaCppAgent, '_async_get_exposed_entities') as get_exposed_entities_mock, \
         patch.object(APIInstance, 'async_call_tool') as call_tool_mock, \
         patch('homeassistant.helpers.template.Template') as template_mock, \
         patch('custom_components.llama_conversation.agent.importlib.import_module') as import_module_mock, \
         patch('custom_components.llama_conversation.agent.install_llama_cpp_python') as install_llama_cpp_python_mock:
        
        entry_mock.return_value = config_entry
        llama_instance_mock = MagicMock()
        llama_class_mock = MagicMock()
        llama_class_mock.return_value = llama_instance_mock
        import_module_mock.return_value = MagicMock(Llama=llama_class_mock)
        install_llama_cpp_python_mock.return_value = True
        get_exposed_entities_mock.return_value = (
            {
                "light.kitchen_light": { "state": "on" },
                "light.office_lamp": { "state": "on" },
                "switch.downstairs_hallway": { "state": "off" },
                "fan.bedroom": { "state": "on" },
            },
            ["light", "switch", "fan"]
        )
        # template_mock.side_affect = lambda template, _: jinja2.Template(template)
        generate_mock = llama_instance_mock.generate
        generate_mock.return_value = list(range(20))

        detokenize_mock = llama_instance_mock.detokenize
        detokenize_mock.return_value = ("I am saying something!\n" + json.dumps({
            "name": "HassTurnOn",
            "arguments": {
                "name": "light.kitchen_light"
            }
        })).encode()

        call_tool_mock.return_value = {"result": "success"}

        agent_obj = LlamaCppAgent(
            hass,
            config_entry
        )

        all_mocks = {
            "llama_class": llama_class_mock,
            "tokenize": llama_instance_mock.tokenize,
            "generate": generate_mock,
        }

        yield agent_obj, all_mocks

# TODO: test base llama agent (ICL loading other languages)

async def test_local_llama_agent(local_llama_agent_fixture):

    local_llama_agent: LlamaCppAgent
    all_mocks: dict[str, MagicMock]
    local_llama_agent, all_mocks = local_llama_agent_fixture
    
    # invoke the conversation agent
    conversation_id = "test-conversation"
    result = await local_llama_agent.async_process(ConversationInput(
        "turn on the kitchen lights", MagicMock(), conversation_id, None, "en"
    ))

    # assert on results + check side effects
    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["llama_class"].assert_called_once_with(
        model_path=local_llama_agent.entry.data.get(CONF_DOWNLOADED_MODEL_FILE),
        n_ctx=local_llama_agent.entry.options.get(CONF_CONTEXT_LENGTH),
        n_batch=local_llama_agent.entry.options.get(CONF_BATCH_SIZE),
        n_threads=local_llama_agent.entry.options.get(CONF_THREAD_COUNT),
        n_threads_batch=local_llama_agent.entry.options.get(CONF_BATCH_THREAD_COUNT),
        flash_attn=local_llama_agent.entry.options.get(CONF_ENABLE_FLASH_ATTENTION)
    )

    all_mocks["tokenize"].assert_called_once()
    all_mocks["generate"].assert_called_once_with(
        ANY,
        temp=local_llama_agent.entry.options.get(CONF_TEMPERATURE),
        top_k=local_llama_agent.entry.options.get(CONF_TOP_K),
        top_p=local_llama_agent.entry.options.get(CONF_TOP_P),
        typical_p=local_llama_agent.entry.options[CONF_TYPICAL_P],
        min_p=local_llama_agent.entry.options[CONF_MIN_P],
        grammar=ANY,
    )

    # reset mock stats
    for mock in all_mocks.values():
        mock.reset_mock()

    # change options then apply them
    local_llama_agent.entry.options[CONF_CONTEXT_LENGTH] = 1024
    local_llama_agent.entry.options[CONF_BATCH_SIZE] = 1024
    local_llama_agent.entry.options[CONF_THREAD_COUNT] = 24
    local_llama_agent.entry.options[CONF_BATCH_THREAD_COUNT] = 24
    local_llama_agent.entry.options[CONF_TEMPERATURE] = 2.0
    local_llama_agent.entry.options[CONF_ENABLE_FLASH_ATTENTION] = True
    local_llama_agent.entry.options[CONF_TOP_K] = 20
    local_llama_agent.entry.options[CONF_TOP_P] = 0.9
    local_llama_agent.entry.options[CONF_MIN_P] = 0.2
    local_llama_agent.entry.options[CONF_TYPICAL_P] = 0.95

    local_llama_agent._update_options()

    all_mocks["llama_class"].assert_called_once_with(
        model_path=local_llama_agent.entry.data.get(CONF_DOWNLOADED_MODEL_FILE),
        n_ctx=local_llama_agent.entry.options.get(CONF_CONTEXT_LENGTH),
        n_batch=local_llama_agent.entry.options.get(CONF_BATCH_SIZE),
        n_threads=local_llama_agent.entry.options.get(CONF_THREAD_COUNT),
        n_threads_batch=local_llama_agent.entry.options.get(CONF_BATCH_THREAD_COUNT),
        flash_attn=local_llama_agent.entry.options.get(CONF_ENABLE_FLASH_ATTENTION)
    )

    # do another turn of the same conversation
    result = await local_llama_agent.async_process(ConversationInput(
        "turn off the office lights", MagicMock(), conversation_id, None, "en"
    ))

    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["tokenize"].assert_called_once()
    all_mocks["generate"].assert_called_once_with(
        ANY,
        temp=local_llama_agent.entry.options.get(CONF_TEMPERATURE),
        top_k=local_llama_agent.entry.options.get(CONF_TOP_K),
        top_p=local_llama_agent.entry.options.get(CONF_TOP_P),
        typical_p=local_llama_agent.entry.options[CONF_TYPICAL_P],
        min_p=local_llama_agent.entry.options[CONF_MIN_P],
        grammar=ANY,
    )
    
@pytest.fixture
def ollama_agent_fixture(config_entry, hass, enable_custom_integrations):
    with patch.object(OllamaAPIAgent, '_load_icl_examples') as load_icl_examples_mock, \
         patch.object(OllamaAPIAgent, 'entry', new_callable=PropertyMock) as entry_mock, \
         patch.object(OllamaAPIAgent, '_async_get_exposed_entities') as get_exposed_entities_mock, \
         patch.object(APIInstance, 'async_call_tool') as call_tool_mock, \
         patch('homeassistant.helpers.template.Template') as template_mock, \
         patch('custom_components.llama_conversation.agent.requests.get') as requests_get_mock, \
         patch('custom_components.llama_conversation.agent.requests.post') as requests_post_mock:
        
        entry_mock.return_value = config_entry
        get_exposed_entities_mock.return_value = (
            {
                "light.kitchen_light": { "state": "on" },
                "light.office_lamp": { "state": "on" },
                "switch.downstairs_hallway": { "state": "off" },
                "fan.bedroom": { "state": "on" },
            },
            ["light", "switch", "fan"]
        )

        response_mock = MagicMock()
        response_mock.json.return_value = { "models": [ {"name": config_entry.data[CONF_CHAT_MODEL] }] }
        requests_get_mock.return_value = response_mock

        call_tool_mock.return_value = {"result": "success"}

        agent_obj = OllamaAPIAgent(
            hass,
            config_entry
        )

        all_mocks = {
            "requests_get": requests_get_mock,
            "requests_post": requests_post_mock
        }

        yield agent_obj, all_mocks

async def test_ollama_agent(ollama_agent_fixture):

    ollama_agent: OllamaAPIAgent
    all_mocks: dict[str, MagicMock]
    ollama_agent, all_mocks = ollama_agent_fixture

    all_mocks["requests_get"].assert_called_once_with(
        "http://localhost:5000/api/tags",
        headers={ "Authorization": "Bearer OpenAI-API-Key" }
    )

    response_mock = MagicMock()
    response_mock.json.return_value = {
        "model": ollama_agent.entry.data[CONF_CHAT_MODEL],
        "created_at": "2023-11-09T21:07:55.186497Z",
        "response": "I am saying something!\n" + json.dumps({
            "name": "HassTurnOn",
            "arguments": {
                "name": "light.kitchen_light"
            }
        }),
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 4648158584,
        "load_duration": 4071084,
        "prompt_eval_count": 36,
        "prompt_eval_duration": 439038000,
        "eval_count": 180,
        "eval_duration": 4196918000
    }
    all_mocks["requests_post"].return_value = response_mock
    
    # invoke the conversation agent
    conversation_id = "test-conversation"
    result = await ollama_agent.async_process(ConversationInput(
        "turn on the kitchen lights", MagicMock(), conversation_id, None, "en"
    ))

    # assert on results + check side effects
    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["requests_post"].assert_called_once_with(
        "http://localhost:5000/api/generate",
        headers={ "Authorization": "Bearer OpenAI-API-Key" },
        json={
            "model":  ollama_agent.entry.data[CONF_CHAT_MODEL],
            "stream": False,
            "keep_alive": f"{ollama_agent.entry.options[CONF_OLLAMA_KEEP_ALIVE_MIN]}m", # prevent ollama from unloading the model
            "options": {
                "num_ctx": ollama_agent.entry.options[CONF_CONTEXT_LENGTH],
                "top_p": ollama_agent.entry.options[CONF_TOP_P],
                "top_k": ollama_agent.entry.options[CONF_TOP_K],
                "typical_p": ollama_agent.entry.options[CONF_TYPICAL_P],
                "temperature": ollama_agent.entry.options[CONF_TEMPERATURE],
                "num_predict": ollama_agent.entry.options[CONF_MAX_TOKENS],
            },
            "prompt": ANY,
            "raw": True
        },
        timeout=ollama_agent.entry.options[CONF_REQUEST_TIMEOUT]
    )

    # reset mock stats
    for mock in all_mocks.values():
        mock.reset_mock()

    # change options
    ollama_agent.entry.options[CONF_CONTEXT_LENGTH] = 1024
    ollama_agent.entry.options[CONF_MAX_TOKENS] = 10
    ollama_agent.entry.options[CONF_REQUEST_TIMEOUT] = 60
    ollama_agent.entry.options[CONF_OLLAMA_KEEP_ALIVE_MIN] = 99
    ollama_agent.entry.options[CONF_REMOTE_USE_CHAT_ENDPOINT] = True
    ollama_agent.entry.options[CONF_OLLAMA_JSON_MODE] = True
    ollama_agent.entry.options[CONF_TEMPERATURE] = 2.0
    ollama_agent.entry.options[CONF_TOP_K] = 20
    ollama_agent.entry.options[CONF_TOP_P] = 0.9
    ollama_agent.entry.options[CONF_TYPICAL_P] = 0.5

    # do another turn of the same conversation
    result = await ollama_agent.async_process(ConversationInput(
        "turn off the office lights", MagicMock(), conversation_id, None, "en"
    ))

    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["requests_post"].assert_called_once_with(
        "http://localhost:5000/api/chat",
        headers={ "Authorization": "Bearer OpenAI-API-Key" },
        json={
            "model":  ollama_agent.entry.data[CONF_CHAT_MODEL],
            "stream": False,
            "format": "json",
            "keep_alive": f"{ollama_agent.entry.options[CONF_OLLAMA_KEEP_ALIVE_MIN]}m", # prevent ollama from unloading the model
            "options": {
                "num_ctx": ollama_agent.entry.options[CONF_CONTEXT_LENGTH],
                "top_p": ollama_agent.entry.options[CONF_TOP_P],
                "top_k": ollama_agent.entry.options[CONF_TOP_K],
                "typical_p": ollama_agent.entry.options[CONF_TYPICAL_P],
                "temperature": ollama_agent.entry.options[CONF_TEMPERATURE],
                "num_predict": ollama_agent.entry.options[CONF_MAX_TOKENS],
            },
            "messages": ANY
        },
        timeout=ollama_agent.entry.options[CONF_REQUEST_TIMEOUT]
    )


@pytest.fixture
def text_generation_webui_agent_fixture(config_entry, hass, enable_custom_integrations):
    with patch.object(TextGenerationWebuiAgent, '_load_icl_examples') as load_icl_examples_mock, \
         patch.object(TextGenerationWebuiAgent, 'entry', new_callable=PropertyMock) as entry_mock, \
         patch.object(TextGenerationWebuiAgent, '_async_get_exposed_entities') as get_exposed_entities_mock, \
         patch.object(APIInstance, 'async_call_tool') as call_tool_mock, \
         patch('homeassistant.helpers.template.Template') as template_mock, \
         patch('custom_components.llama_conversation.agent.requests.get') as requests_get_mock, \
         patch('custom_components.llama_conversation.agent.requests.post') as requests_post_mock:
        
        entry_mock.return_value = config_entry
        get_exposed_entities_mock.return_value = (
            {
                "light.kitchen_light": { "state": "on" },
                "light.office_lamp": { "state": "on" },
                "switch.downstairs_hallway": { "state": "off" },
                "fan.bedroom": { "state": "on" },
            },
            ["light", "switch", "fan"]
        )

        response_mock = MagicMock()
        response_mock.json.return_value = { "model_name": config_entry.data[CONF_CHAT_MODEL] }
        requests_get_mock.return_value = response_mock

        call_tool_mock.return_value = {"result": "success"}

        agent_obj = TextGenerationWebuiAgent(
            hass,
            config_entry
        )

        all_mocks = {
            "requests_get": requests_get_mock,
            "requests_post": requests_post_mock
        }

        yield agent_obj, all_mocks

async def test_text_generation_webui_agent(text_generation_webui_agent_fixture):

    text_generation_webui_agent: TextGenerationWebuiAgent
    all_mocks: dict[str, MagicMock]
    text_generation_webui_agent, all_mocks = text_generation_webui_agent_fixture

    all_mocks["requests_get"].assert_called_once_with(
        "http://localhost:5000/v1/internal/model/info",
        headers={ "Authorization": "Bearer Text-Gen-Webui-Admin-Key" }
    )

    response_mock = MagicMock()
    response_mock.json.return_value = {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "gpt-3.5-turbo-instruct",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{
            "text": "I am saying something!\n" + json.dumps({
                "name": "HassTurnOn",
                "arguments": {
                    "name": "light.kitchen_light"
                }
            }),
            "index": 0,
            "logprobs": None,
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12
        }
    }
    all_mocks["requests_post"].return_value = response_mock
    
    # invoke the conversation agent
    conversation_id = "test-conversation"
    result = await text_generation_webui_agent.async_process(ConversationInput(
        "turn on the kitchen lights", MagicMock(), conversation_id, None, "en"
    ))

    # assert on results + check side effects
    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["requests_post"].assert_called_once_with(
        "http://localhost:5000/v1/completions",
        json={
            "model":  text_generation_webui_agent.entry.data[CONF_CHAT_MODEL],
            "top_p": text_generation_webui_agent.entry.options[CONF_TOP_P],
            "top_k": text_generation_webui_agent.entry.options[CONF_TOP_K],
            "temperature": text_generation_webui_agent.entry.options[CONF_TEMPERATURE],
            "min_p": text_generation_webui_agent.entry.options[CONF_MIN_P],
            "typical_p": text_generation_webui_agent.entry.options[CONF_TYPICAL_P],
            "truncation_length": text_generation_webui_agent.entry.options[CONF_CONTEXT_LENGTH],
            "max_tokens": text_generation_webui_agent.entry.options[CONF_MAX_TOKENS],
            "prompt": ANY
        },
        headers={ "Authorization": "Bearer OpenAI-API-Key" },
        timeout=text_generation_webui_agent.entry.options[CONF_REQUEST_TIMEOUT]
    )

    # reset mock stats
    for mock in all_mocks.values():
        mock.reset_mock()

    text_generation_webui_agent.entry.options[CONF_TEXT_GEN_WEBUI_PRESET] = "Some Preset"

    # do another turn of the same conversation and use a preset
    result = await text_generation_webui_agent.async_process(ConversationInput(
        "turn off the office lights", MagicMock(), conversation_id, None, "en"
    ))

    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["requests_post"].assert_called_once_with(
        "http://localhost:5000/v1/completions",
        json={
            "model":  text_generation_webui_agent.entry.data[CONF_CHAT_MODEL],
            "top_p": text_generation_webui_agent.entry.options[CONF_TOP_P],
            "top_k": text_generation_webui_agent.entry.options[CONF_TOP_K],
            "temperature": text_generation_webui_agent.entry.options[CONF_TEMPERATURE],
            "min_p": text_generation_webui_agent.entry.options[CONF_MIN_P],
            "typical_p": text_generation_webui_agent.entry.options[CONF_TYPICAL_P],
            "truncation_length": text_generation_webui_agent.entry.options[CONF_CONTEXT_LENGTH],
            "max_tokens": text_generation_webui_agent.entry.options[CONF_MAX_TOKENS],
            "preset": "Some Preset",
            "prompt": ANY
        },
        headers={ "Authorization": "Bearer OpenAI-API-Key" },
        timeout=text_generation_webui_agent.entry.options[CONF_REQUEST_TIMEOUT]
    )

    # change options
    text_generation_webui_agent.entry.options[CONF_MAX_TOKENS] = 10
    text_generation_webui_agent.entry.options[CONF_REQUEST_TIMEOUT] = 60
    text_generation_webui_agent.entry.options[CONF_REMOTE_USE_CHAT_ENDPOINT] = True
    text_generation_webui_agent.entry.options[CONF_TEMPERATURE] = 2.0
    text_generation_webui_agent.entry.options[CONF_TOP_P] = 0.9
    text_generation_webui_agent.entry.options[CONF_MIN_P] = 0.2
    text_generation_webui_agent.entry.options[CONF_TYPICAL_P] = 0.95
    text_generation_webui_agent.entry.options[CONF_TEXT_GEN_WEBUI_PRESET] = ""

    response_mock.json.return_value = {
        "id": "chatcmpl-123",
        # text-gen-webui has a typo where it is 'chat.completions' not 'chat.completion'
        "object": "chat.completions",
        "created": 1677652288,
        "model": text_generation_webui_agent.entry.data[CONF_CHAT_MODEL],
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I am saying something!\n" + json.dumps({
                    "name": "HassTurnOn",
                    "arguments": {
                        "name": "light.kitchen_light"
                    }
                }),
            },
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }

    # reset mock stats
    for mock in all_mocks.values():
        mock.reset_mock()

    # do another turn of the same conversation but the chat endpoint
    result = await text_generation_webui_agent.async_process(ConversationInput(
        "turn off the office lights", MagicMock(), conversation_id, None, "en"
    ))

    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["requests_post"].assert_called_once_with(
        "http://localhost:5000/v1/chat/completions",
        json={
            "model":  text_generation_webui_agent.entry.data[CONF_CHAT_MODEL],
            "top_p": text_generation_webui_agent.entry.options[CONF_TOP_P],
            "top_k": text_generation_webui_agent.entry.options[CONF_TOP_K],
            "temperature": text_generation_webui_agent.entry.options[CONF_TEMPERATURE],
            "min_p": text_generation_webui_agent.entry.options[CONF_MIN_P],
            "typical_p": text_generation_webui_agent.entry.options[CONF_TYPICAL_P],
            "truncation_length": text_generation_webui_agent.entry.options[CONF_CONTEXT_LENGTH],
            "max_tokens": text_generation_webui_agent.entry.options[CONF_MAX_TOKENS],
            "mode": text_generation_webui_agent.entry.options[CONF_TEXT_GEN_WEBUI_CHAT_MODE],
            "messages": ANY
        },
        headers={ "Authorization": "Bearer OpenAI-API-Key" },
        timeout=text_generation_webui_agent.entry.options[CONF_REQUEST_TIMEOUT]
    )

    # reset mock stats
    for mock in all_mocks.values():
        mock.reset_mock()

    text_generation_webui_agent.entry.options[CONF_TEXT_GEN_WEBUI_PRESET] = "Some Character"

    # do another turn of the same conversation and use a preset
    result = await text_generation_webui_agent.async_process(ConversationInput(
        "turn off the office lights", MagicMock(), conversation_id, None, "en"
    ))

    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["requests_post"].assert_called_once_with(
        "http://localhost:5000/v1/chat/completions",
        json={
            "model":  text_generation_webui_agent.entry.data[CONF_CHAT_MODEL],
            "top_p": text_generation_webui_agent.entry.options[CONF_TOP_P],
            "top_k": text_generation_webui_agent.entry.options[CONF_TOP_K],
            "temperature": text_generation_webui_agent.entry.options[CONF_TEMPERATURE],
            "min_p": text_generation_webui_agent.entry.options[CONF_MIN_P],
            "typical_p": text_generation_webui_agent.entry.options[CONF_TYPICAL_P],
            "truncation_length": text_generation_webui_agent.entry.options[CONF_CONTEXT_LENGTH],
            "max_tokens": text_generation_webui_agent.entry.options[CONF_MAX_TOKENS],
            "mode": text_generation_webui_agent.entry.options[CONF_TEXT_GEN_WEBUI_CHAT_MODE],
            "character": "Some Character",
            "messages": ANY
        },
        headers={ "Authorization": "Bearer OpenAI-API-Key" },
        timeout=text_generation_webui_agent.entry.options[CONF_REQUEST_TIMEOUT]
    )

    # reset mock stats
    for mock in all_mocks.values():
        mock.reset_mock()

    text_generation_webui_agent.entry.options[CONF_TEXT_GEN_WEBUI_CHAT_MODE] = TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT

    # do another turn of the same conversation and use instruct mode
    result = await text_generation_webui_agent.async_process(ConversationInput(
        "turn off the office lights", MagicMock(), conversation_id, None, "en"
    ))

    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["requests_post"].assert_called_once_with(
        "http://localhost:5000/v1/chat/completions",
        json={
            "model":  text_generation_webui_agent.entry.data[CONF_CHAT_MODEL],
            "top_p": text_generation_webui_agent.entry.options[CONF_TOP_P],
            "top_k": text_generation_webui_agent.entry.options[CONF_TOP_K],
            "min_p": text_generation_webui_agent.entry.options[CONF_MIN_P],
            "typical_p": text_generation_webui_agent.entry.options[CONF_TYPICAL_P],
            "temperature": text_generation_webui_agent.entry.options[CONF_TEMPERATURE],
            "truncation_length": text_generation_webui_agent.entry.options[CONF_CONTEXT_LENGTH],
            "max_tokens": text_generation_webui_agent.entry.options[CONF_MAX_TOKENS],
            "mode": text_generation_webui_agent.entry.options[CONF_TEXT_GEN_WEBUI_CHAT_MODE],
            "instruction_template": "chatml",
            "messages": ANY
        },
        headers={ "Authorization": "Bearer OpenAI-API-Key" },
        timeout=text_generation_webui_agent.entry.options[CONF_REQUEST_TIMEOUT]
    )

@pytest.fixture
def generic_openai_agent_fixture(config_entry, hass, enable_custom_integrations):
    with patch.object(GenericOpenAIAPIAgent, '_load_icl_examples') as load_icl_examples_mock, \
         patch.object(GenericOpenAIAPIAgent, 'entry', new_callable=PropertyMock) as entry_mock, \
         patch.object(GenericOpenAIAPIAgent, '_async_get_exposed_entities') as get_exposed_entities_mock, \
         patch.object(APIInstance, 'async_call_tool') as call_tool_mock, \
         patch('homeassistant.helpers.template.Template') as template_mock, \
         patch('custom_components.llama_conversation.agent.requests.get') as requests_get_mock, \
         patch('custom_components.llama_conversation.agent.requests.post') as requests_post_mock:
        
        entry_mock.return_value = config_entry
        get_exposed_entities_mock.return_value = (
            {
                "light.kitchen_light": { "state": "on" },
                "light.office_lamp": { "state": "on" },
                "switch.downstairs_hallway": { "state": "off" },
                "fan.bedroom": { "state": "on" },
            },
            ["light", "switch", "fan"]
        )

        call_tool_mock.return_value = {"result": "success"}

        agent_obj = GenericOpenAIAPIAgent(
            hass,
            config_entry
        )

        all_mocks = {
            "requests_get": requests_get_mock,
            "requests_post": requests_post_mock
        }

        yield agent_obj, all_mocks

async def test_generic_openai_agent(generic_openai_agent_fixture):

    generic_openai_agent: TextGenerationWebuiAgent
    all_mocks: dict[str, MagicMock]
    generic_openai_agent, all_mocks = generic_openai_agent_fixture

    response_mock = MagicMock()
    response_mock.json.return_value = {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "gpt-3.5-turbo-instruct",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{
            "text": "I am saying something!\n" + json.dumps({
                "name": "HassTurnOn",
                "arguments": {
                    "name": "light.kitchen_light"
                }
            }),
            "index": 0,
            "logprobs": None,
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12
        }
    }
    all_mocks["requests_post"].return_value = response_mock
    
    # invoke the conversation agent
    conversation_id = "test-conversation"
    result = await generic_openai_agent.async_process(ConversationInput(
        "turn on the kitchen lights", MagicMock(), conversation_id, None, "en"
    ))

    # assert on results + check side effects
    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["requests_post"].assert_called_once_with(
        "http://localhost:5000/v1/completions",
        json={
            "model":  generic_openai_agent.entry.data[CONF_CHAT_MODEL],
            "top_p": generic_openai_agent.entry.options[CONF_TOP_P],
            "temperature": generic_openai_agent.entry.options[CONF_TEMPERATURE],
            "max_tokens": generic_openai_agent.entry.options[CONF_MAX_TOKENS],
            "prompt": ANY
        },
        headers={ "Authorization": "Bearer OpenAI-API-Key" },
        timeout=generic_openai_agent.entry.options[CONF_REQUEST_TIMEOUT]
    )

    # reset mock stats
    for mock in all_mocks.values():
        mock.reset_mock()

    # change options
    generic_openai_agent.entry.options[CONF_MAX_TOKENS] = 10
    generic_openai_agent.entry.options[CONF_REQUEST_TIMEOUT] = 60
    generic_openai_agent.entry.options[CONF_REMOTE_USE_CHAT_ENDPOINT] = True
    generic_openai_agent.entry.options[CONF_TEMPERATURE] = 2.0
    generic_openai_agent.entry.options[CONF_TOP_P] = 0.9

    response_mock.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": generic_openai_agent.entry.data[CONF_CHAT_MODEL],
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I am saying something!\n" + json.dumps({
                    "name": "HassTurnOn",
                    "arguments": {
                        "name": "light.kitchen_light"
                    }
                }),
            },
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }

    # do another turn of the same conversation but the chat endpoint
    result = await generic_openai_agent.async_process(ConversationInput(
        "turn off the office lights", MagicMock(), conversation_id, None, "en"
    ))

    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["requests_post"].assert_called_once_with(
        "http://localhost:5000/v1/chat/completions",
        json={
            "model":  generic_openai_agent.entry.data[CONF_CHAT_MODEL],
            "top_p": generic_openai_agent.entry.options[CONF_TOP_P],
            "temperature": generic_openai_agent.entry.options[CONF_TEMPERATURE],
            "max_tokens": generic_openai_agent.entry.options[CONF_MAX_TOKENS],
            "messages": ANY
        },
        headers={ "Authorization": "Bearer OpenAI-API-Key" },
        timeout=generic_openai_agent.entry.options[CONF_REQUEST_TIMEOUT]
    )