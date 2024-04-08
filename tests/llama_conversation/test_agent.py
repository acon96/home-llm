import json
import logging
import pytest
import jinja2
from unittest.mock import patch, MagicMock, PropertyMock, AsyncMock, ANY

from custom_components.llama_conversation.agent import LocalLLaMAAgent
from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_REQUEST_TIMEOUT,
    CONF_BACKEND_TYPE,
    CONF_DOWNLOADED_MODEL_FILE,
    CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
    CONF_ALLOWED_SERVICE_CALL_ARGUMENTS,
    CONF_PROMPT_TEMPLATE,
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
    NO_ICL_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_BACKEND_TYPE,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
    DEFAULT_ALLOWED_SERVICE_CALL_ARGUMENTS,
    DEFAULT_PROMPT_TEMPLATE,
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

import homeassistant.helpers.template
from homeassistant.components.conversation import ConversationInput
from homeassistant.const import (
    CONF_HOST,
    CONF_PORT,
    CONF_SSL
)

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
    

# @pytest.fixture
# def patch_dependency_group1():
#     with patch('path.to.dependency1') as mock1, \
#          patch('path.to.dependency2') as mock2:
#         yield mock1, mock2

# @pytest.fixture
# def patch_dependency_group2():
#     with patch('path.to.dependency3') as mock3, \
#          patch('path.to.dependency4') as mock4:
#         yield mock3, mock4
        

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
            CONF_PROMPT: NO_ICL_PROMPT,
        }
    )

@pytest.fixture
def home_assistant_mock():
    mock_home_assistant = MagicMock()
    async def call_now(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_home_assistant.async_add_executor_job.side_effect = call_now
    mock_home_assistant.services.async_call = AsyncMock()

    yield mock_home_assistant

@pytest.fixture
def local_llama_agent_fixture(config_entry, home_assistant_mock):
    with patch.object(LocalLLaMAAgent, '_load_icl_examples') as load_icl_examples_mock, \
         patch.object(LocalLLaMAAgent, '_load_grammar') as load_grammar_mock, \
         patch.object(LocalLLaMAAgent, 'entry', new_callable=PropertyMock) as entry_mock, \
         patch.object(LocalLLaMAAgent, '_async_get_exposed_entities') as get_exposed_entities_mock, \
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
        detokenize_mock.return_value = json.dumps({
            "to_say": "I am saying something!",
            "service": "light.turn_on",
            "target_device": "light.kitchen_light",
        }).encode()

        agent_obj = LocalLLaMAAgent(
            home_assistant_mock,
            config_entry
        )

        all_mocks = {
            "llama_class": llama_class_mock,
            "tokenize": llama_instance_mock.tokenize,
            "generate": generate_mock,
        }

        yield agent_obj, all_mocks


@pytest.mark.asyncio  # This decorator is necessary for pytest to run async test functions
async def test_local_llama_agent(local_llama_agent_fixture):

    local_llama_agent: LocalLLaMAAgent
    all_mocks: dict[str, MagicMock]
    local_llama_agent, all_mocks = local_llama_agent_fixture
    
    conversation_id = "test-conversation"
    result = await local_llama_agent.async_process(ConversationInput(
        "turn on the kitchen lights", MagicMock(), conversation_id, None, "en"
    ))

    assert result.response.speech['plain']['speech'] == "I am saying something!"

    all_mocks["llama_class"].assert_called_once_with(
        model_path=ANY,
        n_ctx=ANY,
        n_batch=ANY,
        n_threads=ANY,
        n_threads_batch=ANY,
    )

    all_mocks["tokenize"].assert_called_once()
    all_mocks["generate"].assert_called_once_with(
        ANY,
        temp=ANY,
        top_k=ANY,
        top_p=ANY,
        grammar=ANY,
    )