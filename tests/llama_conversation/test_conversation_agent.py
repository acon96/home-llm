"""Tests for the conversation agent.

These drive the agent through Home Assistant's own conversation flow (async_process creates the chat
session and the chat log), so the chat log the integration talks to is the real one.
"""

import dataclasses
from types import MappingProxyType
from typing import Any

import pytest
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AssistantContent,
    ConversationInput,
    SystemContent,
    ToolResultContent,
    UserContent,
)
from homeassistant.components.conversation.chat_log import DATA_CHAT_LOGS
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import intent, llm, template

from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_PROMPT,
    CONF_CONTEXT_LENGTH,
    CONF_MAX_TOKENS,
    DOMAIN,
)
from custom_components.llama_conversation.conversation import NO_SPEECH_MESSAGE, LocalLLMAgent
from custom_components.llama_conversation.utils import MalformedToolCallException

API_ID = "test-llm-api"
API_INSTRUCTIONS = "Call the tools to control the house."
CLIENT_PROMPT = "SYSTEM PROMPT"
CONVERSATION_ID = "test-conversation"


class RecordingTool(llm.Tool):
    """Tool that records every call the chat log makes on the model's behalf."""

    name = "test_tool"
    description = "A test tool"
    parameters = vol.Schema({})

    def __init__(self, result: Any = None) -> None:
        self.calls: list[llm.ToolInput] = []
        self._result = result if result is not None else {"result": "ok"}

    async def async_call(self, hass, tool_input, llm_context):
        self.calls.append(tool_input)
        return self._result


class StubAPI(llm.API):
    """Minimal LLM API the agent can resolve by id."""

    def __init__(self, hass: HomeAssistant, tools: list[llm.Tool] | None = None) -> None:
        super().__init__(hass=hass, id=API_ID, name="Stub API")
        self.tools = list(tools or [])

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        return llm.APIInstance(
            api=self,
            api_prompt=API_INSTRUCTIONS,
            llm_context=llm_context,
            tools=self.tools,
        )


class ScriptedClient:
    """Stands in for a backend client by replaying assistant messages for each generation."""

    def __init__(self, hass: HomeAssistant, replies: list[list[Any]] | None = None) -> None:
        self.hass = hass
        self.replies = (
            replies if replies is not None else [[{"role": "assistant", "content": "hello from llm"}]]
        )
        self.prompt_calls: list[tuple[str, llm.APIInstance | None]] = []
        self.seen_conversations: list[list[conversation.Content]] = []

    def _generate_system_prompt(self, prompt_template, llm_api, entity_options):
        """Render the prompt the way the real client does, so template variables are exercised."""
        self.prompt_calls.append((prompt_template, llm_api))
        return template.Template(prompt_template, self.hass).async_render(
            {"llm_api_instructions": llm_api.api_prompt if llm_api else ""},
            parse_result=False,
        )

    def _count_tokens(self, content, entity_options):
        """Estimate like a backend without a tokenizer, which is what the real base class does too."""
        characters = sum(len(str(getattr(item, "content", "") or "")) for item in content)
        return characters // 4

    async def _async_generate(self, conv, agent_id, chat_log, entity_options):
        self.seen_conversations.append(list(conv))
        return chat_log.async_add_delta_content_stream(agent_id, _replay(self.replies.pop(0)))


async def _replay(messages: list[Any]):
    """Yield scripted chat log deltas, raising for any exception placed in the script."""
    for message in messages:
        if isinstance(message, Exception):
            raise message
        yield message


class DummyEntry:
    _next_id = 0

    def __init__(self, subentry: ConfigSubentry, options: dict[str, Any] | None = None) -> None:
        DummyEntry._next_id += 1
        self.entry_id = f"entry{DummyEntry._next_id}"
        self.options = options or {}
        self.subentries = {subentry.subentry_id: subentry}
        self.runtime_data = None

    def add_update_listener(self, _cb):
        return lambda: None


def make_agent(
    hass: HomeAssistant,
    client: ScriptedClient,
    subentry_data: dict[str, Any] | None = None,
) -> LocalLLMAgent:
    """Build an agent without running the config entry setup."""
    subentry = ConfigSubentry(
        data=MappingProxyType({CONF_CHAT_MODEL: "model", CONF_PROMPT: CLIENT_PROMPT, **(subentry_data or {})}),
        subentry_id="sub1",
        subentry_type="conversation",
        title="Test Agent",
        unique_id=None,
    )
    entry = DummyEntry(subentry)
    entry.runtime_data = client
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry
    return LocalLLMAgent(hass, entry, subentry, client)


def make_user_input(
    text: str = "turn on the lights",
    conversation_id: str | None = CONVERSATION_ID,
    **kwargs: Any,
) -> ConversationInput:
    return ConversationInput(
        text=text,
        context=kwargs.get("context"),
        conversation_id=conversation_id,
        device_id=kwargs.get("device_id"),
        satellite_id=kwargs.get("satellite_id"),
        language=kwargs.get("language", "en"),
        agent_id=kwargs.get("agent_id", "conversation.test_agent"),
        extra_system_prompt=kwargs.get("extra_system_prompt"),
    )


def register_api(hass: HomeAssistant, tools: list[llm.Tool] | None = None) -> StubAPI:
    """Register the stub LLM API. Its tool list stays editable through the returned object."""
    api = StubAPI(hass, tools)
    llm.async_register_api(hass, api)
    return api


def get_log(hass: HomeAssistant) -> conversation.ChatLog:
    return hass.data[DATA_CHAT_LOGS][CONVERSATION_ID]


async def seed(hass: HomeAssistant, *contents: conversation.Content) -> None:
    """Add content to the open chat log the way Home Assistant would."""
    log = get_log(hass)
    for content in contents:
        if isinstance(content, UserContent):
            log.async_add_user_content(content)
        else:
            log.async_add_assistant_content_without_tools(content)


def roles(content: list[conversation.Content]) -> list[str]:
    return [type(item).__name__ for item in content]


def set_options(agent: LocalLLMAgent, **options: Any) -> None:
    """Change the runtime options of an agent mid conversation."""
    hass_data = agent.hass.data[DOMAIN]
    hass_data[agent.entry_id].options.update(options)


def save_subentry_options(agent: LocalLLMAgent, **options: Any) -> None:
    """Replace what the subentry holds, the way saving the model options does."""
    subentry = agent.subentry
    data = dict(subentry.data)
    data.update(options)
    agent.entry.subentries[subentry.subentry_id] = dataclasses.replace(subentry, data=MappingProxyType(data))


@pytest.mark.asyncio
async def test_answers_and_records_the_turn_in_the_chat_log(hass: HomeAssistant):
    client = ScriptedClient(hass)
    agent = make_agent(hass, client)

    result = await agent.async_process(make_user_input())

    assert result.response.speech["plain"]["speech"] == "hello from llm"
    assert result.conversation_id == CONVERSATION_ID
    assert roles(get_log(hass).content) == ["SystemContent", "UserContent", "AssistantContent"]
    assert get_log(hass).content[0].content.startswith(CLIENT_PROMPT)
    # the model gets the system prompt plus the message it is answering
    assert roles(client.seen_conversations[0]) == ["SystemContent", "UserContent"]
    assert client.seen_conversations[0][1].content == "turn on the lights"
    assert client.prompt_calls[0][0] == CLIENT_PROMPT
    assert agent.supported_languages == MATCH_ALL


@pytest.mark.asyncio
async def test_new_conversation_keeps_its_generated_id(hass: HomeAssistant):
    client = ScriptedClient(hass)
    agent = make_agent(hass, client)

    result = await agent.async_process(make_user_input(conversation_id=None))

    assert result.conversation_id is not None
    assert hass.data[DATA_CHAT_LOGS][result.conversation_id].content[-1].content == "hello from llm"


@pytest.mark.asyncio
async def test_new_conversation_id_survives_an_error(hass: HomeAssistant):
    class FailingClient(ScriptedClient):
        async def _async_generate(self, conv, agent_id, chat_log, entity_options):
            raise RuntimeError("backend offline")

    agent = make_agent(hass, FailingClient(hass))

    result = await agent.async_process(make_user_input(conversation_id=None))

    assert result.conversation_id is not None
    assert result.response.as_dict()["data"]["code"] == intent.IntentResponseErrorCode.FAILED_TO_HANDLE.value


@pytest.mark.asyncio
async def test_conversation_is_sent_in_full(hass: HomeAssistant):
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "content": "first answer"}],
        [{"role": "assistant", "content": "second answer"}],
    ])
    agent = make_agent(hass, client)

    await agent.async_process(make_user_input("what is the weather"))
    result = await agent.async_process(make_user_input("turn on the lights"))

    assert result.response.speech["plain"]["speech"] == "second answer"
    assert roles(client.seen_conversations[-1]) == ["SystemContent", "UserContent", "AssistantContent", "UserContent"]
    assert client.seen_conversations[-1][-1].content == "turn on the lights"


@pytest.mark.asyncio
async def test_history_is_trimmed_to_fit_the_model_context(hass: HomeAssistant):
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "content": "r_turn1"}],
        [{"role": "assistant", "content": "r_turn2"}],
    ])
    # no room left after the reply is accounted for, so only the message being answered fits
    agent = make_agent(hass, client, {CONF_CONTEXT_LENGTH: 1024, CONF_MAX_TOKENS: 1024})

    await agent.async_process(make_user_input("u1"))
    await seed(hass, UserContent(content="u2"), AssistantContent(agent_id="agent", content="r_seeded"))
    result = await agent.async_process(make_user_input("u3"))

    assert result.response.speech["plain"]["speech"] == "r_turn2"
    history = client.seen_conversations[-1]
    assert roles(history) == ["SystemContent", "UserContent"]
    assert history[-1].content == "u3"
    # the log keeps everything, trimming only applies to what the model is shown
    assert len([c for c in get_log(hass).content if isinstance(c, UserContent)]) == 3


@pytest.mark.asyncio
async def test_trimming_keeps_a_tool_call_with_its_results(hass: HomeAssistant):
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "content": "u1 answer"}],
        [{"role": "assistant", "content": "doing it", "tool_calls": [llm.ToolInput("test_tool", {})]}],
        [{"role": "assistant", "content": "done"}],
    ])
    # enough context for the last exchange only, which is the one holding the tool call
    agent = make_agent(hass, client, {
        CONF_LLM_HASS_API: [API_ID],
        CONF_MAX_TOOL_CALL_ITERATIONS: 1,
        CONF_CONTEXT_LENGTH: 1024,
        CONF_MAX_TOKENS: 900,
    })
    register_api(hass, [RecordingTool()])

    await agent.async_process(make_user_input("u1"))
    await agent.async_process(make_user_input("u2"))

    history = client.seen_conversations[-1]
    assert isinstance(history[0], SystemContent)
    assert not isinstance(history[1], ToolResultContent), history
    assert history[-1].content == "u2"


@pytest.mark.asyncio
async def test_system_prompt_is_rendered_once_per_conversation(hass: HomeAssistant):
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "content": "a1"}],
        [{"role": "assistant", "content": "a2"}],
    ])
    agent = make_agent(hass, client)

    await agent.async_process(make_user_input("u1"))
    await agent.async_process(make_user_input("u2"))
    await agent.async_process(make_user_input(conversation_id=None))

    # two turns of one conversation, then a conversation of its own
    assert len(client.prompt_calls) == 2
    assert get_log(hass).content[0].content.startswith(CLIENT_PROMPT)


@pytest.mark.asyncio
async def test_a_changed_prompt_template_is_rendered_again(hass: HomeAssistant):
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "content": "a1"}],
        [{"role": "assistant", "content": "a2"}],
    ])
    agent = make_agent(hass, client)

    await agent.async_process(make_user_input("u1"))
    save_subentry_options(agent, **{CONF_PROMPT: "A DIFFERENT SYSTEM PROMPT"})
    await agent.async_process(make_user_input("u2"))

    assert len(client.prompt_calls) == 2
    assert client.prompt_calls[1][0] == "A DIFFERENT SYSTEM PROMPT"


@pytest.mark.asyncio
async def test_satellite_system_prompt_is_appended(hass: HomeAssistant):
    agent = make_agent(hass, ScriptedClient(hass))

    await agent.async_process(make_user_input(extra_system_prompt="Answer in one short sentence."))

    assert "Answer in one short sentence." in get_log(hass).content[0].content


@pytest.mark.asyncio
async def test_api_instructions_are_left_to_the_prompt_template(hass: HomeAssistant):
    """Home Assistant's tool instructions only reach the model where the prompt asks for them."""
    register_api(hass, [RecordingTool()])

    plain_client = ScriptedClient(hass)
    await make_agent(hass, plain_client, {CONF_LLM_HASS_API: [API_ID]}).async_process(make_user_input("u1"))

    log = get_log(hass)
    # the resolved instance still carries them, so a template can place them wherever it wants
    assert plain_client.prompt_calls[0][1].api_prompt == API_INSTRUCTIONS
    assert log.llm_api is not None
    assert [tool.name for tool in log.llm_api.tools] == ["test_tool"]
    # but Home Assistant does not append them behind our back
    assert log.content[0].content.startswith(CLIENT_PROMPT)
    assert API_INSTRUCTIONS not in log.content[0].content

    opting_client = ScriptedClient(hass)
    agent = make_agent(hass, opting_client, {
        CONF_LLM_HASS_API: [API_ID],
        CONF_PROMPT: "My prompt with {{ llm_api_instructions }} instructions",
    })
    await agent.async_process(make_user_input("u2"))

    assert get_log(hass).content[0].content.startswith(f"My prompt with {API_INSTRUCTIONS} instructions")


@pytest.mark.asyncio
async def test_home_assistant_only_adds_its_own_date_time_prompt_when_uncovered(hass: HomeAssistant):
    """Home Assistant appends its date/time prompt unless a tool already covers the date and time.

    Home LLM puts the date into the prompt with Jinja, so this documents the one piece of text Home
    Assistant still appends: it appears when no tool named *GetDateTime is offered, and never with the
    Assist API (which always exposes GetDateTime).
    """
    class DateTimeTool(llm.Tool):
        name = "GetDateTime"
        description = "Get the current date and time"
        parameters = vol.Schema({})

        async def async_call(self, hass, tool_input, llm_context):
            return {"result": "now"}

    answer = [{"role": "assistant", "content": "ok"}]
    no_tools = ScriptedClient(hass, replies=[answer])
    await make_agent(hass, no_tools).async_process(make_user_input("no tools at all"))
    assert "Current time is" in get_log(hass).content[0].content

    client = ScriptedClient(hass, replies=[answer, answer])
    agent = make_agent(hass, client, {CONF_LLM_HASS_API: [API_ID]})
    api = register_api(hass, [RecordingTool()])
    await agent.async_process(make_user_input("a tool that is not the date"))
    assert "Current time is" in get_log(hass).content[0].content

    api.tools[:] = [DateTimeTool()]
    await agent.async_process(make_user_input("a date time tool"))
    assert "Current time is" not in get_log(hass).content[0].content


@pytest.mark.asyncio
async def test_prompt_template_error(hass: HomeAssistant):
    class ExplodingClient(ScriptedClient):
        def _generate_system_prompt(self, prompt_template, llm_api, entity_options):
            raise TemplateError("bad template")

    agent = make_agent(hass, ExplodingClient(hass))

    result = await agent.async_process(make_user_input())

    payload = result.response.as_dict()
    assert payload["response_type"] == intent.IntentResponseType.ERROR.value
    assert payload["data"]["code"] == intent.IntentResponseErrorCode.UNKNOWN.value
    assert payload["speech"]["plain"]["speech"] == "Sorry, I had a problem with my template: bad template"


@pytest.mark.asyncio
async def test_backend_error(hass: HomeAssistant):
    class FailingClient(ScriptedClient):
        async def _async_generate(self, conv, agent_id, chat_log, entity_options):
            raise RuntimeError("backend offline")

    agent = make_agent(hass, FailingClient(hass))

    result = await agent.async_process(make_user_input())

    payload = result.response.as_dict()
    assert payload["data"]["code"] == intent.IntentResponseErrorCode.FAILED_TO_HANDLE.value
    assert "backend offline" in payload["speech"]["plain"]["speech"]


@pytest.mark.asyncio
async def test_unknown_llm_api_error(hass: HomeAssistant):
    agent = make_agent(hass, ScriptedClient(hass), {CONF_LLM_HASS_API: ["missing-api"]})

    result = await agent.async_process(make_user_input())

    payload = result.response.as_dict()
    assert payload["data"]["code"] == intent.IntentResponseErrorCode.UNKNOWN.value
    assert payload["speech"]["plain"]["speech"].startswith("Error preparing LLM API")


@pytest.mark.asyncio
async def test_malformed_tool_call_asks_the_model_to_correct_itself(hass: HomeAssistant):
    client = ScriptedClient(hass, replies=[
        [MalformedToolCallException("agent", "id", "unknown", "{bad", "bad json")],
        [{"role": "assistant", "content": "recovered response"}],
    ])
    agent = make_agent(hass, client, {CONF_MAX_TOOL_CALL_ITERATIONS: 2})

    result = await agent.async_process(make_user_input())

    assert result.response.speech["plain"]["speech"] == "recovered response"
    assert len(client.replies) == 0
    # the correction is scaffolding for the model, so it stays out of the chat log
    assert [c for c in get_log(hass).content if isinstance(c, ToolResultContent)] == []
    correction = client.seen_conversations[-1]
    assert isinstance(correction[-2], AssistantContent)
    assert isinstance(correction[-1], ToolResultContent)


@pytest.mark.asyncio
async def test_tool_calls_run_through_the_chat_log(hass: HomeAssistant):
    tool = RecordingTool()
    register_api(hass, [tool])
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "tool_calls": [llm.ToolInput("test_tool", {"arg": 1})]}],
        [{"role": "assistant", "content": "the light is on now"}],
    ])
    agent = make_agent(hass, client, {CONF_LLM_HASS_API: [API_ID], CONF_MAX_TOOL_CALL_ITERATIONS: 3})

    result = await agent.async_process(make_user_input("turn on the lights"))

    assert result.response.speech["plain"]["speech"] == "the light is on now"
    assert [call.tool_name for call in tool.calls] == ["test_tool"]
    assert tool.calls[0].tool_args == {"arg": 1}
    log = get_log(hass)
    assert roles(log.content) == [
        "SystemContent", "UserContent", "AssistantContent", "ToolResultContent", "AssistantContent",
    ]
    # the second generation gets to see what the tool returned
    assert isinstance(client.seen_conversations[-1][-1], ToolResultContent)
    assert log.llm_input_provided_index == 2


@pytest.mark.asyncio
async def test_running_out_of_tool_call_iterations(hass: HomeAssistant):
    tool = RecordingTool()
    register_api(hass, [tool])
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "tool_calls": [llm.ToolInput("test_tool", {})]}],
        [{"role": "assistant", "tool_calls": [llm.ToolInput("test_tool", {})]}],
    ])
    agent = make_agent(hass, client, {CONF_LLM_HASS_API: [API_ID], CONF_MAX_TOOL_CALL_ITERATIONS: 2})

    result = await agent.async_process(make_user_input())

    payload = result.response.as_dict()
    assert payload["data"]["code"] == intent.IntentResponseErrorCode.FAILED_TO_HANDLE.value
    assert payload["speech"]["plain"]["speech"] == "Sorry, I ran out of attempts to handle your request"
    assert len(tool.calls) == 2


@pytest.mark.asyncio
async def test_single_shot_model_answers_with_its_tool_call(hass: HomeAssistant):
    tool = RecordingTool()
    register_api(hass, [tool])
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "content": "turning on the light", "tool_calls": [llm.ToolInput("test_tool", {})]}],
    ])
    agent = make_agent(hass, client, {CONF_LLM_HASS_API: [API_ID], CONF_MAX_TOOL_CALL_ITERATIONS: 0})

    result = await agent.async_process(make_user_input())

    # the turn ends on the tool result, and there is nothing left to generate
    assert isinstance(get_log(hass).content[-1], ToolResultContent)
    assert result.response.speech["plain"]["speech"] == "turning on the light"
    assert len(tool.calls) == 1
    assert len(client.seen_conversations) == 1


@pytest.mark.asyncio
async def test_a_tool_that_answered_is_allowed_to_speak(hass: HomeAssistant):
    """A script or action answering with its own text should be heard, not silenced."""
    answer = intent.IntentResponse(language="en")
    answer.async_set_speech("the garage door is opening")
    register_api(hass, [RecordingTool(llm.IntentResponseDict(answer))])
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "tool_calls": [llm.ToolInput("test_tool", {})]}],
    ])
    agent = make_agent(hass, client, {CONF_LLM_HASS_API: [API_ID], CONF_MAX_TOOL_CALL_ITERATIONS: 0})

    result = await agent.async_process(make_user_input())

    assert result.response.speech["plain"]["speech"] == "the garage door is opening"


@pytest.mark.asyncio
async def test_tool_response_is_passed_through(hass: HomeAssistant):
    response = intent.IntentResponse(language="en")
    response.response_type = intent.IntentResponseType.QUERY_ANSWER
    response.async_set_card("Changes", "Living room light")
    tool = RecordingTool(llm.IntentResponseDict(response))
    register_api(hass, [tool])
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "tool_calls": [llm.ToolInput("test_tool", {})]}],
        [{"role": "assistant", "content": "living room light turned on"}],
    ])
    agent = make_agent(hass, client, {CONF_LLM_HASS_API: [API_ID], CONF_MAX_TOOL_CALL_ITERATIONS: 3})

    result = await agent.async_process(make_user_input())

    payload = result.response.as_dict()
    assert payload["response_type"] == intent.IntentResponseType.QUERY_ANSWER.value
    assert payload["card"]["simple"]["title"] == "Changes"
    assert payload["speech"]["plain"]["speech"] == "living room light turned on"


@pytest.mark.asyncio
async def test_thinking_blocks_are_not_spoken(hass: HomeAssistant):
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "content": "<think>hidden reasoning</think>The light is on"}],
    ])
    agent = make_agent(hass, client)

    result = await agent.async_process(make_user_input())

    assert result.response.speech["plain"]["speech"] == "The light is on"
    # the log keeps what the model actually produced
    assert get_log(hass).content[-1].content == "<think>hidden reasoning</think>The light is on"


@pytest.mark.asyncio
async def test_thinking_only_answer_does_not_leak_reasoning(hass: HomeAssistant):
    client = ScriptedClient(hass, replies=[[{"role": "assistant", "content": "<think>only reasoning</think>"}]])
    agent = make_agent(hass, client)

    result = await agent.async_process(make_user_input())

    assert result.response.speech["plain"]["speech"] == NO_SPEECH_MESSAGE
    assert "only reasoning" not in result.response.speech["plain"]["speech"]


@pytest.mark.asyncio
async def test_silence_does_not_resurrect_an_earlier_answer(hass: HomeAssistant):
    client = ScriptedClient(hass, replies=[
        [{"role": "assistant", "content": "first answer"}],
        [{"role": "assistant", "content": ""}],
    ])
    agent = make_agent(hass, client)

    await agent.async_process(make_user_input("u1"))
    result = await agent.async_process(make_user_input("u2"))

    # an empty answer is dropped from the log by Home Assistant itself, and is not replaced by an older
    # answer from the conversation
    assert result.response.speech["plain"]["speech"] == NO_SPEECH_MESSAGE
