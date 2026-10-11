"""Tests for AI Task data generation.

The AI Task integration opens a chat log, composes a system prompt, adds the task instructions and only
then calls the entity, so the tests hand in a real chat log prepared the same way.
"""

from typing import Any, cast

import pytest
import voluptuous as vol

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from custom_components.llama_conversation.ai_task import (
    LocalLLMTaskEntity,
    ResultExtractionMethod,
)
from custom_components.llama_conversation.const import (
    CONF_AI_TASK_EXTRACTION_METHOD,
    CONF_AI_TASK_RETRIES,
    CONF_PROMPT,
)
from custom_components.llama_conversation.entity import TextGenerationResult

TASK_PROMPT = "Produce the requested data."


class TaskAPI(llm.API):
    """API a task caller can hand along with the task."""

    def __init__(self, hass: HomeAssistant) -> None:
        super().__init__(hass=hass, id="caller-api", name="Caller API")

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        class CallerTool(llm.Tool):
            name = "caller_tool"
            description = "A tool the caller provided"
            parameters = vol.Schema({})

            async def async_call(self, hass, tool_input, llm_context):
                return {"result": "ok"}

        return llm.APIInstance(api=self, api_prompt="", llm_context=llm_context, tools=[CallerTool()])


class DummyClient:
    def __init__(self, results: list[TextGenerationResult] | TextGenerationResult) -> None:
        results = results if isinstance(results, list) else [results]
        self.results = list(results)
        self.seen_conversations: list[list[conversation.Content]] = []

    def _generate_system_prompt(self, prompt_template, llm_api, entity_options):
        return prompt_template

    def _supports_vision(self, _options):  # pragma: no cover - not needed for tests
        return False

    def _supports_streaming(self, _options):
        return False

    async def _generate(self, messages, llm_api, entity_id, options):
        self.seen_conversations.append(list(messages))
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


class DummyTaskEntity(LocalLLMTaskEntity):
    def __init__(self, hass: HomeAssistant, client: DummyClient, options: dict[str, Any]):
        # Bypass parent init to avoid ConfigEntry/Subentry plumbing.
        self.hass = hass
        self.client = client
        self._runtime_options = {CONF_PROMPT: TASK_PROMPT, **options}
        self.entity_id = "ai_task.test"
        self.entry_id = "entry"
        self.subentry_id = "subentry"
        self._attr_supported_features = 0

    @property
    def runtime_options(self):
        return self._runtime_options


def make_chat_log(hass: HomeAssistant, instructions: str = "do the thing") -> conversation.ChatLog:
    """Prepare a chat log the way the AI Task integration does before calling the entity."""
    chat_log = conversation.ChatLog(hass, "conv-id")
    chat_log.async_add_user_content(conversation.UserContent(content=instructions))
    return chat_log


def make_task(**kwargs: Any) -> ai_task.GenDataTask:
    return ai_task.GenDataTask(name="task", instructions="do the thing", **kwargs)


def roles(chat_log: conversation.ChatLog) -> list[str]:
    return [type(content).__name__ for content in chat_log.content]


@pytest.mark.asyncio
async def test_streaming_lands_in_the_log_as_one_message(hass: HomeAssistant):
    class StreamingClient(DummyClient):
        """Client that streams the answer piece by piece."""

        def _supports_streaming(self, _options):
            return True

        async def _generate_stream(self, messages, llm_api, entity_id, options):
            self.seen_conversations.append(list(messages))
            for piece in ["The ", "answer", " is 42"]:
                yield TextGenerationResult(response=piece, response_streamed=True)
            yield TextGenerationResult(response=None, response_streamed=True)

    entity = DummyTaskEntity(hass, StreamingClient(TextGenerationResult(response="unused")), {
        CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.NONE,
    })
    chat_log = make_chat_log(hass)

    result = await entity._async_generate_data(cast(Any, make_task()), chat_log)

    assert result.data == "The answer is 42"
    assert roles(chat_log) == ["SystemContent", "UserContent", "AssistantContent"]


@pytest.mark.asyncio
async def test_backend_error_is_reported_instead_of_retried(hass: HomeAssistant):
    """A backend that cannot answer is not the model's mistake, so repeating it would hide the reason."""
    class BrokenClient(DummyClient):
        def __init__(self):
            super().__init__(TextGenerationResult(response=""))
            self.attempts = 0

        async def _generate(self, messages, llm_api, entity_id, options):
            self.attempts += 1
            raise HomeAssistantError("The generation request timed out!")

    client = BrokenClient()
    entity = DummyTaskEntity(hass, client, {CONF_AI_TASK_RETRIES: 3})

    with pytest.raises(HomeAssistantError, match="timed out") as err:
        await entity._async_generate_data(cast(Any, make_task()), make_chat_log(hass))

    assert client.attempts == 1
    assert "timed out" in str(err.value)


@pytest.mark.asyncio
async def test_no_extraction_returns_raw_text(hass: HomeAssistant):
    entity = DummyTaskEntity(hass, DummyClient(TextGenerationResult(response="raw text")), {
        CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.NONE,
    })
    chat_log = make_chat_log(hass)

    result = await entity._async_generate_data(cast(Any, make_task()), chat_log)

    assert result.data == "raw text"
    assert result.conversation_id == "conv-id"
    assert chat_log.llm_api is None
    # the answer is part of the log, and Home Assistant's generic task prompt was replaced by ours
    assert roles(chat_log) == ["SystemContent", "UserContent", "AssistantContent"]
    assert chat_log.content[-1].content == "raw text"
    assert chat_log.content[0].content.startswith(TASK_PROMPT)
    # the model is not shown a second copy of the instructions the log already holds
    assert len([c for c in entity.client.seen_conversations[0] if isinstance(c, conversation.UserContent)]) == 1


@pytest.mark.asyncio
async def test_structured_output_success(hass: HomeAssistant):
    entity = DummyTaskEntity(hass, DummyClient(TextGenerationResult(response='{"foo": 1}')), {
        CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.STRUCTURED_OUTPUT,
    })
    chat_log = make_chat_log(hass)
    task = make_task(structure=vol.Schema({"foo": int}))

    result = await entity._async_generate_data(cast(Any, task), chat_log)

    assert result.data == {"foo": 1}


@pytest.mark.asyncio
async def test_structured_output_invalid_json_is_retried_then_raises(hass: HomeAssistant):
    client = DummyClient([
        TextGenerationResult(response="not-json"),
        TextGenerationResult(response="still not-json"),
    ])
    entity = DummyTaskEntity(hass, client, {
        CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.STRUCTURED_OUTPUT,
        CONF_AI_TASK_RETRIES: 1,
    })
    chat_log = make_chat_log(hass)
    task = make_task(structure=vol.Schema({"foo": int}))

    with pytest.raises(HomeAssistantError, match="properly formatted JSON"):
        await entity._async_generate_data(cast(Any, task), chat_log)

    # the model was told what it did wrong before being asked again
    assert roles(chat_log) == [
        "SystemContent", "UserContent", "AssistantContent", "UserContent", "AssistantContent",
    ]
    assert chat_log.content[3].content.startswith("Error:")
    assert len(client.seen_conversations) == 2


@pytest.mark.asyncio
async def test_structured_output_recovers_after_a_retry(hass: HomeAssistant):
    client = DummyClient([
        TextGenerationResult(response="not-json"),
        TextGenerationResult(response='{"foo": 2}'),
    ])
    entity = DummyTaskEntity(hass, client, {
        CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.STRUCTURED_OUTPUT,
        CONF_AI_TASK_RETRIES: 1,
    })
    chat_log = make_chat_log(hass)

    result = await entity._async_generate_data(cast(Any, make_task(structure=vol.Schema({"foo": int}))), chat_log)

    assert result.data == {"foo": 2}
    # the second attempt is told about the failure of the first one
    assert isinstance(client.seen_conversations[-1][3], conversation.UserContent)


@pytest.mark.asyncio
async def test_tool_extraction_success(hass: HomeAssistant):
    entity = DummyTaskEntity(hass, DummyClient(TextGenerationResult(
        response="",
        tool_calls=[llm.ToolInput("submit_response", {"value": 42})],
    )), {CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.TOOL})
    chat_log = make_chat_log(hass)
    task = make_task(structure=vol.Schema({"value": int}))

    result = await entity._async_generate_data(cast(Any, task), chat_log)

    assert result.data == {"value": 42}
    # the tool call ran through the chat log, which recorded its answer
    assert roles(chat_log) == [
        "SystemContent", "UserContent", "AssistantContent", "ToolResultContent",
    ]
    assert chat_log.content[-1].tool_result == {"value": 42}


@pytest.mark.asyncio
async def test_tool_extraction_validates_the_structure(hass: HomeAssistant):
    entity = DummyTaskEntity(hass, DummyClient(TextGenerationResult(
        response="",
        tool_calls=[llm.ToolInput("submit_response", {"value": "nope"})],
    )), {
        CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.TOOL,
        CONF_AI_TASK_RETRIES: 0,
    })
    chat_log = make_chat_log(hass)
    task = make_task(structure=vol.Schema({vol.Required("value"): int}))

    with pytest.raises(HomeAssistantError, match="schema errors"):
        await entity._async_generate_data(cast(Any, task), chat_log)


@pytest.mark.asyncio
async def test_tool_extraction_requires_a_tool_call(hass: HomeAssistant):
    entity = DummyTaskEntity(hass, DummyClient(TextGenerationResult(response="here you go")), {
        CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.TOOL,
        CONF_AI_TASK_RETRIES: 0,
    })
    task = make_task(structure=vol.Schema({"value": int}))

    with pytest.raises(HomeAssistantError, match="at least one tool call"):
        await entity._async_generate_data(cast(Any, task), make_chat_log(hass))


@pytest.mark.asyncio
async def test_tools_the_caller_provided_are_kept(hass: HomeAssistant):
    entity = DummyTaskEntity(hass, DummyClient(TextGenerationResult(
        response="",
        tool_calls=[llm.ToolInput("submit_response", {"value": 7})],
    )), {CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.TOOL})
    chat_log = make_chat_log(hass)
    task = make_task(structure=vol.Schema({"value": int}), llm_api=TaskAPI(hass))

    result = await entity._async_generate_data(cast(Any, task), chat_log)

    assert result.data == {"value": 7}
    tool_names = [tool.name for tool in chat_log.llm_api.tools]
    assert any(name.endswith("submit_response") for name in tool_names)
    assert any(name.endswith("caller_tool") for name in tool_names)


@pytest.mark.asyncio
async def test_tool_calls_without_an_api_are_ignored(hass: HomeAssistant):
    entity = DummyTaskEntity(hass, DummyClient(TextGenerationResult(
        response="the answer",
        tool_calls=[llm.ToolInput("invented_tool", {})],
    )), {CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.NONE})
    chat_log = make_chat_log(hass)

    result = await entity._async_generate_data(cast(Any, make_task()), chat_log)

    assert result.data == "the answer"
    assert roles(chat_log) == ["SystemContent", "UserContent", "AssistantContent"]
