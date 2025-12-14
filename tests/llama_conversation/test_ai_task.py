"""Tests for AI Task extraction behavior."""

from typing import Any, cast

import pytest
import voluptuous as vol

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from custom_components.llama_conversation.ai_task import (
    LocalLLMTaskEntity,
    ResultExtractionMethod,
)
from custom_components.llama_conversation.const import (
    CONF_AI_TASK_EXTRACTION_METHOD,
)
from custom_components.llama_conversation.entity import TextGenerationResult


class DummyGenTask:
    def __init__(self, *, name="task", instructions="do", attachments=None, structure=None):
        self.name = name
        self.instructions = instructions
        self.attachments = attachments or []
        self.structure = structure


class DummyChatLog:
    def __init__(self, content=None):
        self.content = content or []
        self.conversation_id = "conv-id"
        self.llm_api = None

class DummyClient:
    def __init__(self, result: TextGenerationResult):
        self._result = result

    def _supports_vision(self, _options):  # pragma: no cover - not needed for tests
        return False

    async def _generate(self, _messages, _llm_api, _entity_id, _options):
        return self._result


class DummyTaskEntity(LocalLLMTaskEntity):
    def __init__(self, hass, client, options):
        # Bypass parent init to avoid ConfigEntry/Subentry plumbing.
        self.hass = hass
        self.client = client
        self._runtime_options = options
        self.entity_id = "ai_task.test"
        self.entry_id = "entry"
        self.subentry_id = "subentry"
        self._attr_supported_features = 0

    @property
    def runtime_options(self):
        return self._runtime_options


@pytest.mark.asyncio
async def test_no_extraction_returns_raw_text(hass):
    entity = DummyTaskEntity(
        hass,
        DummyClient(TextGenerationResult(response="raw text")),
        {CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.NONE},
    )
    chat_log = DummyChatLog()
    task = DummyGenTask()

    result = await entity._async_generate_data(cast(Any, task), cast(Any, chat_log))

    assert result.data == "raw text"
    assert chat_log.llm_api is None


@pytest.mark.asyncio
async def test_structured_output_success(hass):
    entity = DummyTaskEntity(
        hass,
        DummyClient(TextGenerationResult(response='{"foo": 1}')),
        {CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.STRUCTURED_OUTPUT},
    )
    chat_log = DummyChatLog()
    task = DummyGenTask(structure=vol.Schema({"foo": int}))

    result = await entity._async_generate_data(cast(Any, task), cast(Any, chat_log))

    assert result.data == {"foo": 1}


@pytest.mark.asyncio
async def test_structured_output_invalid_json_raises(hass):
    entity = DummyTaskEntity(
        hass,
        DummyClient(TextGenerationResult(response="not-json")),
        {CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.STRUCTURED_OUTPUT},
    )
    chat_log = DummyChatLog()
    task = DummyGenTask(structure=vol.Schema({"foo": int}))

    with pytest.raises(HomeAssistantError):
        await entity._async_generate_data(cast(Any, task), cast(Any, chat_log))


@pytest.mark.asyncio
async def test_tool_extraction_success(hass):
    tool_call = llm.ToolInput("submit_response", {"value": 42})
    entity = DummyTaskEntity(
        hass,
        DummyClient(TextGenerationResult(response="", tool_calls=[tool_call])),
        {CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.TOOL},
    )
    chat_log = DummyChatLog()
    task = DummyGenTask(structure=vol.Schema({"value": int}))

    result = await entity._async_generate_data(cast(Any, task), cast(Any, chat_log))

    assert result.data == {"value": 42}
    assert chat_log.llm_api is not None


@pytest.mark.asyncio
async def test_tool_extraction_missing_tool_args_raises(hass):
    class DummyToolCall:
        def __init__(self, tool_args=None):
            self.tool_args = tool_args

    tool_call = DummyToolCall(tool_args=None)
    entity = DummyTaskEntity(
        hass,
        DummyClient(TextGenerationResult(response="", tool_calls=cast(Any, [tool_call]))),
        {CONF_AI_TASK_EXTRACTION_METHOD: ResultExtractionMethod.TOOL},
    )
    chat_log = DummyChatLog()
    task = DummyGenTask(structure=vol.Schema({"value": int}))

    with pytest.raises(HomeAssistantError):
        await entity._async_generate_data(cast(Any, task), cast(Any, chat_log))

