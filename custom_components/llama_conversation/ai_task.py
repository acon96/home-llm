"""AI Task integration for Local LLMs."""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from enum import StrEnum
from typing import Any

import voluptuous as vol

try: # HA < 2026.9
    from voluptuous_openapi import convert as convert_to_openapi
except ModuleNotFoundError: # HA >= 2026.9
    from probatio import to_openapi as convert_to_openapi

from homeassistant.helpers import llm
from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads, JSON_DECODE_EXCEPTIONS

from .entity import LocalLLMEntity, LocalLLMClient
from .utils import PreparedLLMAPI, prompt_as_literal
from .const import (
    CONF_PROMPT,
    CONF_RESPONSE_JSON_SCHEMA,
    DEFAULT_AI_TASK_PROMPT,
    CONF_AI_TASK_RETRIES,
    DEFAULT_AI_TASK_RETRIES,
    CONF_AI_TASK_EXTRACTION_METHOD,
    DEFAULT_AI_TASK_EXTRACTION_METHOD,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry[LocalLLMClient],
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != ai_task.DOMAIN:
            continue

        # create one entity per subentry
        ai_task_entity = LocalLLMTaskEntity(hass, config_entry, subentry, config_entry.runtime_data)

        # make sure model is loaded
        await config_entry.runtime_data._async_load_model(dict(subentry.data))

        # register the ai task entity
        async_add_entities([ai_task_entity], config_subentry_id=subentry.subentry_id)


class ResultExtractionMethod(StrEnum):
    NONE = "none"
    STRUCTURED_OUTPUT = "structure"
    TOOL = "tool"

class SubmitResponseTool(llm.Tool):
    name = "submit_response"
    description = "Submit the structured response payload for the AI task"

    def __init__(self, parameters_schema: vol.Schema):
        self.parameters = parameters_schema

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        return tool_input.tool_args or {}


class SubmitResponseAPI(llm.API):
    def __init__(self, hass: HomeAssistant, tools: list[llm.Tool]) -> None:
        self._tools = tools
        super().__init__(
            hass=hass,
            id=f"{DOMAIN}-ai-task-tool",
            name="AI Task Tool API",
        )

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        return llm.APIInstance(
            api=self,
            api_prompt="Call submit_response to return the structured AI task result.",
            llm_context=llm_context,
            tools=self._tools,
            custom_serializer=llm.selector_serializer,
        )


class LocalLLMTaskEntity(
    ai_task.AITaskEntity,
    LocalLLMEntity,
):
    """AI Task entity."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize AI Task entity."""
        super().__init__(*args, **kwargs)

        if self.client._supports_vision(self.runtime_options):
            self._attr_supported_features = (
                ai_task.AITaskEntityFeature.GENERATE_DATA
                | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
            )
        else:
            self._attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA

    async def _generate_once(
        self,
        chat_log: conversation.ChatLog,
        entity_options: dict[str, Any],
    ) -> tuple[str, list[llm.ToolInput]]:
        """Ask the backend for a single assistant message and add it to the chat log."""
        text = ""
        tool_calls: list[llm.ToolInput] = []

        async for content in chat_log.async_add_delta_content_stream(
            self.entity_id, self._async_deltas(chat_log, entity_options)
        ):
            if isinstance(content, conversation.AssistantContent):
                text = content.content or ""
                tool_calls.extend(content.tool_calls or [])

        _LOGGER.debug("AI Task '%s' generated text: %s (tools=%s)", self.entity_id, text, tool_calls or None)
        return text, tool_calls

    async def _async_deltas(
        self,
        chat_log: conversation.ChatLog,
        entity_options: dict[str, Any],
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
        """Stream one backend completion into the chat log's delta format.

        A delta with a role starts a message, a delta without one is concatenated to the message in
        progress, which is what the backends produce. Tool calls are dropped when the task has no LLM API
        to run them, because the chat log refuses them outright.
        """
        message_history = list(chat_log.content)
        llm_api = chat_log.llm_api

        yield conversation.AssistantContentDeltaDict(role="assistant")
        if self.client._supports_streaming(entity_options):
            async for chunk in self.client._generate_stream(
                message_history, llm_api, self.entity_id, entity_options
            ):
                yield conversation.AssistantContentDeltaDict(
                    content=chunk.response,
                    tool_calls=self._tool_calls_for_log(chunk.tool_calls, llm_api),
                )
        else:
            blocking_result = await self.client._generate(
                message_history, llm_api, self.entity_id, entity_options
            )
            yield conversation.AssistantContentDeltaDict(
                content=blocking_result.response,
                tool_calls=self._tool_calls_for_log(blocking_result.tool_calls, llm_api),
            )

    @staticmethod
    def _tool_calls_for_log(
        tool_calls: list[llm.ToolInput] | None,
        llm_api: llm.APIInstance | None,
    ) -> list[llm.ToolInput] | None:
        """Ignore tool calls the chat log has no API to execute."""
        if not tool_calls:
            return None
        if llm_api is None:
            _LOGGER.warning("Model attempted to call a tool but no LLM API was provided, ignoring tool calls")
            return None
        return tool_calls

    def _extract_data(
        self,
        raw_text: str,
        tool_calls: list[llm.ToolInput] | None,
        extraction_method: ResultExtractionMethod,
        chat_log: conversation.ChatLog,
        structure: vol.Schema | None,
    ) -> tuple[ai_task.GenDataTaskResult | None, Exception | None]:
        """Extract the final data from the LLM response based on the extraction method."""
        try:
            if extraction_method == ResultExtractionMethod.NONE or structure is None:
                return ai_task.GenDataTaskResult(
                    conversation_id=chat_log.conversation_id,
                    data=raw_text.strip(),
                ), None

            if extraction_method == ResultExtractionMethod.STRUCTURED_OUTPUT:
                data = json_loads(raw_text)
                return ai_task.GenDataTaskResult(
                    conversation_id=chat_log.conversation_id,
                    data=data,
                ), None

            if extraction_method == ResultExtractionMethod.TOOL:
                first_tool = next(iter(tool_calls or []), None)
                if not first_tool:
                    return None, HomeAssistantError("Please produce at least one tool call with the structured response.")
                
                structure(first_tool.tool_args) # validate tool call against vol schema structure
                return ai_task.GenDataTaskResult(
                    conversation_id=chat_log.conversation_id,
                    data=first_tool.tool_args,
                ), None
        except vol.Invalid as err:
            if isinstance(err, vol.MultipleInvalid):
                # combine all error messages into one
                error_message = "; ".join(f"Error at '{e.path}': {e.error_message}" for e in err.errors)
            else:
                error_message = f"Error at '{err.path}': {err.error_message}"
            return None, HomeAssistantError(f"Please address the following schema errors: {error_message}")
        except JSON_DECODE_EXCEPTIONS as err:
            return None, HomeAssistantError(f"Please produce properly formatted JSON: {repr(err)}")

        raise HomeAssistantError(f"Invalid extraction method for AI Task {extraction_method}")

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        raw_task_prompt = self.runtime_options.get(CONF_PROMPT, DEFAULT_AI_TASK_PROMPT)
        retries = max(0, self.runtime_options.get(CONF_AI_TASK_RETRIES, DEFAULT_AI_TASK_RETRIES))
        extraction_method = self.runtime_options.get(CONF_AI_TASK_EXTRACTION_METHOD, DEFAULT_AI_TASK_EXTRACTION_METHOD)
        max_attempts = retries + 1

        entity_options = {**self.runtime_options}
        api: llm.API | None = task.llm_api
        if task.structure: # set up extraction method specifics
            if extraction_method == ResultExtractionMethod.STRUCTURED_OUTPUT:
                _LOGGER.debug("Using structure for AI Task '%s': %s", task.name, task.structure)
                entity_options[CONF_RESPONSE_JSON_SCHEMA] = convert_to_openapi(task.structure, custom_serializer=llm.selector_serializer)
            elif extraction_method == ResultExtractionMethod.TOOL:
                submit_api = SubmitResponseAPI(self.hass, [SubmitResponseTool(task.structure)])
                api = submit_api if api is None else llm.MergedAPI([api, submit_api])

        # The AI Task integration opened the chat log with its own default prompt and the instructions of
        # the task already in it. Hand our prompt (with the live device states) and tools back to the log
        # so that the model, the trace and any chat log subscriber all see the same conversation.
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=None,
            language=None,
            assistant=ai_task.DOMAIN,
            device_id=None,
        )
        instance = await api.async_get_api_instance(llm_context) if api is not None else None
        try:
            task_prompt = self.client._generate_system_prompt(raw_task_prompt, instance, entity_options)
        except TemplateError as err:
            raise HomeAssistantError(f"Sorry, I had a problem with my template: {err}") from err

        try:
            await chat_log.async_provide_llm_data(
                llm_context,
                PreparedLLMAPI(instance) if instance is not None else None,
                prompt_as_literal(task_prompt),
            )
        except conversation.ConverseError as err:
            raise HomeAssistantError(str(err)) from err

        last_error: Exception | None = None
        for attempt in range(max_attempts):
            _LOGGER.debug("Generating response for %s (attempt %s/%s)...", task.name, attempt + 1, max_attempts)
            data: ai_task.GenDataTaskResult | None = None
            error: Exception | None = None
            try:
                text, tool_calls = await self._generate_once(chat_log, entity_options)
                data, error = self._extract_data(text, tool_calls, extraction_method, chat_log, task.structure)
            except JSON_DECODE_EXCEPTIONS as err:
                _LOGGER.debug("AI Task '%s' produced text that was not valid JSON", task.name)
                error = HomeAssistantError(f"Please produce properly formatted JSON: {repr(err)}")
            except HomeAssistantError:
                # A backend or a tool already described this failure the way Home Assistant should show
                # it, so repeating it to the model would only hide it behind a retry.
                raise
            except Exception as err:
                _LOGGER.exception("Unhandled exception while running AI Task '%s'", task.name)
                raise HomeAssistantError(f"Unhandled error while running AI Task '{task.name}'") from err

            if error is None and data is not None:
                return data

            last_error = error
            if attempt < max_attempts - 1:
                # The chat log is the message history, so the correction has to go through it to give the
                # model something to work with on the next attempt.
                chat_log.async_add_user_content(
                    conversation.UserContent(content=f"Error: {last_error}. Please try again.")
                )

        raise last_error or HomeAssistantError(f"AI Task '{task.name}' failed after {max_attempts} attempts")
