"""AI Task integration for Local LLMs."""
from __future__ import annotations

from json import JSONDecodeError
import logging
from enum import StrEnum
from typing import Any

import voluptuous as vol
from voluptuous_openapi import convert as convert_to_openapi

from homeassistant.helpers import llm
from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from .entity import LocalLLMEntity, LocalLLMClient
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
        message_history: list[conversation.Content],
        llm_api: llm.APIInstance | None,
        entity_options: dict[str, Any],
    ) -> tuple[str, list | None, Exception | None]:
        """Generate a single response from the LLM."""
        collected_tools = None
        text = ""

        # call the LLM client directly (not _async_generate) since that will attempt to execute tool calls
        try:
            if hasattr(self.client, "_generate_stream"):
                async for chunk in self.client._generate_stream(
                    message_history,
                    llm_api,
                    self.entity_id,
                    entity_options,
                ):
                    if chunk.response:
                        text += chunk.response.strip()
                    if chunk.tool_calls:
                        collected_tools = chunk.tool_calls
            else:
                blocking_result = await self.client._generate(
                    message_history,
                    llm_api,
                    self.entity_id,
                    entity_options,
                )
                if blocking_result.response:
                    text = blocking_result.response.strip()
                if blocking_result.tool_calls:
                    collected_tools = blocking_result.tool_calls

            _LOGGER.debug("AI Task '%s' generated text: %s (tools=%s)", self.entity_id, text, collected_tools)
            return text, collected_tools, None
        except JSONDecodeError as err:
            _LOGGER.debug("AI Task '%s' json error generated text: %s (tools=%s)", self.entity_id, text, collected_tools)
            return text, collected_tools, err

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
                    data=raw_text,
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
        except JSONDecodeError as err:
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
        if task.structure: # set up extraction method specifics
            if extraction_method == ResultExtractionMethod.STRUCTURED_OUTPUT:
                _LOGGER.debug("Using structure for AI Task '%s': %s", task.name, task.structure)
                entity_options[CONF_RESPONSE_JSON_SCHEMA] = convert_to_openapi(task.structure, custom_serializer=llm.selector_serializer)
            elif extraction_method == ResultExtractionMethod.TOOL:
                chat_log.llm_api = await SubmitResponseAPI(self.hass, [SubmitResponseTool(task.structure)]).async_get_api_instance(
                    llm.LLMContext(DOMAIN, context=None, language=None, assistant=None, device_id=None)
                )
        
        message_history = list(chat_log.content) if chat_log.content else []
        task_prompt = self.client._generate_system_prompt(raw_task_prompt, llm_api=chat_log.llm_api, entity_options=entity_options)
        system_message = conversation.SystemContent(content=task_prompt)
        if message_history and isinstance(message_history[0], conversation.SystemContent):
            message_history[0] = system_message
        else:
            message_history.insert(0, system_message)

        if not any(isinstance(msg, conversation.UserContent) for msg in message_history):
            message_history.append(
                conversation.UserContent(
                    content=task.instructions, attachments=task.attachments
                )
            )
        try:
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                _LOGGER.debug("Generating response for %s (attempt %s/%s)...", task.name, attempt + 1, max_attempts)
                text, tool_calls, err = await self._generate_once(message_history, chat_log.llm_api, entity_options)
                if err:
                    last_error = err
                    message_history.append(conversation.AssistantContent(agent_id=self.entity_id, content=text, tool_calls=tool_calls))
                    message_history.append(conversation.UserContent(content=f"Error: {str(err)}. Please try again."))
                    continue
                
                data, err = self._extract_data(text, tool_calls, extraction_method, chat_log, task.structure)
                if err:
                    last_error = err
                    message_history.append(conversation.AssistantContent(agent_id=self.entity_id, content=text, tool_calls=tool_calls))
                    message_history.append(conversation.UserContent(content=f"Error: {str(err)}. Please try again."))
                    continue
                
                if data:
                    return data
        except Exception as err:
            _LOGGER.exception("Unhandled exception while running AI Task '%s'", task.name)
            raise HomeAssistantError(f"Unhandled error while running AI Task '{task.name}'") from err
        
        raise last_error or HomeAssistantError(f"AI Task '{task.name}' failed after {max_attempts} attempts")
