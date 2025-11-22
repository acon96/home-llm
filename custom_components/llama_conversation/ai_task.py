"""AI Task integration for Local LLMs."""

from __future__ import annotations

from json import JSONDecodeError
import logging
from enum import StrEnum

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, Context
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from .entity import LocalLLMEntity, LocalLLMClient
from .const import (
    CONF_PROMPT,
    DEFAULT_PROMPT,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry[LocalLLMClient],
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [LocalLLMTaskEntity(hass, config_entry, subentry, config_entry.runtime_data)],
            config_subentry_id=subentry.subentry_id,
        )

class ResultExtractionMethod(StrEnum):
    NONE = "none"
    STRUCTURED_OUTPUT = "structure"
    TOOL = "tool"

class LocalLLMTaskEntity(
    ai_task.AITaskEntity,
    LocalLLMEntity,
):
    """Ollama AI Task entity."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize Ollama AI Task entity."""
        super().__init__(*args, **kwargs)

        if self.client._supports_vision(self.runtime_options):
            self._attr_supported_features = (
                ai_task.AITaskEntityFeature.GENERATE_DATA |
                ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
            )
        else:
            self._attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""

        extraction_method = ResultExtractionMethod.NONE

        try:
            raw_prompt = self.runtime_options.get(CONF_PROMPT, DEFAULT_PROMPT)

            message_history = chat_log.content[:]

            if not isinstance(message_history[0], conversation.SystemContent):
                system_prompt = conversation.SystemContent(content=self.client._generate_system_prompt(raw_prompt, None, self.runtime_options))
                message_history.insert(0, system_prompt)
            
            _LOGGER.debug(f"Generating response for {task.name=}...")
            generation_result = await self.client._async_generate(message_history, self.entity_id, chat_log, self.runtime_options)

            assistant_message = await anext(generation_result)
            if not isinstance(assistant_message, conversation.AssistantContent):
                raise HomeAssistantError("Last content in chat log is not an AssistantContent!")
            text = assistant_message.content

            if not task.structure:
                return ai_task.GenDataTaskResult(
                    conversation_id=chat_log.conversation_id,
                    data=text,
                )
            
            if extraction_method == ResultExtractionMethod.NONE:
                raise HomeAssistantError("Task structure provided but no extraction method was specified!")
            elif extraction_method == ResultExtractionMethod.STRUCTURED_OUTPUT:
                try:
                    data = json_loads(text)
                except JSONDecodeError as err:
                    _LOGGER.error(
                        "Failed to parse JSON response: %s. Response: %s",
                        err,
                        text,
                    )
                    raise HomeAssistantError("Error with Local LLM structured response") from err
            elif extraction_method == ResultExtractionMethod.TOOL:
                try:
                    data = assistant_message.tool_calls[0].tool_args
                except (IndexError, AttributeError) as err:
                    _LOGGER.error(
                        "Failed to extract tool arguments from response: %s. Response: %s",
                        err,
                        text,
                    )
                    raise HomeAssistantError("Error with Local LLM tool response") from err
            else:
                raise ValueError() # should not happen

            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=data,
            )
        except Exception as err:
            _LOGGER.exception("Unhandled exception while running AI Task '%s'", task.name)
            raise HomeAssistantError(f"Unhandled error while running AI Task '{task.name}'") from err
