"""Defines the various LLM Backend Agents"""
from __future__ import annotations
from typing import Literal, List, Tuple, Any
import logging

from homeassistant.components.conversation import ConversationInput, ConversationResult, ConversationEntity
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import TemplateError, HomeAssistantError
from homeassistant.helpers import chat_session, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from custom_components.llama_conversation.utils import MalformedToolCallException

from .entity import LocalLLMEntity, LocalLLMClient, LocalLLMConfigEntry
from .const import (
    CONF_PROMPT,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    DEFAULT_PROMPT,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_NUM_INTERACTIONS,
    DEFAULT_MAX_TOOL_CALL_ITERATIONS,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry: LocalLLMConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> bool:
    """Set up Local LLM Conversation from a config entry."""

    for subentry in entry.subentries.values():
        if subentry.subentry_type != conversation.DOMAIN:
            continue

        # create one agent entity per conversation subentry
        agent_entity = LocalLLMAgent(hass, entry, subentry, entry.runtime_data)

        # make sure model is loaded
        await entry.runtime_data._async_load_model(dict(subentry.data))

        # register the agent entity
        async_add_entities([agent_entity], config_subentry_id=subentry.subentry_id,)

    return True

class LocalLLMAgent(ConversationEntity, AbstractConversationAgent, LocalLLMEntity):
    """Base Local LLM conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, subentry: ConfigSubentry, client: LocalLLMClient) -> None:
        super().__init__(hass, entry, subentry, client)

        if subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process a sentence."""
        with (
            chat_session.async_get_chat_session(
                self.hass, user_input.conversation_id
            ) as session,
            conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            raw_prompt = self.runtime_options.get(CONF_PROMPT, DEFAULT_PROMPT)
            refresh_system_prompt = self.runtime_options.get(CONF_REFRESH_SYSTEM_PROMPT, DEFAULT_REFRESH_SYSTEM_PROMPT)
            remember_conversation = self.runtime_options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION)
            remember_num_interactions = self.runtime_options.get(CONF_REMEMBER_NUM_INTERACTIONS, DEFAULT_REMEMBER_NUM_INTERACTIONS)
            max_tool_call_iterations = self.runtime_options.get(CONF_MAX_TOOL_CALL_ITERATIONS, DEFAULT_MAX_TOOL_CALL_ITERATIONS)
            llm_api: llm.APIInstance | None = None
            if self.runtime_options.get(CONF_LLM_HASS_API):
                try:
                    llm_api = await llm.async_get_api(
                        self.hass,
                        self.runtime_options[CONF_LLM_HASS_API],
                        llm_context=user_input.as_llm_context(DOMAIN)
                    )
                except HomeAssistantError as err:
                    _LOGGER.error("Error getting LLM API: %s", err)
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.UNKNOWN,
                        f"Error preparing LLM API: {err}",
                    )
                    return ConversationResult(
                        response=intent_response, conversation_id=user_input.conversation_id
                    )
                
            # ensure this chat log has the LLM API instance
            chat_log.llm_api = llm_api

            if remember_conversation:
                message_history = chat_log.content[:]
            else:
                message_history = []

            # trim message history before processing if necessary
            if remember_num_interactions and len(message_history) > (remember_num_interactions * 2) + 1:
                new_message_history = [message_history[0]] # copy system prompt
                new_message_history.extend(message_history[1:][-(remember_num_interactions * 2):])

            # re-generate prompt if necessary
            if len(message_history) == 0 or refresh_system_prompt:
                try:
                    system_prompt = conversation.SystemContent(content=self.client._generate_system_prompt(raw_prompt, llm_api, self.runtime_options))
                except TemplateError as err:
                    _LOGGER.error("Error rendering prompt: %s", err)
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.UNKNOWN,
                        f"Sorry, I had a problem with my template: {err}",
                    )
                    return ConversationResult(
                        response=intent_response, conversation_id=user_input.conversation_id
                    )

                if len(message_history) == 0:
                    message_history.append(system_prompt)
                else:
                    message_history[0] = system_prompt

            tool_calls: List[Tuple[llm.ToolInput, Any]] = []
            # if max tool calls is 0 then we expect to generate the response & tool call in one go
            for idx in range(max(1, max_tool_call_iterations)):
                _LOGGER.debug(f"Generating response for {user_input.text=}, iteration {idx+1}/{max_tool_call_iterations}")
                generation_result = await self.client._async_generate(message_history, user_input.agent_id, chat_log, self.runtime_options)
                
                last_generation_had_tool_calls = False
                while True:
                    try:
                        message = await anext(generation_result)
                        message_history.append(message)
                        _LOGGER.debug("Added message to history: %s", message)
                        if message.role == "assistant":
                            if message.tool_calls and len(message.tool_calls) > 0:
                                last_generation_had_tool_calls = True
                            else:
                                last_generation_had_tool_calls = False
                    except StopAsyncIteration:
                        break
                    except MalformedToolCallException as err:
                        message_history.extend(err.as_tool_messages())
                        last_generation_had_tool_calls = True
                        _LOGGER.debug("Malformed tool call produced", exc_info=err)
                    except Exception as err:
                        _LOGGER.exception("There was a problem talking to the backend")
                        intent_response = intent.IntentResponse(language=user_input.language)
                        intent_response.async_set_error(
                            intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                            f"Sorry, there was a problem talking to the backend: {repr(err)}",
                        )
                        return ConversationResult(
                            response=intent_response, conversation_id=user_input.conversation_id
                        )

                # If not multi-turn, break after first tool call
                # also break if no tool calls were made
                if not last_generation_had_tool_calls:
                    break

                # return an error if we run out of attempt without succeeding
                if idx == max_tool_call_iterations - 1 and max_tool_call_iterations > 0:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                        f"Sorry, I ran out of attempts to handle your request",
                    )
                    return ConversationResult(
                        response=intent_response, conversation_id=user_input.conversation_id
                    )
                
            # generate intent response to Home Assistant
            intent_response = intent.IntentResponse(language=user_input.language)
            if len(tool_calls) > 0:
                str_tools = [f"{input.tool_name}({', '.join(str(x) for x in input.tool_args.values())})" for input, response in tool_calls]
                tools_str = '\n'.join(str_tools)
                intent_response.async_set_card(
                    title="Changes",
                    content=f"Ran the following tools:\n{tools_str}"
                )

            has_speech = False
            for i in range(1, len(message_history)):
                cur_msg = message_history[-1 * i]
                if isinstance(cur_msg, conversation.AssistantContent) and cur_msg.content:
                    intent_response.async_set_speech(cur_msg.content)
                    has_speech = True
                    break

            if not has_speech:
                intent_response.async_set_speech("I don't have anything to say right now")
                _LOGGER.debug(message_history)

            return ConversationResult(
                response=intent_response, conversation_id=user_input.conversation_id
            )
