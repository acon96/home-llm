"""Defines the various LLM Backend Agents"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import voluptuous as vol
from typing import Literal, Any, List, Dict, Optional, Tuple, AsyncIterator, AsyncGenerator
from dataclasses import dataclass

from homeassistant.components.conversation import ConversationInput, ConversationResult, ConversationEntity
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError, HomeAssistantError
from homeassistant.helpers import config_validation as cv, intent, template, entity_registry as er, llm, \
    area_registry as ar, device_registry as dr, chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import color


import voluptuous_serialize

from .utils import closest_color, flatten_vol_schema, custom_custom_serializer
from .const import (
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    CONF_BACKEND_TYPE,
    CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
    CONF_PROMPT_TEMPLATE,
    CONF_TOOL_FORMAT,
    CONF_TOOL_MULTI_TURN_CHAT,
    CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    CONF_IN_CONTEXT_EXAMPLES_FILE,
    CONF_NUM_IN_CONTEXT_EXAMPLES,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_SERVICE_CALL_REGEX,
    CONF_CONTEXT_LENGTH,
    DEFAULT_PROMPT,
    DEFAULT_BACKEND_TYPE,
    DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_TOOL_FORMAT,
    DEFAULT_TOOL_MULTI_TURN_CHAT,
    DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    DEFAULT_IN_CONTEXT_EXAMPLES_FILE,
    DEFAULT_NUM_IN_CONTEXT_EXAMPLES,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_NUM_INTERACTIONS,
    DEFAULT_SERVICE_CALL_REGEX,
    DEFAULT_CONTEXT_LENGTH,
    DOMAIN,
    HOME_LLM_API_ID,
    SERVICE_TOOL_NAME,
    PROMPT_TEMPLATE_DESCRIPTIONS,
    TOOL_FORMAT_FULL,
    TOOL_FORMAT_REDUCED,
    TOOL_FORMAT_MINIMAL,
    ALLOWED_SERVICE_CALL_ARGUMENTS,
    SERVICE_TOOL_ALLOWED_SERVICES,
    SERVICE_TOOL_ALLOWED_DOMAINS,
    CONF_BACKEND_TYPE,
    DEFAULT_BACKEND_TYPE,
)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

@dataclass(kw_only=True)
class TextGenerationResult:
    response: Optional[str] = None
    stop_reason: Optional[str] = None
    tool_calls: Optional[List[llm.ToolInput]] = None
    response_streamed: bool = False
    raise_error: bool = False
    error_msg: Optional[str] = None


async def update_listener(hass: HomeAssistant, entry: ConfigEntry):
    """Handle options update."""
    hass.data[DOMAIN][entry.entry_id] = entry

    # call update handler
    agent: LocalLLMAgent = entry.runtime_data
    await hass.async_add_executor_job(agent._update_options)

    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> bool:
    """Set up Local LLM Conversation from a config entry."""

    # handle updates to the options
    entry.async_on_unload(entry.add_update_listener(update_listener))

    # register the agent entity
    async_add_entities([entry.runtime_data])

    return True

def _convert_content(
    chat_content: conversation.Content
) -> dict[str, str]:
    """Create tool response content."""
    role_name = None
    if isinstance(chat_content, conversation.ToolResultContent):
        role_name = "tool"
    elif isinstance(chat_content, conversation.AssistantContent):
        role_name = "assistant"
    elif isinstance(chat_content, conversation.UserContent):
        role_name = "user"
    elif isinstance(chat_content, conversation.SystemContent):
        role_name = "system"
    else:
        raise ValueError(f"Unexpected content type: {type(chat_content)}")

    return { "role": role_name, "message": chat_content.content }

def _convert_content_back(
    agent_id: str,
    message_history_entry: dict[str, str]
) -> Optional[conversation.Content]:
    if message_history_entry["role"] == "tool":
        return conversation.ToolResultContent(agent_id=agent_id, content=message_history_entry["message"])
    if message_history_entry["role"] == "assistant":
        return conversation.AssistantContent(agent_id=agent_id, content=message_history_entry["message"])
    if message_history_entry["role"] == "user":
        return conversation.UserContent(content=message_history_entry["message"])
    if message_history_entry["role"] == "system":
        return conversation.SystemContent(content=message_history_entry["message"])
    
def _parse_raw_tool_call(raw_block: str, llm_api: llm.APIInstance, user_input: ConversationInput) -> tuple[bool, ConversationResult | llm.ToolInput, str | None]:
    parsed_tool_call: dict = json.loads(raw_block)

    if llm_api.api.id == HOME_LLM_API_ID:
        schema_to_validate = vol.Schema({
            vol.Required('service'): str,
            vol.Required('target_device'): str,
            vol.Optional('rgb_color'): str,
            vol.Optional('brightness'): vol.Coerce(float),
            vol.Optional('temperature'): vol.Coerce(float),
            vol.Optional('humidity'): vol.Coerce(float),
            vol.Optional('fan_mode'): str,
            vol.Optional('hvac_mode'): str,
            vol.Optional('preset_mode'): str,
            vol.Optional('duration'): str,
            vol.Optional('item'): str,
        })
    else:
        schema_to_validate = vol.Schema({
            vol.Required("name"): str,
            vol.Required("arguments"): dict,
        })

    try:
        schema_to_validate(parsed_tool_call)
    except vol.Error as ex:
        _LOGGER.info(f"LLM produced an improperly formatted response: {repr(ex)}")

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_error(
            intent.IntentResponseErrorCode.NO_INTENT_MATCH,
            f"I'm sorry, I didn't produce a correctly formatted tool call! Please see the logs for more info.",
        )
        return False, ConversationResult(
            response=intent_response, conversation_id=user_input.conversation_id
        ), ""

    # try to fix certain arguments
    args_dict = parsed_tool_call if llm_api.api.id == HOME_LLM_API_ID else parsed_tool_call["arguments"]

    # make sure brightness is 0-255 and not a percentage
    if "brightness" in args_dict and 0.0 < args_dict["brightness"] <= 1.0:
        args_dict["brightness"] = int(args_dict["brightness"] * 255)

    # convert string "tuple" to a list for RGB colors
    if "rgb_color" in args_dict and isinstance(args_dict["rgb_color"], str):
        args_dict["rgb_color"] = [ int(x) for x in args_dict["rgb_color"][1:-1].split(",") ]

    if llm_api.api.id == HOME_LLM_API_ID:
        to_say = parsed_tool_call.pop("to_say", "")
        tool_input = llm.ToolInput(
            tool_name=SERVICE_TOOL_NAME,
            tool_args=parsed_tool_call,
        )
    else:
        to_say = ""
        tool_input = llm.ToolInput(
            tool_name=parsed_tool_call["name"],
            tool_args=parsed_tool_call["arguments"],
        )

    return True, tool_input, to_say

class LocalLLMAgent(ConversationEntity, AbstractConversationAgent):
    """Base Local LLM conversation agent."""

    hass: HomeAssistant
    entry_id: str
    in_context_examples: Optional[List[Dict[str, str]]]

    _attr_has_entity_name = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self._attr_name = entry.title
        self._attr_unique_id = entry.entry_id

        self.hass = hass
        self.entry_id = entry.entry_id

        self.backend_type = entry.data.get(
            CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE
        )

        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )
        
        # mark true if subclass supports streaming responses
        self._attr_supports_streaming = hasattr(self, "_generate_stream")

        self.in_context_examples = None
        if entry.options.get(CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES, DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES):
            self._load_icl_examples(entry.options.get(CONF_IN_CONTEXT_EXAMPLES_FILE, DEFAULT_IN_CONTEXT_EXAMPLES_FILE))

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    def _load_icl_examples(self, filename: str):
        """Load info used for generating in context learning examples"""
        try:
            icl_filename = os.path.join(os.path.dirname(__file__), filename)

            with open(icl_filename, encoding="utf-8-sig") as f:
                self.in_context_examples = list(csv.DictReader(f))

                if set(self.in_context_examples[0].keys()) != set(["type", "request", "tool", "response" ]):
                    raise Exception("ICL csv file did not have 2 columns: service & response")

            if len(self.in_context_examples) == 0:
                _LOGGER.warning(f"There were no in context learning examples found in the file '{filename}'!")
                self.in_context_examples = None
            else:
                _LOGGER.debug(f"Loaded {len(self.in_context_examples)} examples for ICL")
        except Exception:
            _LOGGER.exception("Failed to load in context learning examples!")
            self.in_context_examples = None

    def _update_options(self):
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

        if self.entry.options.get(CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES, DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES):
            self._load_icl_examples(self.entry.options.get(CONF_IN_CONTEXT_EXAMPLES_FILE, DEFAULT_IN_CONTEXT_EXAMPLES_FILE))
        else:
            self.in_context_examples = None

    @property
    def entry(self) -> ConfigEntry:
        try:
            return self.hass.data[DOMAIN][self.entry_id]
        except KeyError as ex:
            raise Exception("Attempted to use self.entry during startup.") from ex

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    def _load_model(self, entry: ConfigEntry) -> None:
        """Load the model on the backend. Implemented by sub-classes"""
        raise NotImplementedError()

    async def _async_load_model(self, entry: ConfigEntry) -> None:
        """Default implementation is to call _load_model() which probably does blocking stuff"""
        await self.hass.async_add_executor_job(
            self._load_model, entry
        )
    
    def _generate_stream(self, conversation: List[Dict[str, str]]) -> TextGenerationResult:
        raise NotImplementedError()

    def _generate(self, conversation: List[Dict[str, str]]) -> TextGenerationResult:
        """Call the backend to generate a response from the conversation. Implemented by sub-classes"""
        raise NotImplementedError()

    async def _async_generate(self, conversation: List[Dict[str, str]]) -> TextGenerationResult:
        """Default implementation is to call _generate() which probably does blocking stuff"""
        if hasattr(self, '_generate_stream'):
            return await self.hass.async_add_executor_job(
                self._generate_stream, conversation
            )
        return await self.hass.async_add_executor_job(
            self._generate, conversation
        )

    def _warn_context_size(self):
        num_entities = len(self._async_get_exposed_entities()[0])
        context_size = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        _LOGGER.error("There were too many entities exposed when attempting to generate a response for " +
                      f"{self.entry.data[CONF_CHAT_MODEL]} and it exceeded the context size for the model. " +
                      f"Please reduce the number of entities exposed ({num_entities}) or increase the model's context size ({int(context_size)})")
        
    def _transform_result_stream(
        self,
        result: AsyncIterator[TextGenerationResult],
        user_input: ConversationInput,
        chat_log: conversation.chat_log.ChatLog
    ):
        async def async_iterator():
            async for input_chunk in result:
                yield conversation.AssistantContentDeltaDict(
                    content=input_chunk.response,
                    tool_calls=input_chunk.tool_calls
                )
        
        
        chat_log.async_add_delta_content_stream(user_input.agent_id, stream=async_iterator())

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
            return await self._async_handle_message(user_input, chat_log)

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:

        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        prompt_template = self.entry.options.get(CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE)
        template_desc = PROMPT_TEMPLATE_DESCRIPTIONS[prompt_template]
        refresh_system_prompt = self.entry.options.get(CONF_REFRESH_SYSTEM_PROMPT, DEFAULT_REFRESH_SYSTEM_PROMPT)
        remember_conversation = self.entry.options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION)
        remember_num_interactions = self.entry.options.get(CONF_REMEMBER_NUM_INTERACTIONS, DEFAULT_REMEMBER_NUM_INTERACTIONS)
        service_call_regex = self.entry.options.get(CONF_SERVICE_CALL_REGEX, DEFAULT_SERVICE_CALL_REGEX)

        try:
            service_call_pattern = re.compile(service_call_regex, flags=re.MULTILINE)
        except Exception as err:
            _LOGGER.exception("There was a problem compiling the service call regex")

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, there was a problem compiling the service call regex: {err}",
            )

            return ConversationResult(
                response=intent_response, conversation_id=user_input.conversation_id
            )

        llm_api: llm.APIInstance | None = None
        if self.entry.options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    self.hass,
                    self.entry.options[CONF_LLM_HASS_API],
                    llm_context=llm.LLMContext(
                        platform=DOMAIN,
                        context=user_input.context,
                        language=user_input.language,
                        assistant=conversation.DOMAIN,
                        device_id=user_input.device_id,
                    )
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

        if remember_conversation:
            message_history = [ _convert_content(content) for content in chat_log.content ]
        else:
            message_history = []

        # trim message history before processing if necessary
        if remember_num_interactions and len(message_history) > (remember_num_interactions * 2) + 1:
            new_message_history = [message_history[0]] # copy system prompt
            new_message_history.extend(message_history[1:][-(remember_num_interactions * 2):])

        # re-generate prompt if necessary
        if len(message_history) == 0 or refresh_system_prompt:
            try:
                system_prompt = conversation.SystemContent(content=self._generate_system_prompt(raw_prompt, llm_api))
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

        multi_turn_enabled = self.entry.options.get(CONF_TOOL_MULTI_TURN_CHAT, DEFAULT_TOOL_MULTI_TURN_CHAT)
        MAX_TOOL_CALL_ITERATIONS = 3 if multi_turn_enabled else 1
        for _ in range(MAX_TOOL_CALL_ITERATIONS):
            # generate a response
            try:
                _LOGGER.debug(message_history)
                generation_result = await self._async_generate(message_history)
                _LOGGER.debug(generation_result)

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

            response = generation_result.response or ""

            # remove think blocks
            response = re.sub(rf"^.*?{template_desc["chain_of_thought"]["suffix"]}", "", response, flags=re.DOTALL)

            message_history.append({"role": "assistant", "message": response})
            if llm_api is None or (generation_result.tool_calls and len(generation_result.tool_calls) == 0):
                if not generation_result.response_streamed:
                    chat_log.async_add_assistant_content_without_tools(conversation.AssistantContent(
                        agent_id=user_input.agent_id,
                        content=response,
                    ))

                # return the output without messing with it if there is no API exposed to the model
                intent_response = intent.IntentResponse(language=user_input.language)
                return ConversationResult(
                    response=intent_response, conversation_id=user_input.conversation_id
                )

            # execute the tool calls
            tool_calls: List[Tuple[llm.ToolInput, Any]] = []
            for tool_input in generation_result.tool_calls or []:
                tool_response = None
                try:
                    tool_response = llm_api.async_call_tool(tool_input)
                    _LOGGER.debug("Tool response: %s", tool_response)

                    tool_calls.append((tool_input, tool_response))
                except (HomeAssistantError, vol.Invalid) as e:
                    tool_response = {"error": type(e).__name__}
                    if str(e):
                        tool_response["error_text"] = str(e)
                    _LOGGER.debug("Tool response: %s", tool_response)

                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.NO_INTENT_MATCH,
                        f"I'm sorry! I encountered an error calling the tool. See the logs for more info.",
                    )
                    return ConversationResult(
                        response=intent_response, conversation_id=user_input.conversation_id
                    )
                
                if tool_response and multi_turn_enabled:
                    async for tool_result in chat_log.async_add_assistant_content(
                        conversation.AssistantContent(
                            agent_id=user_input.agent_id,
                            content=response,
                            tool_calls=generation_result.tool_calls,
                        ),
                        tool_call_tasks={ x[0].tool_name: x[1] for x in tool_calls}
                    ):
                        message_history.append({"role": "tool", "content": json.dumps(tool_result.tool_result) })

        # generate intent response to Home Assistant
        intent_response = intent.IntentResponse(language=user_input.language)
        if len(tool_calls) > 0:
            str_tools = [f"{input.tool_name}({', '.join(input.tool_args.values())})" for input, response in tool_calls]
            intent_response.async_set_card(
                title="Changes",
                content=f"Ran the following tools:\n{'\n'.join(str_tools)}"
            )
        return ConversationResult(
            response=intent_response, conversation_id=user_input.conversation_id
        )

    def _async_get_exposed_entities(self) -> tuple[dict[str, str], list[str]]:
        """Gather exposed entity states"""
        entity_states = {}
        domains = set()
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)

        for state in self.hass.states.async_all():
            if not async_should_expose(self.hass, CONVERSATION_DOMAIN, state.entity_id):
                continue

            entity = entity_registry.async_get(state.entity_id)
            device = None
            if entity and entity.device_id:
                device = device_registry.async_get(entity.device_id)

            attributes = dict(state.attributes)
            attributes["state"] = state.state

            if entity:
                if entity.aliases:
                    attributes["aliases"] = entity.aliases

                if entity.unit_of_measurement:
                    attributes["state"] = attributes["state"] + " " + entity.unit_of_measurement

            # area could be on device or entity. prefer device area
            area_id = None
            if device and device.area_id:
                area_id = device.area_id
            if entity and entity.area_id:
                area_id = entity.area_id

            if area_id:
                area = area_registry.async_get_area(area_id)
                if area:
                    attributes["area_id"] = area.id
                    attributes["area_name"] = area.name

            entity_states[state.entity_id] = attributes
            domains.add(state.domain)

        return entity_states, list(domains)

    def _format_prompt(
        self, prompt: list[dict], include_generation_prompt: bool = True
    ) -> str:
        """Format a conversation into a raw text completion using the model's prompt template"""
        formatted_prompt = ""

        prompt_template = self.entry.options.get(CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE)
        template_desc = PROMPT_TEMPLATE_DESCRIPTIONS[prompt_template]

        # handle models without a system prompt
        if prompt[0]["role"] == "system" and "system" not in template_desc:
            system_prompt = prompt.pop(0)
            prompt[0]["message"] = system_prompt["message"] + prompt[0]["message"]

        for message in prompt:
            role = message["role"]
            message = message["message"]
            # fall back to the "user" role for unknown roles
            role_desc = template_desc.get(role, template_desc["user"])
            formatted_prompt = (
                formatted_prompt + f"{role_desc['prefix']}{message}{role_desc['suffix']}\n"
            )

        if include_generation_prompt:
            formatted_prompt = formatted_prompt + template_desc["generation_prompt"]

        _LOGGER.debug(formatted_prompt)
        return formatted_prompt

    def _format_tool(self, name: str, parameters: vol.Schema, description: str):
        style = self.entry.options.get(CONF_TOOL_FORMAT, DEFAULT_TOOL_FORMAT)

        if style == TOOL_FORMAT_MINIMAL:
            result = f"{name}({','.join(flatten_vol_schema(parameters))})"
            if description:
                result = result + f" - {description}"
            return result

        raw_parameters: list = voluptuous_serialize.convert(
            parameters, custom_serializer=custom_custom_serializer)

        # handle vol.Any in the key side of things
        processed_parameters = []
        for param in raw_parameters:
            if isinstance(param["name"], vol.Any):
                for possible_name in param["name"].validators:
                    actual_param = param.copy()
                    actual_param["name"] = possible_name
                    actual_param["required"] = False
                    processed_parameters.append(actual_param)
            else:
                processed_parameters.append(param)

        if style == TOOL_FORMAT_REDUCED:
            return {
                "name": name,
                "description": description,
                "parameters": {
                    "properties": {
                        x["name"]: x.get("type", "string") for x in processed_parameters
                    },
                    "required": [
                        x["name"] for x in processed_parameters if x.get("required")
                    ]
                }
            }
        elif style == TOOL_FORMAT_FULL:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            x["name"]: {
                                "type": x.get("type", "string"),
                                "description": x.get("description", ""),
                            } for x in processed_parameters
                        },
                        "required": [
                            x["name"] for x in processed_parameters if x.get("required")
                        ]
                    }
                }
            }

        raise Exception(f"Unknown tool format {style}")

    def _generate_icl_examples(self, num_examples, entity_names):
        entity_names = entity_names[:]
        entity_domains = set([x.split(".")[0] for x in entity_names])

        area_registry = ar.async_get(self.hass)
        all_areas = list(area_registry.async_list_areas())

        if not self.in_context_examples:
            _LOGGER.warning(f"Attempted to generate {num_examples} ICL examples for conversation, but none are available!")
            return []
        
        in_context_examples = [
            x for x in self.in_context_examples
            if x["type"] in entity_domains
        ]

        random.shuffle(in_context_examples)
        random.shuffle(entity_names)

        num_examples_to_generate = min(num_examples, len(in_context_examples))
        if num_examples_to_generate < num_examples:
            _LOGGER.warning(f"Attempted to generate {num_examples} ICL examples for conversation, but only {len(in_context_examples)} are available!")

        examples = []
        for _ in range(num_examples_to_generate):
            chosen_example = in_context_examples.pop()
            request = chosen_example["request"]
            response = chosen_example["response"]

            random_device = [ x for x in entity_names if x.split(".")[0] == chosen_example["type"] ][0]
            random_area = random.choice(all_areas).name
            random_brightness = round(random.random(), 2)
            random_color = random.choice(list(color.COLORS.keys()))

            tool_arguments = {}

            if "<area>" in request:
                request = request.replace("<area>", random_area)
                response = response.replace("<area>", random_area)
                tool_arguments["area"] = random_area

            if "<name>" in request:
                request = request.replace("<name>", random_device)
                response = response.replace("<name>", random_device)
                tool_arguments["name"] = random_device

            if "<brightness>" in request:
                request = request.replace("<brightness>", str(random_brightness))
                response = response.replace("<brightness>", str(random_brightness))
                tool_arguments["brightness"] = random_brightness

            if "<color>" in request:
                request = request.replace("<color>", random_color)
                response = response.replace("<color>", random_color)
                tool_arguments["color"] = random_color

            examples.append({
                "request": request,
                "response": response,
                "tool": {
                    "name": chosen_example["tool"],
                    "arguments": tool_arguments
                }
            })

        return examples

    def _generate_system_prompt(self, prompt_template: str, llm_api: llm.APIInstance | None) -> str:
        """Generate the system prompt with current entity states"""
        entities_to_expose, domains = self._async_get_exposed_entities()

        extra_attributes_to_expose = self.entry.options \
            .get(CONF_EXTRA_ATTRIBUTES_TO_EXPOSE, DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE)

        def expose_attributes(attributes) -> list[str]:
            result = []
            for attribute_name in extra_attributes_to_expose:
                if attribute_name not in attributes:
                    continue

                _LOGGER.debug(f"{attribute_name} = {attributes[attribute_name]}")

                value = attributes[attribute_name]
                if value is not None:
                    # try to apply unit if present
                    unit_suffix = attributes.get(f"{attribute_name}_unit")
                    if unit_suffix:
                        value = f"{value} {unit_suffix}"
                    elif attribute_name == "temperature":
                        # try to get unit or guess otherwise
                        suffix = "F" if value > 50 else "C"
                        value = F"{int(value)} {suffix}"
                    elif attribute_name == "rgb_color":
                        value = F"{closest_color(value)} {value}"
                    elif attribute_name == "volume_level":
                        value = f"vol={int(value*100)}"
                    elif attribute_name == "brightness":
                        value = f"{int(value/255*100)}%"
                    elif attribute_name == "humidity":
                        value = f"{value}%"

                    result.append(str(value))
            return result

        devices = []
        formatted_devices = ""

        # expose devices and their alias as well
        for name, attributes in entities_to_expose.items():
            state = attributes["state"]
            exposed_attributes = expose_attributes(attributes)
            str_attributes = ";".join([state] + exposed_attributes)

            formatted_devices = formatted_devices + f"{name} '{attributes.get('friendly_name')}' = {str_attributes}\n"
            devices.append({
                "entity_id": name,
                "name": attributes.get('friendly_name'),
                "state": state,
                "attributes": exposed_attributes,
                "area_name": attributes.get("area_name"),
                "area_id": attributes.get("area_id"),
                "is_alias": False
            })
            if "aliases" in attributes:
                for alias in attributes["aliases"]:
                    formatted_devices = formatted_devices + f"{name} '{alias}' = {str_attributes}\n"
                    devices.append({
                        "entity_id": name,
                        "name": alias,
                        "state": state,
                        "attributes": exposed_attributes,
                        "area_name": attributes.get("area_name"),
                        "area_id": attributes.get("area_id"),
                        "is_alias": True
                    })

        if llm_api:
            if llm_api.api.id == HOME_LLM_API_ID:
                service_dict = self.hass.services.async_services()
                all_services = []
                scripts_added = False
                for domain in domains:
                    if domain not in SERVICE_TOOL_ALLOWED_DOMAINS:
                        continue

                    # scripts show up as individual services
                    if domain == "script" and not scripts_added:
                        all_services.extend([
                            ("script.reload", vol.Schema({}), ""),
                            ("script.turn_on", vol.Schema({}), ""),
                            ("script.turn_off", vol.Schema({}), ""),
                            ("script.toggle", vol.Schema({}), ""),
                        ])
                        scripts_added = True
                        continue

                    for name, service in service_dict.get(domain, {}).items():
                        if name not in SERVICE_TOOL_ALLOWED_SERVICES:
                            continue

                        args = flatten_vol_schema(service.schema)
                        args_to_expose = set(args).intersection(ALLOWED_SERVICE_CALL_ARGUMENTS)
                        service_schema = vol.Schema({
                            vol.Optional(arg): str for arg in args_to_expose
                        })

                        all_services.append((f"{domain}.{name}", service_schema, ""))

                tools = [
                    self._format_tool(*tool)
                    for tool in all_services
                ]

            else:
                tools = [
                    self._format_tool(tool.name, tool.parameters, tool.description)
                    for tool in llm_api.tools
                ]

            if  self.entry.options.get(CONF_TOOL_FORMAT, DEFAULT_TOOL_FORMAT) == TOOL_FORMAT_MINIMAL:
                formatted_tools = ", ".join(tools)
            else:
                formatted_tools = json.dumps(tools)
        else:
            tools = ["No tools were provided. If the user requests you interact with a device, tell them you are unable to do so."]
            formatted_tools = tools[0]

        render_variables = {
            "devices": devices,
            "formatted_devices": formatted_devices,
            "tools": tools,
            "formatted_tools": formatted_tools,
            "response_examples": []
        }

        # only pass examples if there are loaded examples + an API was exposed
        if self.in_context_examples and llm_api:
            num_examples = int(self.entry.options.get(CONF_NUM_IN_CONTEXT_EXAMPLES, DEFAULT_NUM_IN_CONTEXT_EXAMPLES))
            render_variables["response_examples"] = self._generate_icl_examples(num_examples, list(entities_to_expose.keys()))

        return template.Template(prompt_template, self.hass).async_render(
            render_variables,
            parse_result=False,
        )
