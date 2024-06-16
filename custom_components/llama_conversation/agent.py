"""Defines the various LLM Backend Agents"""
from __future__ import annotations

import aiohttp
import asyncio
import csv
import importlib
import json
import logging
import os
import random
import re
import threading
import time
import voluptuous as vol
from typing import Literal, Any, Callable

from homeassistant.components.conversation import ConversationInput, ConversationResult, AbstractConversationAgent
import homeassistant.components.conversation as ha_conversation
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, CONF_HOST, CONF_PORT, CONF_SSL, MATCH_ALL, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady, ConfigEntryError, TemplateError, HomeAssistantError
from homeassistant.helpers import config_validation as cv, intent, template, entity_registry as er, llm, \
    area_registry as ar, device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_state_change, async_call_later
from homeassistant.components.sensor import SensorEntity
from homeassistant.util import ulid, color

import voluptuous_serialize

from .utils import closest_color, flatten_vol_schema, custom_custom_serializer, install_llama_cpp_python, \
    validate_llama_cpp_python_installation, format_url
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_TYPICAL_P,
    CONF_MIN_P,
    CONF_REQUEST_TIMEOUT,
    CONF_BACKEND_TYPE,
    CONF_DOWNLOADED_MODEL_FILE,
    CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
    CONF_PROMPT_TEMPLATE,
    CONF_TOOL_FORMAT,
    CONF_TOOL_MULTI_TURN_CHAT,
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
    CONF_GENERIC_OPENAI_PATH,
    CONF_CONTEXT_LENGTH,
    CONF_BATCH_SIZE,
    CONF_THREAD_COUNT,
    CONF_BATCH_THREAD_COUNT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_MIN_P,
    DEFAULT_TYPICAL_P,
    DEFAULT_BACKEND_TYPE,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_TOOL_FORMAT,
    DEFAULT_TOOL_MULTI_TURN_CHAT,
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
    DEFAULT_GENERIC_OPENAI_PATH,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_THREAD_COUNT,
    DEFAULT_BATCH_THREAD_COUNT,
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT,
    TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT,
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT,
    DOMAIN,
    HOME_LLM_API_ID,
    SERVICE_TOOL_NAME,
    PROMPT_TEMPLATE_DESCRIPTIONS,
    TOOL_FORMAT_FULL,
    TOOL_FORMAT_REDUCED,
    TOOL_FORMAT_MINIMAL,
    ALLOWED_SERVICE_CALL_ARGUMENTS,
)

# make type checking work for llama-cpp-python without importing it directly at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llama_cpp import Llama as LlamaType
else:
    LlamaType = Any

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

class LocalLLMAgent(AbstractConversationAgent):
    """Base Local LLM conversation agent."""

    hass: HomeAssistant
    entry_id: str
    history: dict[str, list[dict]]
    in_context_examples: list[dict]

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry_id = entry.entry_id
        self.history = {}

        self.backend_type = entry.data.get(
            CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE
        )

        self.in_context_examples = None
        if entry.options.get(CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES, DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES):
            self._load_icl_examples(entry.options.get(CONF_IN_CONTEXT_EXAMPLES_FILE, DEFAULT_IN_CONTEXT_EXAMPLES_FILE))

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

    async def _async_load_model(self, entry: ConfigEntry) -> str:
        """Default implementation is to call _load_model() which probably does blocking stuff"""
        return await self.hass.async_add_executor_job(
            self._load_model, entry
        )
    
    def _generate(self, conversation: dict) -> str:
        """Call the backend to generate a response from the conversation. Implemented by sub-classes"""
        raise NotImplementedError()

    async def _async_generate(self, conversation: dict) -> str:
        """Default implementation is to call _generate() which probably does blocking stuff"""
        return await self.hass.async_add_executor_job(
            self._generate, conversation
        )
    
    def _warn_context_size(self):
        num_entities = len(self._async_get_exposed_entities()[0])
        context_size = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        _LOGGER.error("There were too many entities exposed when attempting to generate a response for " +
                      f"{self.entry.data[CONF_CHAT_MODEL]} and it exceeded the context size for the model. " +
                      f"Please reduce the number of entities exposed ({num_entities}) or increase the model's context size ({int(context_size)})")

    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process a sentence."""

        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        prompt_template = self.entry.options.get(CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE)
        template_desc = PROMPT_TEMPLATE_DESCRIPTIONS[prompt_template]
        refresh_system_prompt = self.entry.options.get(CONF_REFRESH_SYSTEM_PROMPT, DEFAULT_REFRESH_SYSTEM_PROMPT)
        remember_conversation = self.entry.options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION)
        remember_num_interactions = self.entry.options.get(CONF_REMEMBER_NUM_INTERACTIONS, DEFAULT_REMEMBER_NUM_INTERACTIONS)
        service_call_regex = self.entry.options.get(CONF_SERVICE_CALL_REGEX, DEFAULT_SERVICE_CALL_REGEX)

        try:
            service_call_pattern = re.compile(service_call_regex)
        except Exception as err:
            _LOGGER.exception("There was a problem compiling the service call regex")
            
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, there was a problem compiling the service call regex: {err}",
            )
            return ConversationResult(
                response=intent_response, conversation_id=conversation_id
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
                        user_prompt=user_input.text,
                        language=user_input.language,
                        assistant=ha_conversation.DOMAIN,
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

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            conversation = self.history[conversation_id] if remember_conversation else [self.history[conversation_id][0]]
        else:
            conversation_id = ulid.ulid()
            conversation = []
        
        if len(conversation) == 0 or refresh_system_prompt:
            try:
                message = self._generate_system_prompt(raw_prompt, llm_api)
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            
            system_prompt = { "role": "system", "message": message }
            
            if len(conversation) == 0:
                conversation.append(system_prompt)
                if not remember_conversation:
                    self.history[conversation_id] = conversation
            else:
                conversation[0] = system_prompt

        conversation.append({"role": "user", "message": user_input.text})

        # generate a response
        try:
            _LOGGER.debug(conversation)
            response = await self._async_generate(conversation)
            _LOGGER.debug(response)

        except Exception as err:
            _LOGGER.exception("There was a problem talking to the backend")
            
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                f"Sorry, there was a problem talking to the backend: {repr(err)}",
            )
            return ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        
        # remove end of text token if it was returned
        response = response.replace(template_desc["assistant"]["suffix"], "")

        conversation.append({"role": "assistant", "message": response})
        if remember_conversation:
            if remember_num_interactions and len(conversation) > (remember_num_interactions * 2) + 1:
                for i in range(0,2):
                    conversation.pop(1)
            self.history[conversation_id] = conversation

        if llm_api is None:
            # return the output without messing with it if there is no API exposed to the model
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(response.strip())
            return ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        # parse response
        to_say = service_call_pattern.sub("", response.strip())
        for block in service_call_pattern.findall(response.strip()):
            parsed_tool_call: dict = json.loads(block)

            if llm_api.api.id == HOME_LLM_API_ID:
                schema_to_validate = vol.Schema({
                    vol.Required('service'): str,
                    vol.Required('target_device'): str,
                    vol.Optional('rgb_color'): str,
                    vol.Optional('brightness'): float,
                    vol.Optional('temperature'): float,
                    vol.Optional('humidity'): float,
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
                return ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )

            _LOGGER.info(f"calling tool: {block}")

            # try to fix certain arguments
            args_dict = parsed_tool_call if llm_api.api.id == HOME_LLM_API_ID else parsed_tool_call["arguments"]

            # make sure brightness is 0-255 and not a percentage
            if "brightness" in args_dict and 0.0 < args_dict["brightness"] <= 1.0:
                args_dict["brightness"] = int(args_dict["brightness"] * 255)

            # convert string "tuple" to a list for RGB colors
            if "rgb_color" in args_dict and isinstance(args_dict["rgb_color"], str):
                args_dict["rgb_color"] = [ int(x) for x in args_dict["rgb_color"][1:-1].split(",") ]
            
            if llm_api.api.id == HOME_LLM_API_ID:
                to_say = to_say + parsed_tool_call.pop("to_say", "")
                tool_input = llm.ToolInput(
                    tool_name=SERVICE_TOOL_NAME,
                    tool_args=parsed_tool_call,
                )
            else:
                tool_input = llm.ToolInput(
                    tool_name=parsed_tool_call["name"],
                    tool_args=parsed_tool_call["arguments"],
                )

            try:
                tool_response = await llm_api.async_call_tool(tool_input)
                _LOGGER.debug("Tool response: %s", tool_response)
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
                    response=intent_response, conversation_id=conversation_id
                )

        # handle models that generate a function call and wait for the result before providing a response
        if self.entry.options.get(CONF_TOOL_MULTI_TURN_CHAT, DEFAULT_TOOL_MULTI_TURN_CHAT):
            conversation.append({"role": "tool", "message": json.dumps(tool_response)})

            # generate a response based on the tool result
            try:
                _LOGGER.debug(conversation)
                to_say = await self._async_generate(conversation)
                _LOGGER.debug(to_say)

            except Exception as err:
                _LOGGER.exception("There was a problem talking to the backend")
                
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                    f"Sorry, there was a problem talking to the backend: {repr(err)}",
                )
                return ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )

            conversation.append({"role": "assistant", "message": response})
        
        # generate intent response to Home Assistant
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(to_say.strip())
        return ConversationResult(
            response=intent_response, conversation_id=conversation_id
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
                area = area_registry.async_get_area(entity.area_id)
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

    def _generate_system_prompt(self, prompt_template: str, llm_api: llm.APIInstance) -> str:
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
                    if attribute_name == "temperature":
                        value = int(value)
                        if value > 50:
                            value = f"{value}F"
                        else:
                            value = f"{value}C"
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

class LlamaCppAgent(LocalLLMAgent):
    model_path: str
    llm: LlamaType
    grammar: Any
    llama_cpp_module: Any
    remove_prompt_caching_listener: Callable
    model_lock: threading.Lock
    last_cache_prime: float
    last_updated_entities: dict[str, float]
    cache_refresh_after_cooldown: bool
    loaded_model_settings: dict[str, Any]

    def _load_model(self, entry: ConfigEntry) -> None:
        self.model_path = entry.data.get(CONF_DOWNLOADED_MODEL_FILE)

        _LOGGER.info(
            "Using model file '%s'", self.model_path
        )

        if not self.model_path:
            raise Exception(f"Model was not found at '{self.model_path}'!")
        
        validate_llama_cpp_python_installation()

        # don't import it until now because the wheel is installed by config_flow.py
        try:
            self.llama_cpp_module = importlib.import_module("llama_cpp")
        except ModuleNotFoundError:
            # attempt to re-install llama-cpp-python if it was uninstalled for some reason
            install_result = install_llama_cpp_python(self.hass.config.config_dir)
            if not install_result == True:
                raise ConfigEntryError("llama-cpp-python was not installed on startup and re-installing it led to an error!")
            
            validate_llama_cpp_python_installation()
            self.llama_cpp_module = importlib.import_module("llama_cpp")
            
        Llama = getattr(self.llama_cpp_module, "Llama")

        _LOGGER.debug(f"Loading model '{self.model_path}'...")
        self.loaded_model_settings = {}
        self.loaded_model_settings[CONF_CONTEXT_LENGTH] = entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        self.loaded_model_settings[CONF_BATCH_SIZE] = entry.options.get(CONF_BATCH_SIZE, DEFAULT_BATCH_SIZE)
        self.loaded_model_settings[CONF_THREAD_COUNT] = entry.options.get(CONF_THREAD_COUNT, DEFAULT_THREAD_COUNT)
        self.loaded_model_settings[CONF_BATCH_THREAD_COUNT] = entry.options.get(CONF_BATCH_THREAD_COUNT, DEFAULT_BATCH_THREAD_COUNT)
        self.loaded_model_settings[CONF_ENABLE_FLASH_ATTENTION] = entry.options.get(CONF_ENABLE_FLASH_ATTENTION, DEFAULT_ENABLE_FLASH_ATTENTION)

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=int(self.loaded_model_settings[CONF_CONTEXT_LENGTH]),
            n_batch=int(self.loaded_model_settings[CONF_BATCH_SIZE]),
            n_threads=int(self.loaded_model_settings[CONF_THREAD_COUNT]),
            n_threads_batch=int(self.loaded_model_settings[CONF_BATCH_THREAD_COUNT]),
            flash_attn=self.loaded_model_settings[CONF_ENABLE_FLASH_ATTENTION],
        )
        _LOGGER.debug("Model loaded")

        self.grammar = None
        if entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            self._load_grammar(entry.options.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE))
            

        # TODO: check about disk caching
        # self.llm.set_cache(self.llama_cpp_module.LlamaDiskCache(
        #     capacity_bytes=(512 * 10e8),
        #     cache_dir=os.path.join(self.hass.config.media_dirs.get("local", self.hass.config.path("media")), "kv_cache")
        # ))
        
        self.remove_prompt_caching_listener = None
        self.last_cache_prime = None
        self.last_updated_entities = {}
        self.cache_refresh_after_cooldown = False
        self.model_lock = threading.Lock()

        self.loaded_model_settings[CONF_PROMPT_CACHING_ENABLED] = entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED)
        if self.loaded_model_settings[CONF_PROMPT_CACHING_ENABLED]:
            @callback
            async def enable_caching_after_startup(_now) -> None:
                self._set_prompt_caching(enabled=True)
                await self._async_cache_prompt(None, None, None)
            async_call_later(self.hass, 5.0, enable_caching_after_startup)

    def _load_grammar(self, filename: str):
        LlamaGrammar = getattr(self.llama_cpp_module, "LlamaGrammar")
        _LOGGER.debug(f"Loading grammar {filename}...")
        try:
            with open(os.path.join(os.path.dirname(__file__), filename)) as f:
                grammar_str = "".join(f.readlines())
            self.grammar = LlamaGrammar.from_string(grammar_str)
            self.loaded_model_settings[CONF_GBNF_GRAMMAR_FILE] = filename
            _LOGGER.debug("Loaded grammar")
        except Exception:
            _LOGGER.exception("Failed to load grammar!")
            self.grammar = None

    def _update_options(self):
        LocalLLMAgent._update_options(self)

        model_reloaded = False
        if self.loaded_model_settings[CONF_CONTEXT_LENGTH] != self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH) or \
            self.loaded_model_settings[CONF_BATCH_SIZE] != self.entry.options.get(CONF_BATCH_SIZE, DEFAULT_BATCH_SIZE) or \
            self.loaded_model_settings[CONF_THREAD_COUNT] != self.entry.options.get(CONF_THREAD_COUNT, DEFAULT_THREAD_COUNT) or \
            self.loaded_model_settings[CONF_BATCH_THREAD_COUNT] != self.entry.options.get(CONF_BATCH_THREAD_COUNT, DEFAULT_BATCH_THREAD_COUNT) or \
            self.loaded_model_settings[CONF_ENABLE_FLASH_ATTENTION] != self.entry.options.get(CONF_ENABLE_FLASH_ATTENTION, DEFAULT_ENABLE_FLASH_ATTENTION):

            _LOGGER.debug(f"Reloading model '{self.model_path}'...")
            self.loaded_model_settings[CONF_CONTEXT_LENGTH] = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
            self.loaded_model_settings[CONF_BATCH_SIZE] = self.entry.options.get(CONF_BATCH_SIZE, DEFAULT_BATCH_SIZE)
            self.loaded_model_settings[CONF_THREAD_COUNT] = self.entry.options.get(CONF_THREAD_COUNT, DEFAULT_THREAD_COUNT)
            self.loaded_model_settings[CONF_BATCH_THREAD_COUNT] = self.entry.options.get(CONF_BATCH_THREAD_COUNT, DEFAULT_BATCH_THREAD_COUNT)
            self.loaded_model_settings[CONF_ENABLE_FLASH_ATTENTION] = self.entry.options.get(CONF_ENABLE_FLASH_ATTENTION, DEFAULT_ENABLE_FLASH_ATTENTION)

            Llama = getattr(self.llama_cpp_module, "Llama")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=int(self.loaded_model_settings[CONF_CONTEXT_LENGTH]),
                n_batch=int(self.loaded_model_settings[CONF_BATCH_SIZE]),
                n_threads=int(self.loaded_model_settings[CONF_THREAD_COUNT]),
                n_threads_batch=int(self.loaded_model_settings[CONF_BATCH_THREAD_COUNT]),
                flash_attn=self.loaded_model_settings[CONF_ENABLE_FLASH_ATTENTION],
            )
            _LOGGER.debug("Model loaded")
            model_reloaded = True

        if self.entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            current_grammar = self.entry.options.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE)
            if not self.grammar or self.loaded_model_settings[CONF_GBNF_GRAMMAR_FILE] != current_grammar:
                self._load_grammar(current_grammar)
        else:
            self.grammar = None

        if self.entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED):
            self._set_prompt_caching(enabled=True)

            if self.loaded_model_settings[CONF_PROMPT_CACHING_ENABLED] != self.entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED) or \
                model_reloaded:
                self.loaded_model_settings[CONF_PROMPT_CACHING_ENABLED] = self.entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED)
            
                async def cache_current_prompt(_now):
                    await self._async_cache_prompt(None, None, None)
                async_call_later(self.hass, 1.0, cache_current_prompt)
        else:
            self._set_prompt_caching(enabled=False)

    def _async_get_exposed_entities(self) -> tuple[dict[str, str], list[str]]:
        """Takes the super class function results and sorts the entities with the recently updated at the end"""
        entities, domains = LocalLLMAgent._async_get_exposed_entities(self)

        # ignore sorting if prompt caching is disabled
        if not self.entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED):
            return entities, domains
        
        entity_order = { name: None for name in entities.keys() }
        entity_order.update(self.last_updated_entities)

        def sort_key(item):
            item_name, last_updated = item
            # Handle cases where last updated is None
            if last_updated is None:
                return (False, '', item_name)
            else:
                return (True, last_updated, '')
        
        # Sort the items based on the sort_key function
        sorted_items = sorted(list(entity_order.items()), key=sort_key)

        _LOGGER.debug(f"sorted_items: {sorted_items}")

        sorted_entities = {}
        for item_name, _ in sorted_items:
            sorted_entities[item_name] = entities[item_name]

        return sorted_entities, domains

    def _set_prompt_caching(self, *, enabled=True):
        if enabled and not self.remove_prompt_caching_listener:
            _LOGGER.info("enabling prompt caching...")

            entity_ids = [ 
                state.entity_id for state in self.hass.states.async_all() \
                    if async_should_expose(self.hass, CONVERSATION_DOMAIN, state.entity_id) 
            ]

            _LOGGER.debug(f"watching entities: {entity_ids}")

            self.remove_prompt_caching_listener = async_track_state_change(self.hass, entity_ids, self._async_cache_prompt)

        elif not enabled and self.remove_prompt_caching_listener:
            _LOGGER.info("disabling prompt caching...")
            self.remove_prompt_caching_listener()

    @callback
    async def _async_cache_prompt(self, entity, old_state, new_state):
        refresh_start = time.time()

        # track last update time so we can sort the context efficiently
        if entity:
            self.last_updated_entities[entity] = refresh_start

        llm_api: llm.APIInstance | None = None
        if self.entry.options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    self.hass, self.entry.options[CONF_LLM_HASS_API]
                )
            except HomeAssistantError:
                _LOGGER.exception("Failed to get LLM API when caching prompt!")
                return

        _LOGGER.debug(f"refreshing cached prompt because {entity} changed...")
        await self.hass.async_add_executor_job(self._cache_prompt, llm_api)

        refresh_end = time.time()
        _LOGGER.debug(f"cache refresh took {(refresh_end - refresh_start):.2f} sec")

    def _cache_prompt(self, llm_api: llm.API) -> None:
        # if a refresh is already scheduled then exit
        if self.cache_refresh_after_cooldown:
            return
        
        # if we are inside the cooldown period, request a refresh and exit
        current_time = time.time()
        fastest_prime_interval = self.entry.options.get(CONF_PROMPT_CACHING_INTERVAL, DEFAULT_PROMPT_CACHING_INTERVAL)
        if self.last_cache_prime and current_time - self.last_cache_prime < fastest_prime_interval:
            self.cache_refresh_after_cooldown = True
            return
        
        # try to acquire the lock, if we are still running for some reason, request a refresh and exit
        lock_acquired = self.model_lock.acquire(False)
        if not lock_acquired:
            self.cache_refresh_after_cooldown = True
            return
        
        try:
            raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
            prompt = self._format_prompt([
                { "role": "system", "message": self._generate_system_prompt(raw_prompt, llm_api)},
                { "role": "user", "message": "" }
            ], include_generation_prompt=False)
        
            input_tokens = self.llm.tokenize(
                prompt.encode(), add_bos=False
            )

            temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
            top_k = int(self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K))
            top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
            grammar = self.grammar if self.entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR) else None

            _LOGGER.debug(f"Options: {self.entry.options}")

            _LOGGER.debug(f"Processing {len(input_tokens)} input tokens...")

            # grab just one token. should prime the kv cache with the system prompt
            next(self.llm.generate(
                input_tokens,
                temp=temperature,
                top_k=top_k,
                top_p=top_p,
                grammar=grammar
            ))

            self.last_cache_prime = time.time()
        finally:
            self.model_lock.release()

        
        # schedule a refresh using async_call_later
        # if the flag is set after the delay then we do another refresh
        
        @callback
        async def refresh_if_requested(_now):
            if self.cache_refresh_after_cooldown:
                self.cache_refresh_after_cooldown = False

                refresh_start = time.time()
                _LOGGER.debug(f"refreshing cached prompt after cooldown...")
                await self.hass.async_add_executor_job(self._cache_prompt)

                refresh_end = time.time()
                _LOGGER.debug(f"cache refresh took {(refresh_end - refresh_start):.2f} sec")

        refresh_delay = self.entry.options.get(CONF_PROMPT_CACHING_INTERVAL, DEFAULT_PROMPT_CACHING_INTERVAL)
        async_call_later(self.hass, float(refresh_delay), refresh_if_requested)
        
    
    def _generate(self, conversation: dict) -> str:
        prompt = self._format_prompt(conversation)

        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_k = int(self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K))
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        min_p = self.entry.options.get(CONF_MIN_P, DEFAULT_MIN_P)
        typical_p = self.entry.options.get(CONF_TYPICAL_P, DEFAULT_TYPICAL_P)

        _LOGGER.debug(f"Options: {self.entry.options}")

        with self.model_lock:
            input_tokens = self.llm.tokenize(
                prompt.encode(), add_bos=False
            )

            context_len = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
            if len(input_tokens) + max_tokens >= context_len:
                self._warn_context_size()

            _LOGGER.debug(f"Processing {len(input_tokens)} input tokens...")
            output_tokens = self.llm.generate(
                input_tokens,
                temp=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                grammar=self.grammar
            )

            result_tokens = []
            for token in output_tokens:
                if token == self.llm.token_eos():
                    break

                result_tokens.append(token)

                if len(result_tokens) >= max_tokens:
                    break

            result = self.llm.detokenize(result_tokens).decode()

        return result
    
class GenericOpenAIAPIAgent(LocalLLMAgent):
    api_host: str
    api_key: str
    model_name: str

    async def _async_load_model(self, entry: ConfigEntry) -> None:
        self.api_host = format_url(
            hostname=entry.data[CONF_HOST],
            port=entry.data[CONF_PORT],
            ssl=entry.data[CONF_SSL],
            path=""
        )
        
        self.api_key = entry.data.get(CONF_OPENAI_API_KEY)
        self.model_name = entry.data.get(CONF_CHAT_MODEL)


    def _chat_completion_params(self, conversation: dict) -> (str, dict):
        request_params = {}
        api_base_path = self.entry.options.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)

        endpoint = f"/{api_base_path}/chat/completions"
        request_params["messages"] = [ { "role": x["role"], "content": x["message"] } for x in conversation ]

        return endpoint, request_params

    def _completion_params(self, conversation: dict) -> (str, dict):
        request_params = {}
        api_base_path = self.entry.options.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)

        endpoint = f"/{api_base_path}/completions"
        request_params["prompt"] = self._format_prompt(conversation)

        return endpoint, request_params
    
    def _extract_response(self, response_json: dict) -> str:
        choices = response_json["choices"]
        if choices[0]["finish_reason"] != "stop":
            _LOGGER.warning("Model response did not end on a stop token (unfinished sentence)")

        if response_json["object"] in ["chat.completion", "chat.completion.chunk"]:
            return choices[0]["message"]["content"]
        else:
            return choices[0]["text"]
    
    async def _async_generate(self, conversation: dict) -> str:
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        timeout = self.entry.options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        use_chat_api = self.entry.options.get(CONF_REMOTE_USE_CHAT_ENDPOINT, DEFAULT_REMOTE_USE_CHAT_ENDPOINT)
        

        request_params = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if use_chat_api:
            endpoint, additional_params = self._chat_completion_params(conversation)
        else:
            endpoint, additional_params = self._completion_params(conversation)
        
        request_params.update(additional_params)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        session = async_get_clientsession(self.hass)
        response = None
        try:
            async with session.post(
                f"{self.api_host}{endpoint}",
                json=request_params,
                timeout=timeout,
                headers=headers
            ) as response:
                response.raise_for_status()
                result = await response.json()
        except asyncio.TimeoutError:
            return "The generation request timed out! Please check your connection settings, increase the timeout in settings, or decrease the number of exposed entities."
        except aiohttp.ClientError as err:
            _LOGGER.debug(f"Err was: {err}")
            _LOGGER.debug(f"Request was: {request_params}")
            _LOGGER.debug(f"Result was: {response}")
            return f"Failed to communicate with the API! {err}"

        _LOGGER.debug(result)

        return self._extract_response(result)
        
class TextGenerationWebuiAgent(GenericOpenAIAPIAgent):
    admin_key: str

    async def _async_load_model(self, entry: ConfigEntry) -> None:
        await super()._async_load_model(entry)
        self.admin_key = entry.data.get(CONF_TEXT_GEN_WEBUI_ADMIN_KEY, self.api_key)

        try:
            headers = {}
            session = async_get_clientsession(self.hass)

            if self.admin_key:
                headers["Authorization"] = f"Bearer {self.admin_key}"
            
            async with session.get(
                f"{self.api_host}/v1/internal/model/info",
                headers=headers
            ) as response:
                response.raise_for_status()
                currently_loaded_result = await response.json()
            
            loaded_model = currently_loaded_result["model_name"]
            if loaded_model == self.model_name:
                _LOGGER.info(f"Model {self.model_name} is already loaded on the remote backend.")
                return
            else:
                _LOGGER.info(f"Model is not {self.model_name} loaded on the remote backend. Loading it now...")
            
            async with session.post(
                f"{self.api_host}/v1/internal/model/load",
                json={
                    "model_name": self.model_name,
                    # TODO: expose arguments to the user in home assistant UI
                    # "args": {},
                },
                headers=headers
            ) as response:
                response.raise_for_status()

        except Exception as ex:
            _LOGGER.debug("Connection error was: %s", repr(ex))
            raise ConfigEntryNotReady("There was a problem connecting to the remote server") from ex

    def _chat_completion_params(self, conversation: dict) -> (str, dict):
        preset = self.entry.options.get(CONF_TEXT_GEN_WEBUI_PRESET)
        chat_mode = self.entry.options.get(CONF_TEXT_GEN_WEBUI_CHAT_MODE, DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE)

        endpoint, request_params = super()._chat_completion_params(conversation)

        request_params["mode"] = chat_mode
        if chat_mode == TEXT_GEN_WEBUI_CHAT_MODE_CHAT or chat_mode == TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT:
            if preset:
                request_params["character"] = preset
        elif chat_mode == TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT:
            request_params["instruction_template"] = self.entry.options.get(CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE)

        request_params["truncation_length"] = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        request_params["top_k"] = self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K)
        request_params["min_p"] = self.entry.options.get(CONF_MIN_P, DEFAULT_MIN_P)
        request_params["typical_p"] = self.entry.options.get(CONF_TYPICAL_P, DEFAULT_TYPICAL_P)

        return endpoint, request_params
    
    def _completion_params(self, conversation: dict) -> (str, dict):
        preset = self.entry.options.get(CONF_TEXT_GEN_WEBUI_PRESET)

        endpoint, request_params = super()._completion_params(conversation)

        if preset:
            request_params["preset"] = preset

        request_params["truncation_length"] = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        request_params["top_k"] = self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K)
        request_params["min_p"] = self.entry.options.get(CONF_MIN_P, DEFAULT_MIN_P)
        request_params["typical_p"] = self.entry.options.get(CONF_TYPICAL_P, DEFAULT_TYPICAL_P)

        return endpoint, request_params
    
    def _extract_response(self, response_json: dict) -> str:
        choices = response_json["choices"]
        if choices[0]["finish_reason"] != "stop":
            _LOGGER.warning("Model response did not end on a stop token (unfinished sentence)")

        context_len = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        if response_json["usage"]["prompt_tokens"] + max_tokens > context_len:
            self._warn_context_size()

        # text-gen-webui has a typo where it is 'chat.completions' not 'chat.completion'
        if response_json["object"] == "chat.completions":
            return choices[0]["message"]["content"]
        else:
            return choices[0]["text"]
        
class LlamaCppPythonAPIAgent(GenericOpenAIAPIAgent):
    """https://llama-cpp-python.readthedocs.io/en/latest/server/"""
    grammar: str

    async def _async_load_model(self, entry: ConfigEntry):
        await super()._async_load_model(entry)

        return await self.hass.async_add_executor_job(
            self._load_model, entry
        )

    def _load_model(self, entry: ConfigEntry):
        with open(os.path.join(os.path.dirname(__file__), DEFAULT_GBNF_GRAMMAR_FILE)) as f:
            self.grammar = "".join(f.readlines())

    def _chat_completion_params(self, conversation: dict) -> (str, dict):
        top_k = int(self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K))
        endpoint, request_params = super()._chat_completion_params(conversation)

        request_params["top_k"] = top_k

        if self.entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            request_params["grammar"] = self.grammar

        return endpoint, request_params
    
    def _completion_params(self, conversation: dict) -> (str, dict):
        top_k = int(self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K))
        endpoint, request_params = super()._completion_params(conversation)

        request_params["top_k"] = top_k
        
        if self.entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            request_params["grammar"] = self.grammar

        return endpoint, request_params

class OllamaAPIAgent(LocalLLMAgent):
    api_host: str
    api_key: str
    model_name: str

    async def _async_load_model(self, entry: ConfigEntry) -> None:
        self.api_host = format_url(
            hostname=entry.data[CONF_HOST],
            port=entry.data[CONF_PORT],
            ssl=entry.data[CONF_SSL],
            path=""
        )
        self.api_key = entry.data.get(CONF_OPENAI_API_KEY)
        self.model_name = entry.data.get(CONF_CHAT_MODEL)

        # ollama handles loading for us so just make sure the model is available
        try:
            headers = {}
            session = async_get_clientsession(self.hass)
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with session.get(
                f"{self.api_host}/api/tags",
                headers=headers,
            ) as response:
                response.raise_for_status()
                currently_downloaded_result = await response.json()
                
        except Exception as ex:
            _LOGGER.debug("Connection error was: %s", repr(ex))
            raise ConfigEntryNotReady("There was a problem connecting to the remote server") from ex

        model_names = [ x["name"] for x in currently_downloaded_result["models"] ]
        if ":" in self.model_name:
            if not any([ name == self.model_name for name in model_names]):
                raise ConfigEntryNotReady(f"Ollama server does not have the provided model: {self.model_name}")    
        elif not any([ name.split(":")[0] == self.model_name for name in model_names ]):
            raise ConfigEntryNotReady(f"Ollama server does not have the provided model: {self.model_name}")

    def _chat_completion_params(self, conversation: dict) -> (str, dict):
        request_params = {}

        endpoint = "/api/chat"
        request_params["messages"] = [ { "role": x["role"], "content": x["message"] } for x in conversation ]

        return endpoint, request_params

    def _completion_params(self, conversation: dict) -> (str, dict):
        request_params = {}

        endpoint = "/api/generate"
        request_params["prompt"] = self._format_prompt(conversation)
        request_params["raw"] = True # ignore prompt template

        return endpoint, request_params
    
    def _extract_response(self, response_json: dict) -> str:        
        if response_json["done"] not in ["true", True]:
            _LOGGER.warning("Model response did not end on a stop token (unfinished sentence)")
        
        # TODO: this doesn't work because ollama caches prompts and doesn't always return the full prompt length
        # context_len = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        # max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        # if response_json["prompt_eval_count"] + max_tokens > context_len:
        #     self._warn_context_size()

        if "response" in response_json:
            return response_json["response"]
        else:
            return response_json["message"]["content"]
    
    async def _async_generate(self, conversation: dict) -> str:
        context_length = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        top_k = self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K)
        typical_p = self.entry.options.get(CONF_TYPICAL_P, DEFAULT_TYPICAL_P)
        timeout = self.entry.options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        keep_alive = self.entry.options.get(CONF_OLLAMA_KEEP_ALIVE_MIN, DEFAULT_OLLAMA_KEEP_ALIVE_MIN)
        use_chat_api = self.entry.options.get(CONF_REMOTE_USE_CHAT_ENDPOINT, DEFAULT_REMOTE_USE_CHAT_ENDPOINT)
        json_mode = self.entry.options.get(CONF_OLLAMA_JSON_MODE, DEFAULT_OLLAMA_JSON_MODE)
        
        request_params = {
            "model": self.model_name,
            "stream": False,
            "keep_alive": f"{keep_alive}m", # prevent ollama from unloading the model
            "options": {
                "num_ctx": context_length,
                "top_p": top_p,
                "top_k": top_k,
                "typical_p": typical_p,
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if json_mode:
            request_params["format"] = "json"
        
        if use_chat_api:
            endpoint, additional_params = self._chat_completion_params(conversation)
        else:
            endpoint, additional_params = self._completion_params(conversation)
        
        request_params.update(additional_params)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        session = async_get_clientsession(self.hass)
        response = None
        try:
            async with session.post(
                f"{self.api_host}{endpoint}",
                json=request_params,
                timeout=timeout,
                headers=headers
            ) as response:
                response.raise_for_status()
                result = await response.json()
        except asyncio.TimeoutError:
            return "The generation request timed out! Please check your connection settings, increase the timeout in settings, or decrease the number of exposed entities."
        except aiohttp.ClientError as err:
            _LOGGER.debug(f"Err was: {err}")
            _LOGGER.debug(f"Request was: {request_params}")
            _LOGGER.debug(f"Result was: {response}")
            return f"Failed to communicate with the API! {err}"
        
        _LOGGER.debug(result)

        return self._extract_response(result)
