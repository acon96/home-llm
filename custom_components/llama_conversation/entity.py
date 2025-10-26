"""Defines the base logic for exposing a local LLM as an entity."""
from __future__ import annotations

import csv
import logging
import os
import random
from typing import Literal, Any, List, Dict, Optional, Tuple, AsyncIterator, Generator, AsyncGenerator
from dataclasses import dataclass

from homeassistant.components.conversation import ConversationInput, ConversationResult
from homeassistant.components import conversation
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import MATCH_ALL, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError, HomeAssistantError
from homeassistant.helpers import intent, template, entity_registry as er, llm, \
    area_registry as ar, device_registry as dr, entity
from homeassistant.util import color

from .utils import closest_color, parse_raw_tool_call, flatten_vol_schema, MalformedToolCallException
from .const import (
    CONF_CHAT_MODEL,
    CONF_SELECTED_LANGUAGE,
    CONF_PROMPT,
    CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
    CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    CONF_IN_CONTEXT_EXAMPLES_FILE,
    CONF_NUM_IN_CONTEXT_EXAMPLES,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_THINKING_PREFIX,
    CONF_THINKING_SUFFIX,
    CONF_TOOL_CALL_PREFIX,
    CONF_TOOL_CALL_SUFFIX,
    CONF_ENABLE_LEGACY_TOOL_CALLING,
    DEFAULT_PROMPT,
    DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
    DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    DEFAULT_IN_CONTEXT_EXAMPLES_FILE,
    DEFAULT_NUM_IN_CONTEXT_EXAMPLES,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_NUM_INTERACTIONS,
    DEFAULT_MAX_TOOL_CALL_ITERATIONS,
    DOMAIN,
    DEFAULT_THINKING_PREFIX,
    DEFAULT_THINKING_SUFFIX,
    DEFAULT_TOOL_CALL_PREFIX,
    DEFAULT_TOOL_CALL_SUFFIX,
    DEFAULT_ENABLE_LEGACY_TOOL_CALLING,
    HOME_LLM_API_ID,
    SERVICE_TOOL_NAME,
)

_LOGGER = logging.getLogger(__name__)

type LocalLLMConfigEntry = ConfigEntry[LocalLLMClient]


@dataclass(kw_only=True)
class TextGenerationResult:
    response: Optional[str] = None
    stop_reason: Optional[str] = None
    tool_calls: Optional[List[llm.ToolInput]] = None
    response_streamed: bool = False
    raise_error: bool = False
    error_msg: Optional[str] = None

class LocalLLMClient:
    """Base Local LLM conversation agent."""

    hass: HomeAssistant
    in_context_examples: Optional[List[Dict[str, str]]]

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]) -> None:
        self.hass = hass

        self.in_context_examples = None
        if client_options.get(CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES, DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES):
            icl_examples_filename = client_options.get(CONF_IN_CONTEXT_EXAMPLES_FILE, DEFAULT_IN_CONTEXT_EXAMPLES_FILE)
            if icl_examples_filename:
                self._load_icl_examples(icl_examples_filename)

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        raise NotImplementedError()

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
    
    def _update_options(self, entity_options: Dict[str, Any]):
        if entity_options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

        if entity_options.get(CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES, DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES):
            self._load_icl_examples(entity_options.get(CONF_IN_CONTEXT_EXAMPLES_FILE, DEFAULT_IN_CONTEXT_EXAMPLES_FILE))
        else:
            self.in_context_examples = None

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        """Validate connection to the backend. Implemented by sub-classes"""
        return None

    def _load_model(self, entity_options: dict[str, Any]) -> None:
        """Load the model on the backend. Implemented by sub-classes"""
        pass

    async def _async_load_model(self, entity_options: dict[str, Any]) -> None:
        """Default implementation is to call _load_model() which probably does blocking stuff"""
        await self.hass.async_add_executor_job(
            self._load_model, entity_options
        )

    def _unload_model(self, entity_options: dict[str, Any]) -> None:
        """Unload the model to free up space on the backend. Implemented by sub-classes"""
        pass

    async def _async_unload_model(self, entity_options: dict[str, Any]) -> None:
        """Default implementation is to call _unload_model() which probably does blocking stuff"""
        await self.hass.async_add_executor_job(
            self._unload_model, entity_options
        )
    
    def _generate_stream(self, conversation: List[conversation.Content], llm_api: llm.APIInstance | None, user_input: conversation.ConversationInput, entity_options: dict[str, Any]) -> AsyncGenerator[TextGenerationResult, None]:
        """Async generator for streaming responses. Subclasses should implement."""
        raise NotImplementedError()

    async def _generate(self, conversation: List[conversation.Content], llm_api: llm.APIInstance | None, user_input: conversation.ConversationInput, entity_options: dict[str, Any]) -> TextGenerationResult:
        """Call the backend to generate a response from the conversation. Implemented by sub-classes"""
        raise NotImplementedError()

    async def _async_generate(self, conv: List[conversation.Content], user_input: ConversationInput, chat_log: conversation.chat_log.ChatLog, entity_options: dict[str, Any]):
        """Default implementation: if streaming is supported, consume the async generator and return the full result."""
        if hasattr(self, '_generate_stream'):
            # Try to stream and collect the full response
            return await self._transform_result_stream(self._generate_stream(conv, chat_log.llm_api, user_input, entity_options), user_input, chat_log)
        
        # Fallback to "blocking" generate
        blocking_result = await self._generate(conv, chat_log.llm_api, user_input, entity_options)

        return chat_log.async_add_assistant_content(
            conversation.AssistantContent(
                agent_id=user_input.agent_id,
                content=blocking_result.response,
                tool_calls=blocking_result.tool_calls
            )
        )
    
    async def async_get_available_models(self) -> List[str]:
        """Return a list of available models. Implemented by sub-classes"""
        raise NotImplementedError()

    def _warn_context_size(self, model: str, context_size: int):
        num_entities = len(self._async_get_exposed_entities())
        _LOGGER.error("There were too many entities exposed when attempting to generate a response for " +
                      f"{model} and it exceeded the context size for the model. " +
                      f"Please reduce the number of entities exposed ({num_entities}) or increase the model's context size ({context_size})")
        
    async def _transform_result_stream(
        self,
        result: AsyncIterator[TextGenerationResult],
        user_input: ConversationInput,
        chat_log: conversation.chat_log.ChatLog
    ):
        async def async_iterator():
            async for input_chunk in result:
                _LOGGER.debug("Received chunk: %s", input_chunk)

                tool_calls = input_chunk.tool_calls
                # fix tool calls for the service tool
                if tool_calls and chat_log.llm_api and chat_log.llm_api.api.id == HOME_LLM_API_ID:
                    tool_calls = [
                        llm.ToolInput(
                            tool_name=SERVICE_TOOL_NAME,
                            tool_args={**tc.tool_args, "service": tc.tool_name}
                        ) for tc in tool_calls
                    ]
                yield conversation.AssistantContentDeltaDict(
                    content=input_chunk.response,
                    tool_calls=tool_calls
                )
        
        return chat_log.async_add_delta_content_stream(user_input.agent_id, stream=async_iterator())

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        entity_options: Dict[str, Any],
    ) -> conversation.ConversationResult:

        raw_prompt = entity_options.get(CONF_PROMPT, DEFAULT_PROMPT)
        refresh_system_prompt = entity_options.get(CONF_REFRESH_SYSTEM_PROMPT, DEFAULT_REFRESH_SYSTEM_PROMPT)
        remember_conversation = entity_options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION)
        remember_num_interactions = entity_options.get(CONF_REMEMBER_NUM_INTERACTIONS, DEFAULT_REMEMBER_NUM_INTERACTIONS)
        max_tool_call_iterations = entity_options.get(CONF_MAX_TOOL_CALL_ITERATIONS, DEFAULT_MAX_TOOL_CALL_ITERATIONS)

        llm_api: llm.APIInstance | None = None
        if entity_options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    self.hass,
                    entity_options[CONF_LLM_HASS_API],
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
                system_prompt = conversation.SystemContent(content=self._generate_system_prompt(raw_prompt, llm_api, entity_options))
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
            generation_result = await self._async_generate(message_history, user_input, chat_log, entity_options)
            
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
    
    async def _async_parse_completion(
            self, llm_api: llm.APIInstance | None, 
            user_input: ConversationInput,
            entity_options: Dict[str, Any],
            next_token: Optional[Generator[Tuple[Optional[str], Optional[List]]]] = None,
            anext_token: Optional[AsyncGenerator[Tuple[Optional[str], Optional[List]]]] = None,
        ) -> AsyncGenerator[TextGenerationResult, None]:
        think_prefix = entity_options.get(CONF_THINKING_PREFIX, DEFAULT_THINKING_PREFIX)
        think_suffix = entity_options.get(CONF_THINKING_SUFFIX, DEFAULT_THINKING_SUFFIX)
        tool_prefix = entity_options.get(CONF_TOOL_CALL_PREFIX, DEFAULT_TOOL_CALL_PREFIX)
        tool_suffix = entity_options.get(CONF_TOOL_CALL_SUFFIX, DEFAULT_TOOL_CALL_SUFFIX)

        token_generator = None
        if next_token:
            async def async_generator_wrapper() -> AsyncGenerator[Tuple[Optional[str], Optional[List]]]:
                try:
                    result = (None, None)
                    while result:
                        result = await self.hass.async_add_executor_job(lambda: next(next_token, None))
                        if result and (result[0] or result[1]):
                            yield result
                except StopIteration:
                    return
            token_generator = async_generator_wrapper()
        elif anext_token:
            token_generator = anext_token

        if not token_generator:
            raise Exception("Either next_token or anext_token must be provided")
        
        in_thinking = False
        in_tool_call = False
        tool_content = ""
        last_5_tokens = []
        cur_match_length = 0
        async for chunk in token_generator:
            # _LOGGER.debug(f"Handling chunk: {chunk} {in_thinking=} {in_tool_call=} {last_5_tokens=}")
            tool_calls: Optional[List[str | llm.ToolInput | dict]]
            content, tool_calls = chunk

            if not tool_calls:
                tool_calls = []

            result = TextGenerationResult(
                response=None,
                response_streamed=True,
                tool_calls=None
            )
            if content:
                last_5_tokens.append(content)
                if len(last_5_tokens) > 5:
                    last_5_tokens.pop(0)

                potential_block = "".join(last_5_tokens)
                if tool_prefix.startswith("".join(last_5_tokens[-(cur_match_length+1):])):
                    cur_match_length += 1
                else:
                    # flush the current match length by appending it to content
                    if cur_match_length > 0:
                        content += "".join(last_5_tokens[-cur_match_length:])
                    cur_match_length = 0

                if in_tool_call:
                    tool_content += content

                if think_prefix in potential_block and not in_thinking:
                    in_thinking = True
                    last_5_tokens.clear()
                elif think_suffix in potential_block and in_thinking:
                    in_thinking = False
                    content = content.replace(think_suffix, "").strip()
                elif tool_prefix in potential_block and not in_tool_call:
                    in_tool_call = True
                    last_5_tokens.clear()
                elif tool_suffix in potential_block and in_tool_call:
                    in_tool_call = False
                    tool_block = tool_content.strip().removeprefix(tool_prefix).removesuffix(tool_suffix)
                    _LOGGER.debug("Raw tool block extracted: %s", tool_block)
                    tool_calls.append(tool_block)
                    tool_content = ""

                if cur_match_length == 0:
                    result.response = content
            
            parsed_tool_calls: list[llm.ToolInput] = []
            if tool_calls:
                if not llm_api:
                    _LOGGER.warning("Model attempted to call a tool but no LLM API was provided, ignoring tool calls")
                else:
                    for raw_tool_call in tool_calls:
                        if isinstance(raw_tool_call, llm.ToolInput):
                            parsed_tool_calls.append(raw_tool_call)
                        else:
                            if isinstance(raw_tool_call, str):
                                tool_call, to_say = parse_raw_tool_call(raw_tool_call, llm_api, user_input)
                            else:
                                tool_call, to_say = parse_raw_tool_call(raw_tool_call["function"], llm_api, user_input)

                            if tool_call:
                                _LOGGER.debug("Tool call parsed: %s", tool_call)
                                parsed_tool_calls.append(tool_call)
                            if to_say:
                                result.response = to_say

            if len(parsed_tool_calls) > 0:
                result.tool_calls = parsed_tool_calls

            if not in_thinking and not in_tool_call and (cur_match_length == 0 or result.tool_calls):
                yield result
    
    def _async_get_all_exposed_domains(self) -> list[str]:
        """Gather all exposed domains"""
        domains = set()
        for state in self.hass.states.async_all():
            if async_should_expose(self.hass, CONVERSATION_DOMAIN, state.entity_id):
                domains.add(state.domain)

        return list(domains)

    def _async_get_exposed_entities(self) -> dict[str, dict[str, Any]]:
        """Gather exposed entity states"""
        entity_states: dict[str, dict] = {}
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

        return entity_states

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

    def _generate_system_prompt(self, prompt_template: str, llm_api: llm.APIInstance | None, entity_options: Dict[str, Any],) -> str:
        """Generate the system prompt with current entity states"""
        entities_to_expose = self._async_get_exposed_entities()

        extra_attributes_to_expose = entity_options.get(CONF_EXTRA_ATTRIBUTES_TO_EXPOSE, DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE)
        enable_legacy_tool_calling = entity_options.get(CONF_ENABLE_LEGACY_TOOL_CALLING, DEFAULT_ENABLE_LEGACY_TOOL_CALLING)
        tool_call_prefix = entity_options.get(CONF_TOOL_CALL_PREFIX, DEFAULT_TOOL_CALL_PREFIX)
        tool_call_suffix = entity_options.get(CONF_TOOL_CALL_SUFFIX, DEFAULT_TOOL_CALL_SUFFIX)

        def expose_attributes(attributes: dict[str, Any]) -> list[str]:
            result = []
            for attribute_name in extra_attributes_to_expose:
                if attribute_name not in attributes:
                    continue

                # _LOGGER.debug(f"{attribute_name} = {attributes[attribute_name]}")

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

        render_variables = {
            "devices": devices,
            "formatted_devices": formatted_devices,
            "response_examples": [],
            "tool_call_prefix": tool_call_prefix,
            "tool_call_suffix": tool_call_suffix,
        }

        if enable_legacy_tool_calling:
            if llm_api:
                tools = []
                for tool in llm_api.tools:
                    tools.append(f"{tool.name}({','.join(flatten_vol_schema(tool.parameters))})")
                render_variables["tools"] = tools
                render_variables["formatted_tools"] = ", ".join(tools)
            else:
                message = "No tools were provided. If the user requests you interact with a device, tell them you are unable to do so."
                render_variables["tools"] = [message]
                render_variables["formatted_tools"] = message

        # only pass examples if there are loaded examples + an API was exposed
        if self.in_context_examples and llm_api:
            num_examples = int(entity_options.get(CONF_NUM_IN_CONTEXT_EXAMPLES, DEFAULT_NUM_IN_CONTEXT_EXAMPLES))
            render_variables["response_examples"] = self._generate_icl_examples(num_examples, list(entities_to_expose.keys()))

        return template.Template(prompt_template, self.hass).async_render(
            render_variables,
            parse_result=False,
        )


class LocalLLMEntity(entity.Entity):
    """Base LLM Entity"""
    hass: HomeAssistant
    client: LocalLLMClient
    entry_id: str
    in_context_examples: Optional[List[Dict[str, str]]]

    _attr_has_entity_name = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, subentry: ConfigSubentry, client: LocalLLMClient) -> None:
        """Initialize the agent."""
        self._attr_name = subentry.title
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            model=subentry.data.get(CONF_CHAT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

        self.hass = hass
        self.entry_id = entry.entry_id
        self.subentry_id = subentry.subentry_id
        self.client = client

        # create update handler
        self.async_on_remove(entry.add_update_listener(self._async_update_options))

    async def _async_update_options(self, hass: HomeAssistant, config_entry: LocalLLMConfigEntry):
        for subentry in config_entry.subentries.values():
            # handle subentry updates, but only invoke for this entity
            if subentry.subentry_id == self.subentry_id:
                await hass.async_add_executor_job(self.client._update_options, self.runtime_options)

    @property
    def entry(self) -> ConfigEntry:
        try:
            return self.hass.data[DOMAIN][self.entry_id]
        except KeyError as ex:
            raise Exception("Attempted to use self.entry during startup.") from ex

    @property
    def subentry(self) -> ConfigSubentry:
        try:
            return self.entry.subentries[self.subentry_id]
        except KeyError as ex:
            raise Exception("Attempted to use self.subentry during startup.") from ex
        
    @property
    def runtime_options(self) -> dict[str, Any]:
        """Return the runtime options for this entity."""
        return {**self.entry.options, **self.subentry.data}

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return self.entry.options.get(CONF_SELECTED_LANGUAGE, MATCH_ALL)