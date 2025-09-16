"""Defines the OpenAI API compatible agents"""
from __future__ import annotations
import json

import aiohttp
import asyncio
import datetime
import logging
from typing import List, Dict, Tuple, AsyncGenerator, Any, Optional

from homeassistant.exceptions import HomeAssistantError
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers import llm

from custom_components.llama_conversation.utils import format_url, get_oai_formatted_messages, get_oai_formatted_tools, parse_raw_tool_call
from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_REQUEST_TIMEOUT,
    CONF_OPENAI_API_KEY,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_GENERIC_OPENAI_PATH,
    CONF_ENABLE_LEGACY_TOOL_CALLING,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES,
    DEFAULT_GENERIC_OPENAI_PATH,
    DEFAULT_ENABLE_LEGACY_TOOL_CALLING,
)
from custom_components.llama_conversation.entity import TextGenerationResult, LocalLLMClient

_LOGGER = logging.getLogger(__name__)

class GenericOpenAIAPIClient(LocalLLMClient):
    """Implements the OpenAPI-compatible text completion and chat completion API backends."""

    api_host: str
    api_key: str
    model_name: str

    _attr_supports_streaming = True

    async def _async_load_model(self, entry: ConfigEntry) -> None:
        self.api_host = format_url(
            hostname=entry.data[CONF_HOST],
            port=entry.data[CONF_PORT],
            ssl=entry.data[CONF_SSL],
            path=""
        )

        self.api_key = entry.data.get(CONF_OPENAI_API_KEY, "")
        self.model_name = entry.data.get(CONF_CHAT_MODEL, "")

    def _generate_stream(self, 
                         conversation: List[conversation.Content],
                         llm_api: llm.APIInstance | None,
                         user_input: conversation.ConversationInput,
                         entity_options: dict[str, Any]) -> AsyncGenerator[TextGenerationResult, None]:
        max_tokens = entity_options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = entity_options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = entity_options.get(CONF_TOP_P, DEFAULT_TOP_P)
        timeout = entity_options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        enable_legacy_tool_calling = entity_options.get(CONF_ENABLE_LEGACY_TOOL_CALLING, DEFAULT_ENABLE_LEGACY_TOOL_CALLING)

        endpoint, additional_params = self._chat_completion_params(entity_options)
        messages = get_oai_formatted_messages(conversation)

        request_params = {
            "model": self.model_name,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": messages
        }

        tools = None
        # "legacy" tool calling passes the tools directly as part of the system prompt instead of as "tools"
        # most local backends absolutely butcher any sort of prompt formatting when using tool calling
        if llm_api and not enable_legacy_tool_calling:
            tools = get_oai_formatted_tools(llm_api, self._async_get_all_exposed_domains())
            request_params["tools"] = tools

        request_params.update(additional_params)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        _LOGGER.debug(f"Generating completion with {len(messages)} messages and {len(tools) if tools else 0} tools...")

        session = async_get_clientsession(self.hass)

        async def anext_token() -> AsyncGenerator[Tuple[Optional[str], Optional[List]], None]:
            response = None
            chunk = None
            try:
                async with session.post(
                    f"{self.api_host}{endpoint}",
                    json=request_params,
                    timeout=timeout,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    async for line_bytes in response.content:
                        raw_line = line_bytes.decode("utf-8").strip()
                        if raw_line.startswith("error: "):
                            raise Exception(f"Error from server: {raw_line}")
                        chunk = raw_line.removeprefix("data: ")
                        if "[DONE]" in chunk:
                            break

                        if chunk and chunk.strip():
                            yield self._extract_response(json.loads(chunk), llm_api)
            except asyncio.TimeoutError as err:
                raise HomeAssistantError("The generation request timed out! Please check your connection settings, increase the timeout in settings, or decrease the number of exposed entities.") from err
            except aiohttp.ClientError as err:
                raise HomeAssistantError(f"Failed to communicate with the API! {err}") from err

        return self._async_parse_completion(llm_api, entity_options, anext_token=anext_token())
    
    def _chat_completion_params(self, entity_options: dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        request_params = {}
        api_base_path = entity_options.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)
        endpoint = f"/{api_base_path}/chat/completions"
        return endpoint, request_params

    def _extract_response(self, response_json: dict, llm_api: llm.APIInstance | None) -> Tuple[Optional[str], Optional[List]]:
        if len(response_json["choices"]) == 0: # finished
            return None, None
        
        choice = response_json["choices"][0]
        tool_calls = None
        if response_json["object"] == "chat.completion":
            response_text = choice["message"]["content"]
            streamed = False
        elif response_json["object"] == "chat.completion.chunk":
            response_text = choice["delta"].get("content", "")
            if "tool_calls" in choice["delta"]:
                tool_calls = []
                for call in choice["delta"]["tool_calls"]:
                    tool_args, to_say = parse_raw_tool_call(
                        call["function"], llm_api)
                    
                    if tool_args:
                        tool_calls.append(tool_args)
                        
                    if to_say:
                        response_text += to_say
            streamed = True
        else:
            response_text = choice["text"]
            streamed = False

        if not streamed or streamed and choice["finish_reason"]:
            if choice["finish_reason"] == "length" or choice["finish_reason"] == "content_filter":
                _LOGGER.warning("Model response did not end on a stop token (unfinished sentence)")

        return response_text, tool_calls


class GenericOpenAIResponsesAPIClient(LocalLLMClient):
    """Implements the OpenAPI-compatible Responses API backend."""

    api_host: str
    api_key: str
    model_name: str

    _attr_supports_streaming = False

    _last_response_id: str | None = None
    _last_response_id_time: datetime.datetime | None = None

    async def _async_load_model(self, entry: ConfigEntry) -> None:
        self.api_host = format_url(
            hostname=entry.data[CONF_HOST],
            port=entry.data[CONF_PORT],
            ssl=entry.data[CONF_SSL],
            path=""
        )

        self.api_key = entry.data.get(CONF_OPENAI_API_KEY, "")
        self.model_name = entry.data.get(CONF_CHAT_MODEL, "")

    def _responses_params(self, conversation: List[conversation.Content], api_base_path: str) -> Tuple[str, Dict[str, Any]]:
        request_params = {}
        api_base_path = self.entry.data.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)

        endpoint = f"/{api_base_path}/responses"
        # Find the last user message in the conversation and use its content as the input
        input_text: str | None = None
        for msg in reversed(conversation):
            try:
                if msg.role == "user":
                    input_text = msg.content
                    break
            except Exception:
                continue

        if input_text is None:
            # fallback to the last message content
            input_text = getattr(conversation[-1], "content", "")

        request_params["input"] = input_text

        # Assign previous_response_id if relevant
        if self._last_response_id and self._last_response_id_time and self.entry.options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION):
            # If the last response was generated recently, use it as a context
            configured_memory_time: datetime.timedelta = datetime.timedelta(minutes=self.entry.options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES))
            last_conversation_age: datetime.timedelta = datetime.datetime.now() - self._last_response_id_time
            _LOGGER.debug(f"Conversation ID age: {last_conversation_age}")
            if last_conversation_age < configured_memory_time:
                _LOGGER.debug(f"Using previous response ID {self._last_response_id} for context")
                request_params["previous_response_id"] = self._last_response_id
            else:
                _LOGGER.debug(f"Previous response ID {self._last_response_id} is too old, not using it for context")

        return endpoint, request_params

    def _validate_response_payload(self, response_json: dict) -> bool:
        """
        Validate that the payload given matches the expected structure for the Responses API.

        API ref: https://platform.openai.com/docs/api-reference/responses/object

        Returns True or raises an error
        """
        required_response_keys = ["object", "output", "status", "id"]
        missing_keys = [key for key in required_response_keys if key not in response_json]
        if missing_keys:
            raise ValueError(f"Response JSON is missing required keys: {', '.join(missing_keys)}")

        if response_json["object"] != "response":
            raise ValueError(f"Response JSON object is not 'response', got {response_json['object']}")

        if "error" in response_json and response_json["error"] is not None:
            error = response_json["error"]
            _LOGGER.error(f"Response received error payload.")
            if "message" not in error:
                raise ValueError("Response JSON error is missing 'message' key")
            raise ValueError(f"Response JSON error: {error['message']}")

        return True

    def _check_response_status(self, response_json: dict) -> None:
        """
        Check the status of the response and logs a message if it is not 'completed'.

        API ref: https://platform.openai.com/docs/api-reference/responses/object#responses_object-status
        """
        if response_json["status"] != "completed":
            _LOGGER.warning(f"Response status is not 'completed', got {response_json['status']}. Details: {response_json.get('incomplete_details', 'No details provided')}")

    def _extract_response(self, response_json: dict, llm_api: llm.APIInstance | None, user_input: conversation.ConversationInput) -> str | None:
        self._validate_response_payload(response_json)
        self._check_response_status(response_json)

        outputs = response_json["output"]

        if len(outputs) > 1:
            _LOGGER.warning("Received multiple outputs from the Responses API, returning the first one.")

        output = outputs[0]

        if not output["type"] == "message":
            raise NotImplementedError(f"Response output type is not 'message', got {output['type']}")

        if len(output["content"]) > 1:
            _LOGGER.warning("Received multiple content items in the response output, returning the first one.")

        content = output["content"][0]

        output_type = content["type"]

        to_return: str | None = None

        if output_type == "refusal":
            _LOGGER.info("Received a refusal from the Responses API.")
            to_return = content["refusal"]
        elif output_type == "output_text":
            to_return = content["text"]
        else:
            raise ValueError(f"Response output content type is not expected, got {output_type}")

        # Save the response_id and return the successful response.
        response_id = response_json["id"]
        self._last_response_id = response_id
        self._last_response_id_time = datetime.datetime.now()

        return to_return

    async def _generate(self, 
                        conversation: List[conversation.Content],
                        llm_api: llm.APIInstance | None,
                        user_input: conversation.ConversationInput,
                        entity_options: dict[str, Any]) -> TextGenerationResult:
        """Generate a response using the OpenAI-compatible Responses API (non-streaming endpoint wrapped as a single-chunk stream)."""

        timeout = entity_options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        endpoint, additional_params = self._responses_params(conversation, api_base_path=entity_options.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH))

        request_params: Dict[str, Any] = {
            "model": self.model_name,
        }
        request_params.update(additional_params)

        headers: Dict[str, Any] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        session = async_get_clientsession(self.hass)
        response = None

        try:
            async with session.post(
                f"{self.api_host}{endpoint}",
                json=request_params,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers=headers,
            ) as response:
                response.raise_for_status()
                response_json = await response.json()

                try:
                    text = self._extract_response(response_json, llm_api, user_input)
                    return TextGenerationResult(response=text, response_streamed=False)
                except Exception as err:
                    _LOGGER.exception("Failed to parse Responses API payload: %s", err)
                    return TextGenerationResult(raise_error=True, error_msg=f"Failed to parse Responses API payload: {err}")
        except asyncio.TimeoutError:
            return TextGenerationResult(raise_error=True, error_msg="The generation request timed out! Please check your connection settings, increase the timeout in settings, or decrease the number of exposed entities.")
        except aiohttp.ClientError as err:
            _LOGGER.debug(f"Err was: {err}")
            _LOGGER.debug(f"Request was: {request_params}")
            _LOGGER.debug(f"Result was: {response}")
            return TextGenerationResult(raise_error=True, error_msg=f"Failed to communicate with the API! {err}")