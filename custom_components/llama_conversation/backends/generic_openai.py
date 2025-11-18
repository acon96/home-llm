"""Defines the OpenAI API compatible agents"""
from __future__ import annotations
import json

import aiohttp
import asyncio
import datetime
import logging
from typing import List, Dict, Tuple, AsyncGenerator, Any, Optional

from homeassistant.core import HomeAssistant
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
    RECOMMENDED_CHAT_MODELS,
)
from custom_components.llama_conversation.entity import TextGenerationResult, LocalLLMClient

_LOGGER = logging.getLogger(__name__)

class GenericOpenAIAPIClient(LocalLLMClient):
    """Implements the OpenAPI-compatible text completion and chat completion API backends."""

    api_host: str
    api_key: str

    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]) -> None:
        super().__init__(hass, client_options)
        self.api_host = format_url(
            hostname=client_options[CONF_HOST],
            port=client_options[CONF_PORT],
            ssl=client_options[CONF_SSL],
            path="/" + client_options.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)
        )

        self.api_key = client_options.get(CONF_OPENAI_API_KEY, "")

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        host = client_options[CONF_HOST]
        port = client_options[CONF_PORT]
        ssl = client_options[CONF_SSL]
        path = "/" + client_options[CONF_GENERIC_OPENAI_PATH]
        return f"Generic OpenAI at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        headers = {}
        api_key = user_input.get(CONF_OPENAI_API_KEY)
        api_base_path = user_input.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            session = async_get_clientsession(hass)
            async with session.get(
                format_url(
                    hostname=user_input[CONF_HOST],
                    port=user_input[CONF_PORT],
                    ssl=user_input[CONF_SSL],
                    path=f"/{api_base_path}/models"
                ),
                timeout=aiohttp.ClientTimeout(total=5), # quick timeout
                headers=headers
            ) as response:
                if response.ok:
                    return None
                else:
                    return f"HTTP Status {response.status}"
        except Exception as ex:
            return str(ex)

    async def async_get_available_models(self) -> List[str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            session = async_get_clientsession(self.hass)
            async with session.get(
                f"{self.api_host}/models",
                timeout=aiohttp.ClientTimeout(total=5), # quick timeout
                headers=headers
            ) as response:
                response.raise_for_status()
                models_result = await response.json()
        except:
            _LOGGER.exception("Failed to get available models")
            return RECOMMENDED_CHAT_MODELS
            
        return [ model["id"] for model in models_result["data"] ]

    def _generate_stream(self, 
                         conversation: List[conversation.Content],
                         llm_api: llm.APIInstance | None,
                         agent_id: str,
                         entity_options: dict[str, Any]) -> AsyncGenerator[TextGenerationResult, None]:
        model_name = entity_options[CONF_CHAT_MODEL]
        temperature = entity_options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = entity_options.get(CONF_TOP_P, DEFAULT_TOP_P)
        timeout = entity_options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        enable_legacy_tool_calling = entity_options.get(CONF_ENABLE_LEGACY_TOOL_CALLING, DEFAULT_ENABLE_LEGACY_TOOL_CALLING)

        endpoint, additional_params = self._chat_completion_params(entity_options)
        messages = get_oai_formatted_messages(conversation, user_content_as_list=True)

        request_params = {
            "model": model_name,
            "stream": True,
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

        async def anext_token() -> AsyncGenerator[Tuple[Optional[str], Optional[List[llm.ToolInput]]], None]:
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
                            to_say, tool_calls = self._extract_response(json.loads(chunk), llm_api, agent_id)
                            if to_say or tool_calls:
                                yield to_say, tool_calls
            except asyncio.TimeoutError as err:
                raise HomeAssistantError("The generation request timed out! Please check your connection settings, increase the timeout in settings, or decrease the number of exposed entities.") from err
            except aiohttp.ClientError as err:
                raise HomeAssistantError(f"Failed to communicate with the API! {err}") from err

        return self._async_parse_completion(llm_api, agent_id, entity_options, anext_token=anext_token())
    
    def _chat_completion_params(self, entity_options: dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        request_params = {}
        endpoint = "/chat/completions"
        return endpoint, request_params

    def _extract_response(self, response_json: dict, llm_api: llm.APIInstance | None, agent_id: str) -> Tuple[Optional[str], Optional[List[llm.ToolInput]]]:
        if "choices" not in response_json or len(response_json["choices"]) == 0: # finished
            _LOGGER.warning("Response missing or empty 'choices'. Keys present: %s. Full response: %s",
                            list(response_json.keys()), response_json)
            return None, None
        
        choice = response_json["choices"][0]
        tool_calls = None
        if response_json["object"] == "chat.completion":
            response_text = choice["message"]["content"]
            streamed = False
        elif response_json["object"] == "chat.completion.chunk":
            response_text = choice["delta"].get("content", "")
            if "tool_calls" in choice["delta"] and choice["delta"]["tool_calls"] is not None:
                tool_calls = []
                for call in choice["delta"]["tool_calls"]:
                    tool_call, to_say = parse_raw_tool_call(
                        call["function"], llm_api, agent_id)
                    
                    if tool_call:
                        tool_calls.append(tool_call)
                        
                    if to_say:
                        response_text += to_say
            streamed = True
        else:
            response_text = choice["text"]
            streamed = False

        if not streamed or (streamed and choice.get("finish_reason")):
            finish_reason = choice.get("finish_reason")
            if finish_reason in ("length", "content_filter"):
                _LOGGER.warning("Model response did not end on a stop token (unfinished sentence)")

        return response_text, tool_calls


class GenericOpenAIResponsesAPIClient(LocalLLMClient):
    """Implements the OpenAPI-compatible Responses API backend."""

    api_host: str
    api_key: str

    _attr_supports_streaming = False

    _last_response_id: str | None = None
    _last_response_id_time: datetime.datetime | None = None

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]) -> None:
        super().__init__(hass, client_options)
        self.api_host = format_url(
            hostname=client_options[CONF_HOST],
            port=client_options[CONF_PORT],
            ssl=client_options[CONF_SSL],
            path="/" + client_options.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)
        )

        self.api_key = client_options.get(CONF_OPENAI_API_KEY, "")

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        host = client_options[CONF_HOST]
        port = client_options[CONF_PORT]
        ssl = client_options[CONF_SSL]
        path = "/" + client_options[CONF_GENERIC_OPENAI_PATH]
        return f"Generic OpenAI at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"

    def _responses_params(self, conversation: List[conversation.Content], entity_options: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        request_params = {}

        endpoint = "/responses"
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
        if self._last_response_id and self._last_response_id_time and entity_options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION):
            # If the last response was generated recently, use it as a context
            configured_memory_time: datetime.timedelta = datetime.timedelta(minutes=entity_options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES))
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

    def _extract_response(self, response_json: dict) -> str | None:
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
                        agent_id: str,
                        entity_options: dict[str, Any]) -> TextGenerationResult:
        """Generate a response using the OpenAI-compatible Responses API (non-streaming endpoint wrapped as a single-chunk stream)."""

        model_name = entity_options.get(CONF_CHAT_MODEL)
        timeout = entity_options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        endpoint, additional_params = self._responses_params(conversation, entity_options)

        request_params: Dict[str, Any] = {
            "model": model_name,
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
                    text = self._extract_response(response_json)
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
