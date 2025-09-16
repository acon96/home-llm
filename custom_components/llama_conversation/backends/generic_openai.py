"""Defines the OpenAI API compatible agents"""
from __future__ import annotations

import aiohttp
import asyncio
import datetime
import logging
from typing import List, Dict, Tuple, Optional, Any

from homeassistant.components import conversation as conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from custom_components.llama_conversation.utils import format_url
from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_REQUEST_TIMEOUT,
    CONF_OPENAI_API_KEY,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_REMOTE_USE_CHAT_ENDPOINT,
    CONF_GENERIC_OPENAI_PATH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES,
    DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
    DEFAULT_GENERIC_OPENAI_PATH,
)
from custom_components.llama_conversation.conversation import LocalLLMAgent, TextGenerationResult

_LOGGER = logging.getLogger(__name__)

class BaseOpenAICompatibleAPIAgent(LocalLLMAgent):
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

    async def _async_generate_with_parameters(self, endpoint: str, additional_params: dict) -> TextGenerationResult:
        """Generate a response using the OpenAI-compatible API"""

        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        timeout = self.entry.options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)

        request_params = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

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
            return TextGenerationResult(raise_error=True, error_msg="The generation request timed out! Please check your connection settings, increase the timeout in settings, or decrease the number of exposed entities.")
        except aiohttp.ClientError as err:
            _LOGGER.debug(f"Err was: {err}")
            _LOGGER.debug(f"Request was: {request_params}")
            _LOGGER.debug(f"Result was: {response}")
            return TextGenerationResult(raise_error=True, error_msg=f"Failed to communicate with the API! {err}")

        _LOGGER.debug(result)

        return self._extract_response(result)

    def _extract_response(self, response_json: dict) -> TextGenerationResult:
        raise NotImplementedError("Subclasses must implement _extract_response()")

class GenericOpenAIAPIAgent(BaseOpenAICompatibleAPIAgent):
    """Implements the OpenAPI-compatible text completion and chat completion API backends."""

    def _chat_completion_params(self, conversation: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        request_params = {}
        api_base_path = self.entry.data.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)

        endpoint = f"/{api_base_path}/chat/completions"
        request_params["messages"] = [ { "role": x["role"], "content": x["message"] } for x in conversation ]

        return endpoint, request_params

    def _completion_params(self, conversation: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        request_params = {}
        api_base_path = self.entry.data.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)

        endpoint = f"/{api_base_path}/completions"
        request_params["prompt"] = self._format_prompt(conversation)

        return endpoint, request_params

    def _extract_response(self, response_json: dict) -> TextGenerationResult:
        choice = response_json["choices"][0]
        if response_json["object"] == "chat.completion":
            response_text = choice["message"]["content"]
            streamed = False
        elif response_json["object"] == "chat.completion.chunk":
            response_text = choice["message"]["content"]
            streamed = True
        else:
            response_text = choice["text"]
            streamed = False

        if not streamed or streamed and choice["finish_reason"]:
            if choice["finish_reason"] == "length" or choice["finish_reason"] == "content_filter":
                _LOGGER.warning("Model response did not end on a stop token (unfinished sentence)")

        return TextGenerationResult(
            response=response_text,
            stop_reason=choice["finish_reason"],
            response_streamed=streamed
        )

    async def _async_generate(self, conversation: List[Dict[str, str]]) -> TextGenerationResult:
        use_chat_api = self.entry.options.get(CONF_REMOTE_USE_CHAT_ENDPOINT, DEFAULT_REMOTE_USE_CHAT_ENDPOINT)

        if use_chat_api:
            endpoint, additional_params = self._chat_completion_params(conversation)
        else:
            endpoint, additional_params = self._completion_params(conversation)

        result = await self._async_generate_with_parameters(endpoint, additional_params)

        return result

class GenericOpenAIResponsesAPIAgent(BaseOpenAICompatibleAPIAgent):
    """Implements the OpenAPI-compatible Responses API backend."""

    _last_response_id: str | None = None
    _last_response_id_time: datetime.datetime = None

    def _responses_params(self, conversation: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        request_params = {}
        api_base_path = self.entry.data.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)

        endpoint = f"/{api_base_path}/responses"
        request_params["input"] = conversation[-1]["message"] # last message in the conversation is the user input

        # Assign previous_response_id if relevant
        if self._last_response_id and self.entry.options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION):
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

    def _extract_response(self, response_json: dict) -> TextGenerationResult:
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

    async def _async_generate(self, conversation: List[Dict[str, str]]) -> TextGenerationResult:
        """Generate a response using the OpenAI-compatible Responses API"""

        endpoint, additional_params = self._responses_params(conversation)
        return await self._async_generate_with_parameters(endpoint, additional_params)