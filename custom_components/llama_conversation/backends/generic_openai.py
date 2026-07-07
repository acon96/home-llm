"""Defines the OpenAI API compatible agents"""
from __future__ import annotations

import asyncio
import datetime
import logging
from typing import List, Dict, Tuple, AsyncGenerator, Any, Optional

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.components import conversation
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.helpers import llm
from openai import AsyncOpenAI, OpenAIError

from custom_components.llama_conversation.utils import format_url, get_oai_formatted_messages, get_oai_formatted_tools
from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_REQUEST_TIMEOUT,
    CONF_API_KEY,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_API_PATH,
    CONF_ENABLE_LEGACY_TOOL_CALLING,
    CONF_TOOL_RESPONSE_AS_STRING,
    CONF_RESPONSE_JSON_SCHEMA,
    CONF_USE_SERVER_SAMPLING_DEFAULTS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES,
    DEFAULT_API_PATH,
    DEFAULT_ENABLE_LEGACY_TOOL_CALLING,
    DEFAULT_TOOL_RESPONSE_AS_STRING,
    DEFAULT_USE_SERVER_SAMPLING_DEFAULTS,
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
            path="/" + client_options.get(CONF_API_PATH, DEFAULT_API_PATH)
        )

        self.api_key = client_options.get(CONF_API_KEY, "")

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        host = client_options[CONF_HOST]
        port = client_options[CONF_PORT]
        ssl = client_options[CONF_SSL]
        path = "/" + client_options[CONF_API_PATH]
        return f"Generic OpenAI at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        api_key = user_input.get(CONF_API_KEY)
        api_base_path = user_input.get(CONF_API_PATH, DEFAULT_API_PATH)
        try:
            async with AsyncOpenAI(
                api_key=api_key,
                base_url=format_url(
                    hostname=user_input[CONF_HOST],
                    port=user_input[CONF_PORT],
                    ssl=user_input[CONF_SSL],
                    path=f"/{api_base_path.lstrip('/')}",
                ),
                timeout=5,
            ) as client:
                await client.models.list()
        except Exception as ex:
            return str(ex)

    async def async_get_available_models(self) -> List[str]:
        try:
            async with AsyncOpenAI(api_key=self.api_key, base_url=self.api_host, timeout=5) as client:
                return [m.id async for m in client.models.list()]
        except (asyncio.TimeoutError, OpenAIError):
            _LOGGER.warning("Falling back to recommended models because the API model list request failed.")
            _LOGGER.exception("Failed to get available models")
            return RECOMMENDED_CHAT_MODELS

    def _generate_stream(self, 
                         conversation: List[conversation.Content],
                         llm_api: llm.APIInstance | None,
                         agent_id: str,
                         entity_options: dict[str, Any],
                        ) -> AsyncGenerator[TextGenerationResult, None]:
        model_name = entity_options[CONF_CHAT_MODEL]
        temperature = entity_options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = entity_options.get(CONF_TOP_P, DEFAULT_TOP_P)
        max_tokens = entity_options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        timeout = entity_options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        enable_legacy_tool_calling = entity_options.get(CONF_ENABLE_LEGACY_TOOL_CALLING, DEFAULT_ENABLE_LEGACY_TOOL_CALLING)
        tool_response_as_string = entity_options.get(CONF_TOOL_RESPONSE_AS_STRING, DEFAULT_TOOL_RESPONSE_AS_STRING)

        _, additional_params = self._chat_completion_params(entity_options)
        messages = get_oai_formatted_messages(conversation, user_content_as_list=True, tool_result_to_str=tool_response_as_string)

        use_server_sampling_defaults = entity_options.get(CONF_USE_SERVER_SAMPLING_DEFAULTS, DEFAULT_USE_SERVER_SAMPLING_DEFAULTS)
        request_params = {
            "model": model_name,
            # "stream": True, # we are using the streaming method, therefore we dont need to specify it
            "max_tokens": max_tokens,
            "messages": messages
        }
        if not use_server_sampling_defaults:
            request_params["temperature"] = temperature
            request_params["top_p"] = top_p

        response_json_schema = entity_options.get(CONF_RESPONSE_JSON_SCHEMA)
        if response_json_schema:
            request_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ha_task",
                    "schema": response_json_schema,
                    "strict": True,
                },
            }

        tools = None
        if llm_api and not enable_legacy_tool_calling:
            tools = get_oai_formatted_tools(llm_api, self._async_get_all_exposed_domains())
            request_params["tools"] = tools

        _LOGGER.debug(f"Generating completion with {len(messages)} messages and {len(tools) if tools else 0} tools...")

        async def anext_token() -> AsyncGenerator[Tuple[Optional[str], Optional[List[dict]]], None]:
            try:                    
                async with AsyncOpenAI(api_key=self.api_key, base_url=self.api_host, timeout=timeout) as client:
                    async with client.chat.completions.stream(**request_params, extra_body=additional_params) as stream:
                        async for event in stream:
                            if event.type == "content.delta": # normal text chunks we can yield
                                yield event.delta, None
                            elif event.type == "tool_calls.function.arguments.done": # function calls need to wait until complete to be yielded
                                yield None, [{"function": {"name": event.name, "arguments": event.parsed_arguments}}]
            except asyncio.TimeoutError as err:
                _LOGGER.debug("OpenAI API timeout during streaming generation: params=%s, error=%s", request_params, err)
                raise HomeAssistantError("The generation request timed out! Please check your connection settings, increase the timeout in settings, or decrease the number of exposed entities.") from err
            except OpenAIError as err:
                _LOGGER.debug("OpenAI API error during streaming generation: params=%s, error=%s", request_params, err)
                raise HomeAssistantError(f"Failed to communicate with the API! {err}") from err

        return self._async_stream_parse_completion(llm_api, agent_id, entity_options, anext_token=anext_token())

    async def _generate(
        self,
        conversation: List[conversation.Content],
        llm_api: llm.APIInstance | None,
        agent_id: str,
        entity_options: dict[str, Any],
    ) -> TextGenerationResult:
        model_name = entity_options[CONF_CHAT_MODEL]
        temperature = entity_options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = entity_options.get(CONF_TOP_P, DEFAULT_TOP_P)
        max_tokens = entity_options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        timeout = entity_options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        enable_legacy_tool_calling = entity_options.get(CONF_ENABLE_LEGACY_TOOL_CALLING, DEFAULT_ENABLE_LEGACY_TOOL_CALLING)
        tool_response_as_string = entity_options.get(CONF_TOOL_RESPONSE_AS_STRING, DEFAULT_TOOL_RESPONSE_AS_STRING)

        endpoint, additional_params = self._chat_completion_params(entity_options)
        messages = get_oai_formatted_messages(conversation, user_content_as_list=True, tool_result_to_str=tool_response_as_string)

        use_server_sampling_defaults = entity_options.get(CONF_USE_SERVER_SAMPLING_DEFAULTS, DEFAULT_USE_SERVER_SAMPLING_DEFAULTS)
        request_params: Dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if not use_server_sampling_defaults:
            request_params["temperature"] = temperature
            request_params["top_p"] = top_p

        response_json_schema = entity_options.get(CONF_RESPONSE_JSON_SCHEMA)
        if response_json_schema:
            request_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ha_task",
                    "schema": response_json_schema,
                    "strict": True,
                },
            }

        if llm_api and not enable_legacy_tool_calling:
            request_params["tools"] = get_oai_formatted_tools(llm_api, self._async_get_all_exposed_domains())

        _LOGGER.debug(f"Generating non-stream completion with {len(messages)} messages...")

        try:
            async with AsyncOpenAI(api_key=self.api_key, base_url=self.api_host, timeout=timeout) as client:
                completion = await client.chat.completions.create(**request_params, extra_body=additional_params)
        except asyncio.TimeoutError as err:
            _LOGGER.debug("OpenAI API timeout during generation: params=%s, error=%s", request_params, err)
            raise HomeAssistantError("The generation request timed out! Please check your connection settings, increase the timeout in settings, or decrease the number of exposed entities.") from err
        except OpenAIError as err:
            _LOGGER.debug("OpenAI API error during generation: params=%s, error=%s", request_params, err)
            raise HomeAssistantError(f"Failed to communicate with the API! {err}") from err

        first_choice = completion.choices[0] if completion.choices else None
        message = first_choice.message if first_choice else None

        content = ""
        if message and message.content:
            if isinstance(message.content, str):
                content = message.content
            else:
                # Some OpenAI-compatible APIs return structured content parts.
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in message.content
                )

        raw_tool_calls: list[dict] = []
        if message and message.tool_calls:
            for tool_call in message.tool_calls:
                function = getattr(tool_call, "function", None)
                if not function:
                    continue
                raw_tool_calls.append(
                    {
                        "function": {
                            "name": getattr(function, "name", ""),
                            "arguments": getattr(function, "arguments", "{}"),
                        }
                    }
                )

        async def single_chunk() -> AsyncGenerator[Tuple[Optional[str], Optional[List[dict]]], None]:
            yield content, (raw_tool_calls or None)

        return await self._collect_result_stream(
            self._async_stream_parse_completion(
                llm_api,
                agent_id,
                entity_options,
                anext_token=single_chunk(),
            )
        )
    
    def _chat_completion_params(self, entity_options: dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        request_params = {}
        endpoint = "/chat/completions"
        return endpoint, request_params


class GenericOpenAIResponsesAPIClient(LocalLLMClient):
    """Implements the OpenAPI-compatible Responses API backend."""

    api_host: str
    api_key: str

    _attr_supports_streaming = True

    _last_response_id: str | None = None
    _last_response_id_time: datetime.datetime | None = None

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]) -> None:
        super().__init__(hass, client_options)
        self.api_host = format_url(
            hostname=client_options[CONF_HOST],
            port=client_options[CONF_PORT],
            ssl=client_options[CONF_SSL],
            path="/" + client_options.get(CONF_API_PATH, DEFAULT_API_PATH)
        )
        self.api_key = client_options.get(CONF_API_KEY, "")

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        host = client_options[CONF_HOST]
        port = client_options[CONF_PORT]
        ssl = client_options[CONF_SSL]
        path = "/" + client_options[CONF_API_PATH]
        return f"Generic OpenAI at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"

    def _build_responses_request_params(self, conversation: List[conversation.Content], entity_options: Dict[str, Any]) -> Dict[str, Any]:
        """Build the base request params dict for the Responses API call."""
        params: Dict[str, Any] = {}

        input_text: str | None = None
        for msg in reversed(conversation):
            try:
                if msg.role == "user":
                    input_text = msg.content
                    break
            except Exception:
                continue

        if input_text is None:
            input_text = getattr(conversation[-1], "content", "")

        params["input"] = input_text

        if (
            self._last_response_id
            and self._last_response_id_time
            and entity_options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION)
        ):
            configured_memory_time: datetime.timedelta = datetime.timedelta(
                minutes=entity_options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES)
            )
            last_conversation_age: datetime.timedelta = datetime.datetime.now() - self._last_response_id_time
            _LOGGER.debug(f"Conversation ID age: {last_conversation_age}")
            if last_conversation_age < configured_memory_time:
                _LOGGER.debug(f"Using previous response ID {self._last_response_id} for context")
                params["previous_response_id"] = self._last_response_id
            else:
                _LOGGER.debug(f"Previous response ID {self._last_response_id} is too old, not using it for context")

        return params

    def _check_response_status(self, response) -> None:
        """Check the response status and log a warning if not 'completed'."""
        if response.status != "completed":
            _LOGGER.warning(
                f"Response status is not 'completed', got {response.status}. "
                f"Details: {getattr(response, 'incomplete_details', 'No details provided')}"
            )

    def _extract_text_from_response(self, response) -> str | None:
        """Extract text content from an SDK Response object."""
        self._check_response_status(response)

        outputs = response.output
        if not outputs:
            raise ValueError("Response contains no output items.")

        if len(outputs) > 1:
            _LOGGER.warning("Received multiple outputs from the Responses API, returning the first one.")

        output = outputs[0]
        if output.type != "message":
            raise NotImplementedError(f"Response output type is not 'message', got {output.type}")

        content_items = output.content
        if len(content_items) > 1:
            _LOGGER.warning("Received multiple content items in the response output, returning the first one.")

        content = content_items[0]
        output_type = content.type

        if output_type == "refusal":
            _LOGGER.info("Received a refusal from the Responses API.")
            return content.refusal
        elif output_type == "output_text":
            return content.text
        else:
            raise ValueError(f"Response output content type is not expected, got {output_type}")

    def _generate_stream(
        self,
        conversation: List[conversation.Content],
        llm_api: llm.APIInstance | None,
        agent_id: str,
        entity_options: dict[str, Any],
    ) -> AsyncGenerator[TextGenerationResult, None]:
        model_name = entity_options.get(CONF_CHAT_MODEL)
        timeout = entity_options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)

        request_params: Dict[str, Any] = {
            "model": model_name,
            **self._build_responses_request_params(conversation, entity_options),
        }

        response_json_schema = entity_options.get(CONF_RESPONSE_JSON_SCHEMA)
        if response_json_schema:
            request_params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "ha_task",
                    "schema": response_json_schema,
                    "strict": True,
                }
            }

        async def anext_token() -> AsyncGenerator[Tuple[Optional[str], Optional[List[dict]]], None]:
            try:
                async with AsyncOpenAI(api_key=self.api_key, base_url=self.api_host, timeout=timeout) as client:
                    async with client.responses.stream(**request_params) as stream:
                        async for event in stream:
                            if event.type == "response.output_text.delta":
                                yield event.delta, None
                            elif event.type == "response.function_call_arguments.done":
                                yield None, [{"function": {"name": event.name, "arguments": event.arguments}}]
                        final = await stream.get_final_response()
                        self._last_response_id = final.id
                        self._last_response_id_time = datetime.datetime.now()
            except asyncio.TimeoutError as err:
                _LOGGER.debug("OpenAI Responses API timeout during streaming generation: params=%s, error=%s", request_params, err)
                raise HomeAssistantError(
                    "The generation request timed out! Please check your connection settings, "
                    "increase the timeout in settings, or decrease the number of exposed entities."
                ) from err
            except OpenAIError as err:
                _LOGGER.debug("OpenAI Responses API error during streaming generation: params=%s, error=%s", request_params, err)
                raise HomeAssistantError(f"Failed to communicate with the API! {err}") from err

        return self._async_stream_parse_completion(llm_api, agent_id, entity_options, anext_token=anext_token())

    async def _generate(
        self,
        conversation: List[conversation.Content],
        llm_api: llm.APIInstance | None,
        agent_id: str,
        entity_options: dict[str, Any],
    ) -> TextGenerationResult:
        """Generate a response using the OpenAI SDK Responses API."""
        model_name = entity_options.get(CONF_CHAT_MODEL)
        timeout = entity_options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)

        request_params: Dict[str, Any] = {
            "model": model_name,
            **self._build_responses_request_params(conversation, entity_options),
        }

        response_json_schema = entity_options.get(CONF_RESPONSE_JSON_SCHEMA)
        if response_json_schema:
            request_params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "ha_task",
                    "schema": response_json_schema,
                    "strict": True,
                }
            }

        try:
            async with AsyncOpenAI(api_key=self.api_key, base_url=self.api_host, timeout=timeout) as client:
                response = await client.responses.create(**request_params)
        except asyncio.TimeoutError as err:
            _LOGGER.debug("OpenAI Responses API timeout during generation: params=%s, error=%s", request_params, err)
            return TextGenerationResult(
                raise_error=True,
                error_msg="The generation request timed out! Please check your connection settings, "
                          "increase the timeout in settings, or decrease the number of exposed entities."
            )
        except OpenAIError as err:
            _LOGGER.debug("OpenAI Responses API error during generation: params=%s, error=%s", request_params, err)
            return TextGenerationResult(raise_error=True, error_msg=f"Failed to communicate with the API! {err}")

        try:
            text = self._extract_text_from_response(response)
            if not text:
                return TextGenerationResult(raise_error=True, error_msg="The Responses API returned an empty response.")
            self._last_response_id = response.id
            self._last_response_id_time = datetime.datetime.now()
            return TextGenerationResult(response=text)
        except Exception as err:
            _LOGGER.exception("Failed to parse Responses API payload: %s", err)
            return TextGenerationResult(raise_error=True, error_msg=f"Failed to parse Responses API payload: {err}")
