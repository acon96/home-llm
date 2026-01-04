"""Defines the Ollama compatible agent backed by the official python client."""
from __future__ import annotations

import logging
import ssl
from collections.abc import Mapping
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import certifi
import httpx
from ollama import AsyncClient, ChatResponse, ResponseError

from homeassistant.components import conversation as conversation
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from custom_components.llama_conversation.utils import format_url, get_oai_formatted_messages, get_oai_formatted_tools
from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_TYPICAL_P,
    CONF_MIN_P,
    CONF_ENABLE_THINK_MODE,
    CONF_REQUEST_TIMEOUT,
    CONF_API_KEY,
    CONF_API_PATH,
    CONF_OLLAMA_KEEP_ALIVE_MIN,
    CONF_OLLAMA_JSON_MODE,
    CONF_CONTEXT_LENGTH,
    CONF_ENABLE_LEGACY_TOOL_CALLING,
    CONF_TOOL_RESPONSE_AS_STRING,
    CONF_RESPONSE_JSON_SCHEMA,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_TYPICAL_P,
    DEFAULT_MIN_P,
    DEFAULT_ENABLE_THINK_MODE,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_API_PATH,
    DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
    DEFAULT_OLLAMA_JSON_MODE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_ENABLE_LEGACY_TOOL_CALLING,
    DEFAULT_TOOL_RESPONSE_AS_STRING,
)

from custom_components.llama_conversation.entity import LocalLLMClient, TextGenerationResult

_LOGGER = logging.getLogger(__name__)


def _normalize_path(path: str | None) -> str:
    if not path:
        return ""
    trimmed = str(path).strip("/")
    return f"/{trimmed}" if trimmed else ""


def _build_default_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    try:
        context.load_verify_locations(certifi.where())
    except OSError as err:
        _LOGGER.debug("Failed to load certifi bundle for Ollama client: %s", err)
    return context

class OllamaAPIClient(LocalLLMClient):
    api_host: str
    api_key: Optional[str]

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]) -> None:
        super().__init__(hass, client_options)
        base_path = _normalize_path(client_options.get(CONF_API_PATH, DEFAULT_API_PATH))
        self.api_host = format_url(
            hostname=client_options[CONF_HOST],
            port=client_options[CONF_PORT],
            ssl=client_options[CONF_SSL],
            path=base_path,
        )
        self.api_key = client_options.get(CONF_API_KEY) or None
        self._headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        self._ssl_context = _build_default_ssl_context() if client_options.get(CONF_SSL) else None

    def _build_client(self, *, timeout: float | int | httpx.Timeout | None = None) -> AsyncClient:
        timeout_config: httpx.Timeout | float | None = timeout
        if isinstance(timeout, (int, float)):
            timeout_config = httpx.Timeout(timeout)

        return AsyncClient(
            host=self.api_host,
            headers=self._headers,
            timeout=timeout_config,
            verify=self._ssl_context,
        )

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        host = client_options[CONF_HOST]
        port = client_options[CONF_PORT]
        ssl = client_options[CONF_SSL]
        path = _normalize_path(client_options.get(CONF_API_PATH, DEFAULT_API_PATH))
        return f"Ollama at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        api_key = user_input.get(CONF_API_KEY)
        base_path = _normalize_path(user_input.get(CONF_API_PATH, DEFAULT_API_PATH))
        timeout_config: httpx.Timeout | float | None = httpx.Timeout(5)

        verify_context = None
        if user_input.get(CONF_SSL):
            verify_context = await hass.async_add_executor_job(_build_default_ssl_context)

        client = AsyncClient(
            host=format_url(
                hostname=user_input[CONF_HOST],
                port=user_input[CONF_PORT],
                ssl=user_input[CONF_SSL],
                path=base_path,
            ),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            timeout=timeout_config,
            verify=verify_context,
        )
        
        try:
            await client.list()
        except httpx.TimeoutException:
            return "Connection timed out"
        except ResponseError as err:
            return f"HTTP Status {err.status_code}: {err.error}"
        except ConnectionError as err:
            return str(err)

        return None

    async def async_get_available_models(self) -> List[str]:
        client = self._build_client(timeout=5)
        try:
            response = await client.list()
        except httpx.TimeoutException as err:
            raise HomeAssistantError("Timed out while fetching models from the Ollama server") from err
        except (ResponseError, ConnectionError) as err:
            raise HomeAssistantError(f"Failed to fetch models from the Ollama server: {err}") from err

        models: List[str] = []
        for model in getattr(response, "models", []) or []:
            candidate = getattr(model, "name", None) or getattr(model, "model", None)
            if candidate:
                models.append(candidate)

        return models

    def _extract_response(self, response_chunk: ChatResponse) -> Tuple[Optional[str], Optional[List[dict]]]:
        content = response_chunk.message.content
        raw_tool_calls = response_chunk.message.tool_calls

        if raw_tool_calls:
            # return openai formatted tool calls
            tool_calls = [{
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
            } for call in raw_tool_calls]
        else:
            tool_calls = None

        return content, tool_calls

    @staticmethod
    def _format_keep_alive(value: Any) -> Any:
        as_text = str(value).strip()
        return 0 if as_text in {"0", "0.0"} else f"{as_text}m"

    def _generate_stream(
        self,
        conversation: List[conversation.Content],
        llm_api: llm.APIInstance | None,
        agent_id: str,
        entity_options: Dict[str, Any],
    ) -> AsyncGenerator[TextGenerationResult, None]:
        model_name = entity_options.get(CONF_CHAT_MODEL, "")
        context_length = entity_options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        max_tokens = entity_options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = entity_options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = entity_options.get(CONF_TOP_P, DEFAULT_TOP_P)
        top_k = entity_options.get(CONF_TOP_K, DEFAULT_TOP_K)
        typical_p = entity_options.get(CONF_TYPICAL_P, DEFAULT_TYPICAL_P)
        timeout = entity_options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        keep_alive = entity_options.get(CONF_OLLAMA_KEEP_ALIVE_MIN, DEFAULT_OLLAMA_KEEP_ALIVE_MIN)
        enable_legacy_tool_calling = entity_options.get(CONF_ENABLE_LEGACY_TOOL_CALLING, DEFAULT_ENABLE_LEGACY_TOOL_CALLING)
        tool_response_as_string = entity_options.get(CONF_TOOL_RESPONSE_AS_STRING, DEFAULT_TOOL_RESPONSE_AS_STRING)
        think_mode = entity_options.get(CONF_ENABLE_THINK_MODE, DEFAULT_ENABLE_THINK_MODE)
        json_mode = entity_options.get(CONF_OLLAMA_JSON_MODE, DEFAULT_OLLAMA_JSON_MODE)

        options = {
            "num_ctx": context_length,
            "top_p": top_p,
            "top_k": top_k,
            "typical_p": typical_p,
            "temperature": temperature,
            "num_predict": max_tokens,
            "min_p": entity_options.get(CONF_MIN_P, DEFAULT_MIN_P),
        }

        messages = get_oai_formatted_messages(conversation, tool_args_to_str=False, tool_result_to_str=tool_response_as_string)
        tools = None
        if llm_api and not enable_legacy_tool_calling:
            tools = get_oai_formatted_tools(llm_api, self._async_get_all_exposed_domains())
        keep_alive_payload = self._format_keep_alive(keep_alive)

        async def anext_token() -> AsyncGenerator[Tuple[Optional[str], Optional[List[dict]]], None]:
            client = self._build_client(timeout=timeout)
            try:
                format_option = entity_options.get(CONF_RESPONSE_JSON_SCHEMA, "json" if json_mode else None)
                stream = await client.chat(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    think=think_mode,
                    format=format_option,
                    options=options,
                    keep_alive=keep_alive_payload,
                )

                async for chunk in stream:
                    yield self._extract_response(chunk)
            except httpx.TimeoutException as err:
                raise HomeAssistantError(
                    "The generation request timed out! Please check your connection settings, increase the timeout in settings, or decrease the number of exposed entities."
                ) from err
            except (ResponseError, ConnectionError) as err:
                raise HomeAssistantError(f"Failed to communicate with the API! {err}") from err

        return self._async_stream_parse_completion(llm_api, agent_id, entity_options, anext_token=anext_token())
