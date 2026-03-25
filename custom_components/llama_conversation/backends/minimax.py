"""Defines the MiniMax API backend."""
from __future__ import annotations

import aiohttp
import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_REQUEST_TIMEOUT,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_ENABLE_LEGACY_TOOL_CALLING,
    CONF_TOOL_RESPONSE_AS_STRING,
    CONF_RESPONSE_JSON_SCHEMA,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_ENABLE_LEGACY_TOOL_CALLING,
    DEFAULT_TOOL_RESPONSE_AS_STRING,
)
from custom_components.llama_conversation.entity import LocalLLMClient, TextGenerationResult
from custom_components.llama_conversation.utils import get_oai_formatted_messages, get_oai_formatted_tools

_LOGGER = logging.getLogger(__name__)

MINIMAX_DEFAULT_BASE_URL = "https://api.minimax.io/v1"

MINIMAX_AVAILABLE_MODELS = [
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
]


class MiniMaxAPIClient(LocalLLMClient):
    """Implements the MiniMax API backend via OpenAI-compatible endpoint."""

    api_host: str
    api_key: str

    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]) -> None:
        super().__init__(hass, client_options)

        base_url = client_options.get(CONF_BASE_URL, MINIMAX_DEFAULT_BASE_URL)
        self.api_host = base_url.rstrip("/")
        self.api_key = client_options.get(CONF_API_KEY, "")

    @staticmethod
    def get_name(client_options: dict[str, Any]) -> str:
        base_url = client_options.get(CONF_BASE_URL, MINIMAX_DEFAULT_BASE_URL)
        return f"MiniMax API at '{base_url}'"

    @staticmethod
    async def async_validate_connection(
        hass: HomeAssistant, user_input: Dict[str, Any]
    ) -> str | None:
        """Validate connection to the MiniMax API."""
        api_key = user_input.get(CONF_API_KEY, "")
        base_url = user_input.get(CONF_BASE_URL, MINIMAX_DEFAULT_BASE_URL).rstrip("/")

        if not api_key:
            return "API key is required"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Validate by making a minimal chat completion request
        request_body = {
            "model": "MiniMax-M2.5-highspeed",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }

        try:
            session = async_get_clientsession(hass)
            async with session.post(
                f"{base_url}/chat/completions",
                json=request_body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.ok:
                    return None
                elif response.status == 401:
                    return "Invalid API key"
                else:
                    body = await response.text()
                    return f"HTTP {response.status}: {body[:200]}"
        except asyncio.TimeoutError:
            return "Connection timed out"
        except Exception as ex:
            return str(ex)

    async def async_get_available_models(self) -> List[str]:
        """Return available MiniMax models (hardcoded, no /v1/models endpoint)."""
        return MINIMAX_AVAILABLE_MODELS

    def _generate_stream(
        self,
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
        enable_legacy_tool_calling = entity_options.get(
            CONF_ENABLE_LEGACY_TOOL_CALLING, DEFAULT_ENABLE_LEGACY_TOOL_CALLING
        )
        tool_response_as_string = entity_options.get(
            CONF_TOOL_RESPONSE_AS_STRING, DEFAULT_TOOL_RESPONSE_AS_STRING
        )

        # Clamp temperature to MiniMax range [0.0, 1.0]
        temperature = max(0.0, min(1.0, float(temperature)))

        messages = get_oai_formatted_messages(
            conversation,
            user_content_as_list=True,
            tool_result_to_str=tool_response_as_string,
        )

        request_params: Dict[str, Any] = {
            "model": model_name,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": messages,
        }

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
            tools = get_oai_formatted_tools(
                llm_api, self._async_get_all_exposed_domains()
            )
            request_params["tools"] = tools

        headers = {"Authorization": f"Bearer {self.api_key}"}

        _LOGGER.debug(
            "MiniMax: generating completion with model=%s, %d messages and %d tools...",
            model_name,
            len(messages),
            len(tools) if tools else 0,
        )

        session = async_get_clientsession(self.hass)

        async def anext_token() -> AsyncGenerator[
            Tuple[Optional[str], Optional[List[dict]]], None
        ]:
            try:
                async with session.post(
                    f"{self.api_host}/chat/completions",
                    json=request_params,
                    timeout=timeout,
                    headers=headers,
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
                            to_say, tool_calls = self._extract_response(
                                json.loads(chunk)
                            )
                            if to_say or tool_calls:
                                yield to_say, tool_calls
            except asyncio.TimeoutError as err:
                raise HomeAssistantError(
                    "The generation request timed out! Please check your connection "
                    "settings, increase the timeout in settings, or decrease the "
                    "number of exposed entities."
                ) from err
            except aiohttp.ClientError as err:
                raise HomeAssistantError(
                    f"Failed to communicate with the MiniMax API! {err}"
                ) from err

        return self._async_stream_parse_completion(
            llm_api, agent_id, entity_options, anext_token=anext_token()
        )

    def _extract_response(
        self, response_json: dict
    ) -> Tuple[Optional[str], Optional[List[dict]]]:
        if "choices" not in response_json or len(response_json["choices"]) == 0:
            _LOGGER.warning(
                "Response missing or empty 'choices'. Keys present: %s. Full response: %s",
                list(response_json.keys()),
                response_json,
            )
            return None, None

        choice = response_json["choices"][0]
        tool_calls = None
        obj_type = response_json.get("object", "")

        if obj_type == "chat.completion":
            response_text = choice["message"]["content"]
        elif obj_type == "chat.completion.chunk":
            response_text = choice["delta"].get("content", "")
            if (
                "tool_calls" in choice["delta"]
                and choice["delta"]["tool_calls"] is not None
            ):
                tool_calls = [
                    call["function"] for call in choice["delta"]["tool_calls"]
                ]
        else:
            response_text = choice.get("text", "")

        if choice.get("finish_reason") in ("length", "content_filter"):
            _LOGGER.warning(
                "Model response did not end on a stop token (unfinished sentence)"
            )

        return response_text, tool_calls
