"""Defines the ollama compatible agent"""
from __future__ import annotations
from warnings import deprecated

import aiohttp
import asyncio
import json
import logging
from typing import Optional, Tuple, Dict, List, Any, AsyncGenerator

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.components import conversation as conversation
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers import llm

from custom_components.llama_conversation.utils import format_url, get_oai_formatted_messages, get_oai_formatted_tools
from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_TYPICAL_P,
    CONF_REQUEST_TIMEOUT,
    CONF_OPENAI_API_KEY,
    CONF_GENERIC_OPENAI_PATH,
    CONF_OLLAMA_KEEP_ALIVE_MIN,
    CONF_OLLAMA_JSON_MODE,
    CONF_CONTEXT_LENGTH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_TYPICAL_P,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_GENERIC_OPENAI_PATH,
    DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
    DEFAULT_OLLAMA_JSON_MODE,
    DEFAULT_CONTEXT_LENGTH,
)

from custom_components.llama_conversation.entity import LocalLLMClient, TextGenerationResult

_LOGGER = logging.getLogger(__name__)

@deprecated("Use the built-in Ollama integration instead")
class OllamaAPIClient(LocalLLMClient):
    api_host: str
    api_key: Optional[str]

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]) -> None:
        super().__init__(hass, client_options)
        self.api_host = format_url(
            hostname=client_options[CONF_HOST],
            port=client_options[CONF_PORT],
            ssl=client_options[CONF_SSL],
            path=client_options.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)
        )

        self.api_key = client_options.get(CONF_OPENAI_API_KEY, "")

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        host = client_options[CONF_HOST]
        port = client_options[CONF_PORT]
        ssl = client_options[CONF_SSL]
        path = "/" + client_options[CONF_GENERIC_OPENAI_PATH]
        return f"Ollama at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"

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
                    path=f"/{api_base_path}/api/tags"
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

        session = async_get_clientsession(self.hass)
        async with session.get(
            f"{self.api_host}/api/tags",
            timeout=aiohttp.ClientTimeout(total=5), # quick timeout
            headers=headers
        ) as response:
            response.raise_for_status()
            models_result = await response.json()

        return [x["name"] for x in models_result["models"]]

    def _extract_response(self, response_json: Dict) -> Tuple[Optional[str], Optional[List[llm.ToolInput]]]:
        # TODO: this doesn't work because ollama caches prompts and doesn't always return the full prompt length
        # context_len = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        # max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        # if response_json["prompt_eval_count"] + max_tokens > context_len:
        #     self._warn_context_size()

        if "response" in response_json:
            response = response_json["response"]
            tool_calls = None
            stop_reason = None
            if response_json["done"] not in ["true", True]:
                _LOGGER.warning("Model response did not end on a stop token (unfinished sentence)")
        else:
            response = response_json["message"]["content"]
            raw_tool_calls = response_json["message"].get("tool_calls")
            tool_calls = [ llm.ToolInput(tool_name=x["function"]["name"], tool_args=x["function"]["arguments"]) for x in raw_tool_calls] if raw_tool_calls else None
            stop_reason = response_json.get("done_reason")

        # _LOGGER.debug(f"{response=} {tool_calls=}")

        return response, tool_calls

    def _generate_stream(self, conversation: List[conversation.Content], llm_api: llm.APIInstance | None, user_input: conversation.ConversationInput, entity_options: Dict[str, Any]) -> AsyncGenerator[TextGenerationResult, None]:
        model_name = entity_options.get(CONF_CHAT_MODEL, "")
        context_length = entity_options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        max_tokens = entity_options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = entity_options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = entity_options.get(CONF_TOP_P, DEFAULT_TOP_P)
        top_k = entity_options.get(CONF_TOP_K, DEFAULT_TOP_K)
        typical_p = entity_options.get(CONF_TYPICAL_P, DEFAULT_TYPICAL_P)
        timeout = entity_options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        keep_alive = entity_options.get(CONF_OLLAMA_KEEP_ALIVE_MIN, DEFAULT_OLLAMA_KEEP_ALIVE_MIN)
        json_mode = entity_options.get(CONF_OLLAMA_JSON_MODE, DEFAULT_OLLAMA_JSON_MODE)

        request_params = {
            "model": model_name,
            "stream": True,
            "keep_alive": f"{keep_alive}m", # prevent ollama from unloading the model
            "options": {
                "num_ctx": context_length,
                "top_p": top_p,
                "top_k": top_k,
                "typical_p": typical_p,
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if json_mode:
            request_params["format"] = "json"

        if llm_api:
            request_params["tools"] = get_oai_formatted_tools(llm_api, self._async_get_all_exposed_domains())

        endpoint = "/api/chat"
        request_params["messages"] = get_oai_formatted_messages(conversation, tool_args_to_str=False)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        session = async_get_clientsession(self.hass)

        async def anext_token() -> AsyncGenerator[Tuple[Optional[str], Optional[List[llm.ToolInput]]], None]:
            response = None
            chunk = None
            try:
                async with session.post(
                    f"{self.api_host}{endpoint}",
                    json=request_params,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    
                    while True:
                        chunk = await response.content.readline()
                        if not chunk:
                            break
                        
                        yield self._extract_response(json.loads(chunk))
            except asyncio.TimeoutError as err:
                raise HomeAssistantError("The generation request timed out! Please check your connection settings, increase the timeout in settings, or decrease the number of exposed entities.") from err
            except aiohttp.ClientError as err:
                raise HomeAssistantError(f"Failed to communicate with the API! {err}") from err

        return self._async_parse_completion(llm_api, user_input, entity_options, anext_token=anext_token())
