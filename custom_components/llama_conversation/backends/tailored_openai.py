"""Defines the various openai-like agents"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple, Dict, List, Any

from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TOP_K,
    CONF_TYPICAL_P,
    CONF_MIN_P,
    CONF_USE_GBNF_GRAMMAR,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_TEXT_GEN_WEBUI_PRESET,
    CONF_TEXT_GEN_WEBUI_ADMIN_KEY,
    CONF_TEXT_GEN_WEBUI_CHAT_MODE,
    CONF_CONTEXT_LENGTH,
    CONF_GENERIC_OPENAI_PATH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_K,
    DEFAULT_MIN_P,
    DEFAULT_TYPICAL_P,
    DEFAULT_USE_GBNF_GRAMMAR,
    DEFAULT_GBNF_GRAMMAR_FILE,
    DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE,
    DEFAULT_CONTEXT_LENGTH,
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT,
    TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT,
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT,
)
from custom_components.llama_conversation.backends.generic_openai import GenericOpenAIAPIClient
from custom_components.llama_conversation.utils import format_url

_LOGGER = logging.getLogger(__name__)

class TextGenerationWebuiClient(GenericOpenAIAPIClient):
    admin_key: Optional[str]

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]) -> None:
        super().__init__(hass, client_options)

        self.admin_key = client_options.get(CONF_TEXT_GEN_WEBUI_ADMIN_KEY)

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        host = client_options[CONF_HOST]
        port = client_options[CONF_PORT]
        ssl = client_options[CONF_SSL]
        path = "/" + client_options[CONF_GENERIC_OPENAI_PATH]
        return f"Text-Gen WebUI at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"

    async def _async_load_model(self, entity_options: dict[str, Any]) -> None:
        model_name = entity_options.get(CONF_CHAT_MODEL)
        try:
            headers = {}
            session = async_get_clientsession(self.hass)

            if self.admin_key:
                headers["Authorization"] = f"Bearer {self.admin_key}"

            async with session.get(
                f"{self.api_host}/internal/model/info",
                headers=headers
            ) as response:
                response.raise_for_status()
                currently_loaded_result = await response.json()

            loaded_model = currently_loaded_result["model_name"]
            if loaded_model == model_name:
                _LOGGER.info(f"Model {model_name} is already loaded on the remote backend.")
                return
            else:
                _LOGGER.info(f"Model is not {model_name} loaded on the remote backend. Loading it now...")

            async with session.post(
                f"{self.api_host}/internal/model/load",
                json={
                    "model_name": model_name,
                },
                headers=headers
            ) as response:
                response.raise_for_status()

        except Exception as ex:
            _LOGGER.debug("Connection error was: %s", repr(ex))
            raise ConfigEntryNotReady("There was a problem connecting to the remote server") from ex

    def _chat_completion_params(self, entity_options: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        preset = entity_options.get(CONF_TEXT_GEN_WEBUI_PRESET)
        chat_mode = entity_options.get(CONF_TEXT_GEN_WEBUI_CHAT_MODE, DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE)

        endpoint, request_params = super()._chat_completion_params(entity_options)

        request_params["mode"] = chat_mode
        if chat_mode == TEXT_GEN_WEBUI_CHAT_MODE_CHAT or chat_mode == TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT:
            if preset:
                request_params["character"] = preset

        request_params["truncation_length"] = entity_options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        request_params["top_k"] = entity_options.get(CONF_TOP_K, DEFAULT_TOP_K)
        request_params["min_p"] = entity_options.get(CONF_MIN_P, DEFAULT_MIN_P)
        request_params["typical_p"] = entity_options.get(CONF_TYPICAL_P, DEFAULT_TYPICAL_P)

        return endpoint, request_params
    

class LlamaCppServerClient(GenericOpenAIAPIClient):
    grammar: str

    def __init__(self, hass: HomeAssistant, client_options: Dict[str, Any]):
        super().__init__(hass, client_options)

        grammar_file_name = client_options.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), grammar_file_name)) as f:
            self.grammar = "".join(f.readlines())

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        host = client_options[CONF_HOST]
        port = client_options[CONF_PORT]
        ssl = client_options[CONF_SSL]
        path = "/" + client_options[CONF_GENERIC_OPENAI_PATH]
        return f"Llama.cpp Server at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"
    
    def _chat_completion_params(self, entity_options: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        top_k = int(entity_options.get(CONF_TOP_K, DEFAULT_TOP_K))
        max_tokens = int(entity_options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        endpoint, request_params = super()._chat_completion_params(entity_options)

        request_params["top_k"] = top_k
        request_params["max_tokens"] = max_tokens

        if entity_options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            request_params["grammar"] = self.grammar

        return endpoint, request_params
