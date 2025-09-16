"""Defines the various openai-like agents"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from custom_components.llama_conversation.const import (
    CONF_MAX_TOKENS,
    CONF_TOP_K,
    CONF_TYPICAL_P,
    CONF_MIN_P,
    CONF_USE_GBNF_GRAMMAR,
    CONF_TEXT_GEN_WEBUI_PRESET,
    CONF_TEXT_GEN_WEBUI_ADMIN_KEY,
    CONF_TEXT_GEN_WEBUI_CHAT_MODE,
    CONF_CONTEXT_LENGTH,
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

_LOGGER = logging.getLogger(__name__)

class TextGenerationWebuiClient(GenericOpenAIAPIClient):
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
    

async def _async_validate_text_generation_webui(self, user_input: dict) -> tuple:
    """
    Validates a connection to text-generation-webui and that the model exists on the remote server

    :param user_input: the input dictionary used to build the connection
    :return: a tuple of (error message name, exception detail); both can be None
    """
    try:
        headers = {}
        api_key = user_input.get(CONF_TEXT_GEN_WEBUI_ADMIN_KEY, user_input.get(CONF_OPENAI_API_KEY))
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        session = async_get_clientsession(self.hass)
        async with session.get(
            format_url(
                hostname=self.model_config[CONF_HOST],
                port=self.model_config[CONF_PORT],
                ssl=self.model_config[CONF_SSL],
                path="/v1/internal/model/list"
            ),
            timeout=5, # quick timeout
            headers=headers
        ) as response:
            response.raise_for_status()
            models = await response.json()

        for model in models["model_names"]:
            if model == self.model_config[CONF_CHAT_MODEL].replace("/", "_"):
                return None, None, []

        return "missing_model_api", None, models["model_names"]

    except Exception as ex:
        _LOGGER.info("Connection error was: %s", repr(ex))
        return "failed_to_connect", ex, []

class LlamaCppServerClient(GenericOpenAIAPIClient):
    grammar: str

    async def _async_load_model(self, entry: ConfigEntry):
        await super()._async_load_model(entry)

        return await self.hass.async_add_executor_job(
            self._load_model, entry
        )

    def _load_model(self, entry: ConfigEntry):
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), DEFAULT_GBNF_GRAMMAR_FILE)) as f:
            self.grammar = "".join(f.readlines())
    
    def _chat_completion_params(self, entity_options: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        top_k = int(entity_options.get(CONF_TOP_K, DEFAULT_TOP_K))
        endpoint, request_params = super()._chat_completion_params(entity_options)

        request_params["top_k"] = top_k

        if entity_options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            request_params["grammar"] = self.grammar

        return endpoint, request_params
