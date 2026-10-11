"""Defines the various openai-like agents"""
from __future__ import annotations

from typing import Tuple, Dict, Any

from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.core import HomeAssistant

from custom_components.llama_conversation.const import (
    CONF_MAX_TOKENS,
    CONF_TOP_K,
    CONF_GBNF_GRAMMAR,
    CONF_API_PATH,
    DEFAULT_MAX_TOKENS,
)
from custom_components.llama_conversation.backends.generic_openai import GenericOpenAIAPIClient
from custom_components.llama_conversation.utils import format_url


class LlamaCppServerClient(GenericOpenAIAPIClient):
    """Backend for the HTTP server that ships with llama.cpp."""

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        host = client_options[CONF_HOST]
        port = client_options[CONF_PORT]
        ssl = client_options[CONF_SSL]
        path = "/" + client_options[CONF_API_PATH]
        return f"Llama.cpp Server at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"

    def _chat_completion_params(self, entity_options: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        max_tokens = int(entity_options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        endpoint, request_params = super()._chat_completion_params(entity_options)

        # sampling parameters that are not configured are left to the server
        if entity_options.get(CONF_TOP_K) is not None:
            request_params["top_k"] = int(entity_options[CONF_TOP_K])
        request_params["max_tokens"] = max_tokens

        grammar = (entity_options.get(CONF_GBNF_GRAMMAR) or "").strip()
        if grammar:
            request_params["grammar"] = grammar

        return endpoint, request_params
