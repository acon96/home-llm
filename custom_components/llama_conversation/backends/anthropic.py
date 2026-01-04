"""Defines the Anthropic-compatible Messages API backend."""
from __future__ import annotations

import aiohttp
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic, APIError, APIConnectionError, APITimeoutError, AuthenticationError

from homeassistant.components import conversation as conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from voluptuous_openapi import convert as convert_to_openapi

from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_TOP_K,
    CONF_REQUEST_TIMEOUT,
    CONF_ENABLE_LEGACY_TOOL_CALLING,
    CONF_TOOL_RESPONSE_AS_STRING,
    CONF_API_KEY,
    CONF_API_PATH,
    CONF_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_ENABLE_LEGACY_TOOL_CALLING,
    DEFAULT_TOOL_RESPONSE_AS_STRING,
    DEFAULT_API_PATH,
)

from custom_components.llama_conversation.entity import LocalLLMClient, TextGenerationResult
from custom_components.llama_conversation.utils import get_file_contents_base64

_LOGGER = logging.getLogger(__name__)


def _convert_to_anthropic_messages(
    conversation_messages: List[conversation.Content],
    tool_result_to_str: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert Home Assistant conversation format to Anthropic Messages API format.

    Returns:
        Tuple of (system_prompt, messages_list)

    Note: Anthropic requires system prompt as a separate parameter, not in messages.
    """
    system_prompt = ""
    messages: List[Dict[str, Any]] = []

    for message in conversation_messages:
        if message.role == "system":
            # Anthropic handles system prompts separately
            system_prompt = message.content if hasattr(message, 'content') else str(message)
        elif message.role == "user":
            content = []
            msg_content = message.content if hasattr(message, 'content') else str(message)
            if msg_content:
                content.append({"type": "text", "text": msg_content})

            # Handle image attachments (Anthropic supports vision)
            if hasattr(message, 'attachments') and message.attachments:
                for attachment in message.attachments:
                    if hasattr(attachment, 'mime_type') and attachment.mime_type.startswith("image/"):
                        try:
                            image_data = get_file_contents_base64(attachment.path)
                            # get_file_contents_base64 returns data:mime;base64,xxx format
                            # Extract just the base64 part for Anthropic
                            if image_data.startswith("data:"):
                                # Remove the data URI prefix
                                image_data = image_data.split(",", 1)[1] if "," in image_data else image_data
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": attachment.mime_type,
                                    "data": image_data,
                                }
                            })
                        except Exception as e:
                            _LOGGER.warning("Failed to load image attachment: %s", e)

            if content:
                messages.append({"role": "user", "content": content})
        elif message.role == "assistant":
            content = []
            msg_content = message.content if hasattr(message, 'content') else None
            if msg_content:
                content.append({"type": "text", "text": str(msg_content)})

            # Handle tool calls (Anthropic's tool_use format)
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_id = getattr(tool_call, 'id', None) or f"toolu_{id(tool_call)}"
                    content.append({
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_call.tool_name,
                        "input": tool_call.tool_args if isinstance(tool_call.tool_args, dict) else {},
                    })

            if content:
                messages.append({"role": "assistant", "content": content})
        elif message.role == "tool_result":
            # Anthropic expects tool results in user messages with tool_result content
            tool_result = message.tool_result if hasattr(message, 'tool_result') else {}
            if tool_result_to_str:
                result_content = json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)
            else:
                result_content = str(tool_result)

            tool_call_id = getattr(message, 'tool_call_id', None) or "unknown"

            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": result_content,
                }]
            })

    return system_prompt, messages


def _convert_tools_to_anthropic_format(
    llm_api: llm.APIInstance,
) -> List[Dict[str, Any]]:
    """Convert Home Assistant LLM tools to Anthropic tool format."""
    tools: List[Dict[str, Any]] = []

    for tool in sorted(llm_api.tools, key=lambda t: t.name):
        schema = convert_to_openapi(tool.parameters, custom_serializer=llm_api.custom_serializer)
        tools.append({
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": schema,
        })

    return tools


class AnthropicAPIClient(LocalLLMClient):
    """Implements the Anthropic-compatible Messages API backend."""

    api_key: str
    base_url: str
    api_path: str

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]) -> None:
        super().__init__(hass, client_options)

        self.api_key = client_options.get(CONF_API_KEY, "")
        self.base_url = client_options.get(CONF_BASE_URL, "")
        self.api_path = client_options.get(CONF_API_PATH, DEFAULT_API_PATH)

    async def _async_build_client(self, timeout: float | None = None) -> AsyncAnthropic:
        """Build an async Anthropic client (runs in executor to avoid blocking SSL ops)."""
        effective_timeout = timeout or DEFAULT_REQUEST_TIMEOUT

        kwargs: Dict[str, Any] = {
            "timeout": effective_timeout,
            "base_url": self.base_url,
            # Use dummy key for SDK, set auth via headers for compatible API support
            "api_key": "dummy-key-for-sdk",
            "default_headers": {
                "Authorization": self.api_key,
                "x-api-key": self.api_key,
            },
        }

        def create_client():
            return AsyncAnthropic(**kwargs)

        return await self.hass.async_add_executor_job(create_client)

    @staticmethod
    def get_name(client_options: dict[str, Any]) -> str:
        base_url = client_options.get(CONF_BASE_URL, "")
        return f"Anthropic-compatible API at '{base_url}'"

    @staticmethod
    async def async_validate_connection(
        hass: HomeAssistant, user_input: Dict[str, Any]
    ) -> str | None:
        """Validate connection to the Anthropic-compatible API."""
        api_key = user_input.get(CONF_API_KEY, "")
        base_url = user_input.get(CONF_BASE_URL, "")

        if not api_key:
            return "API key is required"

        if not base_url:
            return "Base URL is required"

        try:
            kwargs: Dict[str, Any] = {
                "timeout": 10.0,
                "base_url": base_url,
                "api_key": "dummy-key-for-sdk",
                "default_headers": {
                    "Authorization": api_key,
                    "x-api-key": api_key,
                },
            }

            # Create client in executor to avoid blocking SSL operations
            def create_client():
                return AsyncAnthropic(**kwargs)

            client = await hass.async_add_executor_job(create_client)

            # Fetch models to validate connection
            await client.models.list()
            return None
        except AuthenticationError as err:
            _LOGGER.error("Anthropic authentication error: %s", err)
            return f"Invalid API key: {err}"
        except APITimeoutError as err:
            _LOGGER.error("Anthropic timeout error: %s", err)
            return "Connection timed out"
        except APIConnectionError as err:
            _LOGGER.error("Anthropic connection error: %s", err)
            return f"Connection error: {err}"
        except APIError as err:
            _LOGGER.error("Anthropic API error: status=%s, message=%s", getattr(err, 'status_code', 'N/A'), err)
            return f"API error ({getattr(err, 'status_code', 'unknown')}): {err}"
        except Exception as err:
            _LOGGER.exception("Unexpected error validating Anthropic connection")
            return f"Unexpected error: {err}"

    async def async_get_available_models(self) -> List[str]:
        """Return available models from the API."""
        try:
            client = await self._async_build_client(timeout=10)
            response = await client.models.list()
            models = []
            for model in response.data:
                model_id = getattr(model, 'id', None)
                if model_id:
                    models.append(model_id)
            if models:
                return models
        except Exception as err:
            _LOGGER.warning("Failed to fetch models from API: %s", err)

        # Try fallback with aiohttp direct request
        try:
            headers = {
                "Authorization": self.api_key,
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
            }

            base = self.base_url.rstrip("/")
            path = self.api_path.strip("/")
            models_url = f"{base}/{path}/models"

            async with aiohttp.ClientSession() as session:
                async with session.get(models_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []
                        for model in data.get("data", []):
                            model_id = model.get("id")
                            if model_id:
                                models.append(model_id)
                        if models:
                            return models
        except Exception as err:
            _LOGGER.debug("Fallback models fetch also failed: %s", err)

        return []

    def _supports_vision(self, entity_options: dict[str, Any]) -> bool:
        """Anthropic models support vision."""
        return True

    def _generate_stream(
        self,
        conversation: List[conversation.Content],
        llm_api: llm.APIInstance | None,
        agent_id: str,
        entity_options: dict[str, Any],
    ) -> AsyncGenerator[TextGenerationResult, None]:
        """Generate streaming response using Anthropic's Messages API."""

        model_name = entity_options.get(CONF_CHAT_MODEL, "")
        max_tokens = int(entity_options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        temperature = entity_options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = entity_options.get(CONF_TOP_P, DEFAULT_TOP_P)
        top_k = entity_options.get(CONF_TOP_K, DEFAULT_TOP_K)
        timeout = entity_options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        enable_legacy_tool_calling = entity_options.get(
            CONF_ENABLE_LEGACY_TOOL_CALLING, DEFAULT_ENABLE_LEGACY_TOOL_CALLING
        )
        tool_response_as_string = entity_options.get(
            CONF_TOOL_RESPONSE_AS_STRING, DEFAULT_TOOL_RESPONSE_AS_STRING
        )

        # Convert conversation to Anthropic format
        system_prompt, messages = _convert_to_anthropic_messages(
            conversation, tool_result_to_str=tool_response_as_string
        )

        # Prepare tools if available and not using legacy tool calling
        tools = None
        if llm_api and not enable_legacy_tool_calling:
            tools = _convert_tools_to_anthropic_format(llm_api)

        _LOGGER.debug(
            "Generating completion with model=%s, %d messages, and %d tools...",
            model_name,
            len(messages),
            len(tools) if tools else 0,
        )

        async def anext_token() -> AsyncGenerator[Tuple[Optional[str], Optional[List[dict]]], None]:
            client = await self._async_build_client(timeout=timeout)

            request_params: Dict[str, Any] = {
                "model": model_name,
                "max_tokens": max_tokens,
                "messages": messages,
            }

            # Add optional parameters
            if system_prompt:
                request_params["system"] = system_prompt
            if tools:
                request_params["tools"] = tools
            if temperature is not None:
                request_params["temperature"] = temperature
            if top_p is not None:
                request_params["top_p"] = top_p
            if top_k is not None and top_k > 0:
                request_params["top_k"] = top_k

            try:
                current_tool_call: Dict[str, Any] | None = None

                async with client.messages.stream(**request_params) as stream:
                    async for event in stream:
                        event_type = getattr(event, 'type', None)

                        if event_type == "content_block_start":
                            block = getattr(event, 'content_block', None)
                            if block and getattr(block, 'type', None) == "tool_use":
                                current_tool_call = {
                                    "id": getattr(block, 'id', ''),
                                    "name": getattr(block, 'name', ''),
                                    "input": "",
                                }
                        elif event_type == "content_block_delta":
                            delta = getattr(event, 'delta', None)
                            if delta:
                                delta_type = getattr(delta, 'type', None)
                                if delta_type == "text_delta":
                                    text = getattr(delta, 'text', '')
                                    if text:
                                        yield text, None
                                elif delta_type == "input_json_delta":
                                    if current_tool_call:
                                        partial_json = getattr(delta, 'partial_json', '')
                                        current_tool_call["input"] += partial_json
                        elif event_type == "content_block_stop":
                            if current_tool_call:
                                # Parse the accumulated JSON and yield the tool call
                                try:
                                    tool_args = json.loads(current_tool_call["input"]) if current_tool_call["input"] else {}
                                except json.JSONDecodeError:
                                    tool_args = {}

                                tool_call_dict = {
                                    "function": {
                                        "name": current_tool_call["name"],
                                        "arguments": tool_args,
                                    },
                                    "id": current_tool_call["id"],
                                }
                                yield None, [tool_call_dict]
                                current_tool_call = None
                        elif event_type == "message_stop":
                            break

            except APITimeoutError as err:
                raise HomeAssistantError(
                    "The generation request timed out! Please check your connection "
                    "settings, increase the timeout in settings, or decrease the "
                    "number of exposed entities."
                ) from err
            except APIConnectionError as err:
                raise HomeAssistantError(
                    f"Failed to connect to the Anthropic-compatible API: {err}"
                ) from err
            except APIError as err:
                raise HomeAssistantError(
                    f"Anthropic API error: {err}"
                ) from err

        return self._async_stream_parse_completion(
            llm_api, agent_id, entity_options, anext_token=anext_token()
        )
