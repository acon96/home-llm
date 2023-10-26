"""The OpenAI Conversation integration."""
from __future__ import annotations
from functools import partial

import logging
from typing import Literal

from transformers import AutoModelForCausalLM, AutoTokenizer
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, MATCH_ALL
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
)
from homeassistant.helpers import config_validation as cv, intent, selector, template
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""
    # try:
    #     await hass.async_add_executor_job(
    #         partial(
    #             openai.Engine.list,
    #             api_key=entry.data[CONF_API_KEY],
    #             request_timeout=10,
    #         )
    #     )
    # except error.AuthenticationError as err:
    #     _LOGGER.error("Invalid API key: %s", err)
    #     return False
    # except error.OpenAIError as err:
    #     raise ConfigEntryNotReady(err) from err

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data[CONF_CHAT_MODEL]

    conversation.async_set_agent(
        hass, entry, await async_create_llama_agent(hass, entry)
    )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    hass.data[DOMAIN].pop(entry.entry_id)
    conversation.async_unset_agent(hass, entry)
    return True


async def async_create_llama_agent(hass: HomeAssistant, entry: ConfigEntry):
    result = LLaMAAgent(hass, entry)
    await result.async_init()
    return result


class LLaMAAgent(conversation.AbstractConversationAgent):
    """OpenAI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict]] = {}

        self.model_name = self.entry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        self.tokenizer = None
        self.model = None

    async def async_init(self):
        self.tokenizer = await self.hass.async_add_executor_job(
            partial(
                AutoTokenizer.from_pretrained, self.model_name, trust_remote_code=True
            )
        )

        self.model = await self.hass.async_add_executor_job(
            partial(
                AutoModelForCausalLM.from_pretrained,
                self.model_name,
                trust_remote_code=True,
            )
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""

        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        max_new_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            chat_context = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            try:
                chat_context = self._async_generate_prompt(raw_prompt, user_input.text)

            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )

        _LOGGER.debug("Prompt for %s: %s", self.model_name, chat_context)

        try:
            response = await self._async_generate(chat_context, max_new_tokens)
        except Exception as err:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, there was a problem talking to the backend: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        _LOGGER.debug("Response '%s'", response)
        self.history[conversation_id] = chat_context + response

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

    async def _async_tokenize(self, prompt):
        return await self.hass.async_add_executor_job(self._tokenize, prompt)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self._tokenize(prompt)
        outputs = self.model.generate(**inputs, max_length=max_new_tokens)
        text = self.tokenizer.batch_decode(outputs)[0]
        return text

    async def _asnyc_generate(self, prompt: str, max_new_tokens: int) -> str:
        return await self.hass.async_add_executor_job(
            self._Generate, prompt, max_new_tokens
        )

    def _async_generate_prompt(self, prompt_template: str, user_input: str) -> str:
        """Generate a prompt for the user."""
        return template.Template(prompt_template, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "user_input": user_input,
            },
            parse_result=False,
        )
