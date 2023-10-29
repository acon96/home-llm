"""The Local LLaMA Conversation integration."""
from __future__ import annotations
from functools import partial

import logging
from typing import Literal

from transformers import AutoModelForCausalLM, AutoTokenizer
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_should_expose,
)
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
    """Set up Local LLaMA Conversation from a config entry."""

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data[CONF_CHAT_MODEL]

    conversation.async_set_agent(hass, entry, LLaMAAgent(hass, entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Local LLaMA."""
    hass.data[DOMAIN].pop(entry.entry_id)
    conversation.async_unset_agent(hass, entry)
    return True


class LLaMAAgent(conversation.AbstractConversationAgent):
    """Local LLaMA conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict]] = {}

        # TODO: load the model using the api endpoint
        self.model_name = self.entry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)

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
            prompt = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            try:
                prompt = self._async_generate_prompt(raw_prompt, user_input.text)

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

        _LOGGER.debug("Prompt for %s: %s", self.model_name, prompt)

        try:
            generate_parameters = {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "temperature": temperature,
            }

            response = await self._async_generate(generate_parameters)
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
        self.history[conversation_id] = prompt + response

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _generate(self, generate_params: dict) -> str:
        return generate_params["prompt"] + "\nSomething was generated!!!"

    async def _async_generate(self, prompt: str, max_new_tokens: int) -> str:
        return await self.hass.async_add_executor_job(
            self._generate, prompt, max_new_tokens
        )

    def _async_get_exposed_entities(self) -> tuple[dict[str, str], list[str]]:
        """Gather exposed entity states"""
        entity_states = {}
        domains = set()
        for state in self.hass.states.async_all():
            if not async_should_expose(self.hass, DOMAIN, state.entity_id):
                continue

            # TODO: also expose the "friendly name"
            entity_states[state.entity_id] = state.state
            domains.add(state.domain)

        _LOGGER.debug(f"Exposed entities: {entity_states}")

        return entity_states, list(domains)

    def _async_generate_prompt(self, prompt_template: str, user_input: str) -> str:
        """Generate a prompt for the user."""
        entities_to_expose, domains = self._async_get_exposed_entities()

        formatted_states = "\n".join(
            [f"{k} = {v}" for k, v in entities_to_expose.items()]
        )

        service_dict = self.hass.services.async_services()
        all_services = []
        for domain in domains:
            all_services.extend(service_dict.get(domain, {}).keys())
            # all_services.extend(
            #     [f"{domain}.{name}" for name in service_dict.get(domain, {}).keys()]
            # )
        formatted_services = ", ".join(all_services)

        return template.Template(prompt_template, self.hass).async_render(
            {
                "states": formatted_states,
                "services": formatted_services,
                "user_input": user_input,
            },
            parse_result=False,
        )
