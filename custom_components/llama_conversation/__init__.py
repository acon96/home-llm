"""The Local LLaMA Conversation integration."""
from __future__ import annotations

import logging
from typing import Literal
from types import MappingProxyType
from typing import Callable
import numpy.typing as npt
import numpy as np

from llama_cpp import Llama
import requests
import re
import os

from homeassistant.components import conversation
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import (
    async_should_expose,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, CONF_HOST, CONF_PORT, MATCH_ALL
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
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_BACKEND_TYPE,
    CONF_DOWNLOADED_MODEL_FILE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_BACKEND_TYPE,
    BACKEND_TYPE_REMOTE,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)



async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Local LLaMA Conversation from a config entry."""

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry

    use_local_backend = entry.data.get(
        CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE
    ) != BACKEND_TYPE_REMOTE

    if use_local_backend:
        _LOGGER.info(
            "Using model file '%s'", entry.data.get(CONF_DOWNLOADED_MODEL_FILE)
        )

    def create_agent():
        return LLaMAAgent(hass, entry)

    # load the model in an executor job because it takes a while and locks up the UI otherwise
    agent = await hass.async_add_executor_job(create_agent)

    conversation.async_set_agent(hass, entry, agent)
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

        self.use_local_backend = self.entry.data.get(
            CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE
        ) != BACKEND_TYPE_REMOTE

        self.api_host = None
        self.llm = None

        if self.use_local_backend:
            model_path = self.entry.data.get(CONF_DOWNLOADED_MODEL_FILE)
            if not model_path:
                raise Exception(f"Model was not found at '{model_path}'!")

            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_batch=2048,
                # n_threads=16,
                # n_threads_batch=4,
            )

            _LOGGER.info("Model loaded")
        else:
            # TODO: load the model using the api endpoint
            host = entry.data[CONF_HOST]
            port = entry.data[CONF_PORT]
            self.api_host = f"http://{host}:{port}"

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
        top_k = self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            prompt = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            try:
                prompt = [
                    {
                        "role": "system",
                        "message": self._async_generate_prompt(raw_prompt),
                    }
                ]

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

        prompt.append({"role": "user", "message": user_input.text})

        _LOGGER.debug("Prompt: %s", prompt)

        try:
            generate_parameters = {
                "prompt": await self._async_format_prompt(prompt),
                "max_new_tokens": max_new_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
            }

            response = await self._async_generate(generate_parameters)
            _LOGGER.debug("Response '%s'", response)

        except Exception as err:
            intent_response = intent.IntentResponse(language=user_input.language)
            import traceback

            traceback.print_exc()
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, there was a problem talking to the backend: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        prompt.append({"role": "assistant", "message": response})
        self.history[conversation_id] = prompt

        to_say = response.strip().split("\n")[0]
        pattern = re.compile(r"```homeassistant\n([\S\n]*)```")
        for block in pattern.findall(response.strip()):
            services = block.split("\n")
            _LOGGER.info(f"running services: {' '.join(services)}")

            for line in services:
                if len(line) == 0:
                    break

                service = line.split("(")[0]
                entity = line.split("(")[1][:-1]
                domain = entity.split(".")[0]
                try:
                    await self.hass.services.async_call(
                        domain,
                        service,
                        service_data={ATTR_ENTITY_ID: entity},
                        blocking=True,
                    )
                except Exception as err:
                    to_say += f"\nFailed to run: {line}"
                    _LOGGER.debug(f"err: {err}; {repr(err)}")

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(to_say)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _generate_remote(self, generate_params: dict) -> str:
        try:
            result = requests.post(
                f"{self.api_host}/v1/generate", json=generate_params, timeout=30
            )
            result.raise_for_status()
        except requests.RequestException as err:
            _LOGGER.debug(f"Err was: {err}")
            return "Failed to communicate with the API!"

        return result.json()["results"][0]["text"]

    def _generate_local(self, generate_params: dict) -> str:
        input_tokens = self.llm.tokenize(
            generate_params["prompt"].encode(), add_bos=False
        )

        _LOGGER.debug(f"Processing {len(input_tokens)} input tokens...")
        print(generate_params["prompt"], end="")
        output_tokens = self.llm.generate(
            input_tokens,
            temp=generate_params["temperature"],
            top_k=generate_params["top_k"],
            top_p=generate_params["top_p"],
        )

        result_tokens = []
        for token in output_tokens:
            if token == self.llm.token_eos():
                break

            result_tokens.append(token)
            print(self.llm.detokenize([token]).decode(), end="")

            if len(result_tokens) >= generate_params["max_new_tokens"]:
                break

        result = self.llm.detokenize(result_tokens).decode()
        _LOGGER.debug(f"result was: {result}")

        return result

    async def _async_generate(self, generate_parameters: dict) -> str:
        if self.use_local_backend:
            return await self.hass.async_add_executor_job(
                self._generate_local, generate_parameters
            )
        else:
            return await self.hass.async_add_executor_job(
                self._generate_remote, generate_parameters
            )

    def _async_get_exposed_entities(self) -> tuple[dict[str, str], list[str]]:
        """Gather exposed entity states"""
        entity_states = {}
        domains = set()
        for state in self.hass.states.async_all():
            if not async_should_expose(self.hass, CONVERSATION_DOMAIN, state.entity_id):
                continue

            # TODO: also expose the "friendly name"
            entity_states[state.entity_id] = { "state": state.state, "friendly_name": state.attributes["friendly_name"] }
            domains.add(state.domain)

        _LOGGER.debug(f"Exposed entities: {entity_states}")

        return entity_states, list(domains)

    async def _async_format_prompt(
        self, prompt: list[dict], include_generation_prompt: bool = True
    ) -> str:
        formatted_prompt = ""
        for message in prompt:
            role = message["role"]
            message = message["message"]
            formatted_prompt = (
                formatted_prompt + f"<|im_start|>{role} {message}<|im_end|>\n"
            )

        if include_generation_prompt:
            formatted_prompt = formatted_prompt + "<|im_start|>assistant"
        return formatted_prompt

    def _async_generate_prompt(self, prompt_template: str) -> str:
        """Generate a prompt for the user."""
        entities_to_expose, domains = self._async_get_exposed_entities()

        formatted_states = "\n".join(
            [f"{name} '{attributes['friendly_name']}' = {attributes['state']}" for name, attributes in entities_to_expose.items()]
        ) + "\n"

        service_dict = self.hass.services.async_services()
        all_services = []
        for domain in domains:
            # all_services.extend(service_dict.get(domain, {}).keys())
            all_services.extend(
                [f"{domain}.{name}" for name in service_dict.get(domain, {}).keys()]
            )
        formatted_services = ", ".join(all_services)

        return template.Template(prompt_template, self.hass).async_render(
            {
                "devices": formatted_states,
                "services": formatted_services,
            },
            parse_result=False,
        )
