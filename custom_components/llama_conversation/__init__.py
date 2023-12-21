"""The Local LLaMA Conversation integration."""
from __future__ import annotations

import logging
from typing import Literal

from huggingface_hub import hf_hub_download
import requests
import re

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
    CONF_USE_LOCAL_BACKEND,
    CONF_DOWNLOADED_MODEL_FILE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_USE_LOCAL_BACKEND,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Local LLaMA Conversation from a config entry."""

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry

    model_name = entry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
    use_local_backend = entry.data.get(CONF_USE_LOCAL_BACKEND, DEFAULT_USE_LOCAL_BACKEND)

    if use_local_backend:
        _LOGGER.info("Downloading/Caching model '%s'...", model_name)
        expected_filename = model_name.split("/")[1] + ".q5_k_m.gguf"
        destination_file = hf_hub_download(
            repo_id=model_name,
            repo_type="model",
            filename=expected_filename
        )

        entry.data[CONF_DOWNLOADED_MODEL_FILE] = destination_file

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

        self.model_name = self.entry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        self.use_local_backend = self.entry.data.get(CONF_USE_LOCAL_BACKEND, DEFAULT_USE_LOCAL_BACKEND)

        self.api_host = None
        self.llm = None

        if self.use_local_backend:
            from llama_cpp import Llama
            model_path = self.entry.data.get(CONF_DOWNLOADED_MODEL_FILE)
            if not model_path:
                raise Exception("Model was not successfully downloaded!")
            
            self.llm = Llama(model_path=model_path)
        else:

            # TODO: load the model using the api endpoint
            host = entry.data.get(CONF_HOST, DEFAULT_HOST)
            port = entry.data.get(CONF_PORT, DEFAULT_PORT)
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
                prompt = [{ "role": "system", "message": self._async_generate_prompt(raw_prompt) }]

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
            
        prompt.append({ "role": "user", "message": user_input.text })

        _LOGGER.debug("Prompt for %s: %s", self.model_name, prompt)

        try:
            generate_parameters = {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
            }

            response = await self._async_generate(generate_parameters)
            _LOGGER.debug("Response '%s'", response)

        except Exception as err:
            intent_response = intent.IntentResponse(language=user_input.language)
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
                f"{self.api_host}/api/v1/generate", json=generate_params, timeout=30
            )
            result.raise_for_status()
        except requests.RequestException as err:
            _LOGGER.debug(f"Err was: {err}")
            return "Failed to communicate with the API!"

        return result.json()["results"][0]["text"]
    
    def _generate_local(self, generate_params: dict) -> str:
        tokens = self.llm.tokenize(generate_params["prompt"])

        result = self.llm.generate(
            tokens,
            temp=generate_params["temperture"],
            top_k=generate_params["top_k"],
            top_p=generate_params["top_p"],
        )

        return self.llm.detokenize(result)

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
            entity_states[state.entity_id] = state.state
            domains.add(state.domain)

        _LOGGER.debug(f"Exposed entities: {entity_states}")

        return entity_states, list(domains)
    
    def _format_prompt(self, prompt: list[dict], include_generation_prompt: bool = True) -> str:
        formatted_prompt = ""
        for message in prompt:
            role = message["role"]
            message = message["message"]
            formatted_prompt = formatted_prompt + f"<|im_start|>{role} {message}<|im_end|>\n"

        if include_generation_prompt:
            formatted_prompt = formatted_prompt + "<|im_start|>assistant"
        return formatted_prompt

    def _async_generate_prompt(self, prompt_template: str) -> str:
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
                "devices": formatted_states,
                "services": formatted_services,
            },
            parse_result=False,
        )
