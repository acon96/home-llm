"""The Local LLaMA Conversation integration."""
from __future__ import annotations

import logging
import importlib
from typing import Literal

import requests
import re
import os
import json
import webcolors

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
    CONF_REQUEST_TIMEOUT,
    CONF_BACKEND_TYPE,
    CONF_DOWNLOADED_MODEL_FILE,
    CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
    CONF_PROMPT_TEMPLATE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_BACKEND_TYPE,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
    DEFAULT_PROMPT_TEMPLATE,
    BACKEND_TYPE_REMOTE,
    DOMAIN,
    GBNF_GRAMMAR_FILE,
    PROMPT_TEMPLATE_DESCRIPTIONS,
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

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

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
        self.grammar = None

        model_path = self.entry.data.get(CONF_DOWNLOADED_MODEL_FILE)
        self.model_name = self.entry.data.get(CONF_CHAT_MODEL, model_path)
        self.extra_attributes_to_expose = self.entry.data \
            .get(CONF_EXTRA_ATTRIBUTES_TO_EXPOSE, DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE).split(",")

        if self.use_local_backend:
            if not model_path:
                raise Exception(f"Model was not found at '{model_path}'!")
            
            # don't import it until now because the wheel is installed by config_flow.py
            module = importlib.import_module("llama_cpp")
            Llama = getattr(module, "Llama")
            LlamaGrammar = getattr(module, "LlamaGrammar")

            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_batch=2048,
                # n_threads=16,
                # n_threads_batch=4,
            )

            with open(os.path.join(os.path.dirname(__file__), GBNF_GRAMMAR_FILE)) as f:
                grammar_str = "".join(f.readlines())
            self.grammar = LlamaGrammar.from_string(grammar_str)

            _LOGGER.info("Model loaded")
        else:
            host = entry.data[CONF_HOST]
            port = entry.data[CONF_PORT]
            self.api_host = f"http://{host}:{port}"

            self._load_remote_model()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""

        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
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
                "max_tokens": max_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
            }

            _LOGGER.debug(generate_parameters["prompt"])
            response = await self._async_generate(generate_parameters)
            _LOGGER.debug(response)

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

        exposed_entities = list(self._async_get_exposed_entities()[0].keys())
        pattern = re.compile(r"```homeassistant\n([\S\n]*)```")
        
        to_say = pattern.sub("", response).strip()
        for block in pattern.findall(response.strip()):
            services = block.split("\n")
            _LOGGER.info(f"running services: {' '.join(services)}")

            for line in services:
                if len(line) == 0:
                    break

                # parse old format or JSON format
                try:
                    json_output = json.loads(line)
                    service = json_output["service"]
                    entity = json_output["target_device"]
                    domain, service = tuple(service.split("."))
                except Exception:
                    service = line.split("(")[0]
                    entity = line.split("(")[1][:-1]
                    domain, service = tuple(service.split("."))

                # only acknowledge requests to exposed entities
                if entity not in exposed_entities:
                    to_say += f"Can't find device '{entity}'"
                else:
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
    
    def _load_remote_model(self):
        # TODO: check if model is already loaded
        try:
            load_result = requests.post(
                f"{self.api_host}/v1/internal/model/load",
                json={
                    "model_name": self.model_name,
                    # TODO: expose arguments to the user in home assistant UI
                    # "args": {},
                }
            )
            load_result.raise_for_status()

        except Exception as ex:
            _LOGGER.error("Connection error was: %s", repr(ex))

    def _generate_remote(self, generate_params: dict) -> str:
        try:
            generate_params["model"] = self.model_name
            del generate_params["top_k"]

            timeout = self.entry.options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)

            result = requests.post(
                f"{self.api_host}/v1/completions", json=generate_params, timeout=timeout
            )
            result.raise_for_status()
        except requests.RequestException as err:
            _LOGGER.debug(f"Err was: {err}")
            return "Failed to communicate with the API!"

        choices = result.json()["choices"]

        if choices[0]["finish_reason"] != "stop":
            _LOGGER.warn("Model response did not end on a stop token (unfinished sentence)")

        return choices[0]["text"]

    def _generate_local(self, generate_params: dict) -> str:
        input_tokens = self.llm.tokenize(
            generate_params["prompt"].encode(), add_bos=False
        )

        _LOGGER.debug(f"Processing {len(input_tokens)} input tokens...")
        output_tokens = self.llm.generate(
            input_tokens,
            temp=generate_params["temperature"],
            top_k=generate_params["top_k"],
            top_p=generate_params["top_p"],
            grammar=self.grammar
        )

        result_tokens = []
        for token in output_tokens:
            if token == self.llm.token_eos():
                break

            result_tokens.append(token)

            if len(result_tokens) >= generate_params["max_tokens"]:
                break

        result = self.llm.detokenize(result_tokens).decode()

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

            attributes = dict(state.attributes)
            attributes["state"] = state.state
            entity_states[state.entity_id] = attributes
            domains.add(state.domain)

        _LOGGER.debug(f"Exposed entities: {entity_states}")

        return entity_states, list(domains)

    async def _async_format_prompt(
        self, prompt: list[dict], include_generation_prompt: bool = True
    ) -> str:
        formatted_prompt = ""

        prompt_template = self.entry.options.get(CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE)
        template_desc = PROMPT_TEMPLATE_DESCRIPTIONS[prompt_template]

        for message in prompt:
            role = message["role"]
            message = message["message"]
            role_desc = template_desc[role]
            formatted_prompt = (
                formatted_prompt + f"{role_desc['prefix']}{message}{role_desc['suffix']}\n"
            )

        if include_generation_prompt:
            formatted_prompt = formatted_prompt + template_desc["generation_prompt"]
            
        return formatted_prompt

    def _async_generate_prompt(self, prompt_template: str) -> str:
        """Generate a prompt for the user."""
        entities_to_expose, domains = self._async_get_exposed_entities()

        def expose_attributes(attributes):
            result = attributes["state"]
            for attribute_name in self.extra_attributes_to_expose:
                if attribute_name not in attributes:
                    continue

                _LOGGER.info(f"{attribute_name} = {attributes[attribute_name]}")

                value = attributes[attribute_name]
                if value is not None:
                    if attribute_name == "current_temperature":
                        value = int(value)
                        if value > 50:
                            value = f"{value}F"
                        else:
                            value = f"{value}C"
                    elif attribute_name == "rgb_color":
                        value = F"{closest_color(value)} {value}"
                    elif attribute_name == "volume_level":
                        value = f"{int(value*100)}%"
                        
                    result = result + ";" + str(value)
            return result

        formatted_states = "\n".join(
            [f"{name} '{attributes.get('friendly_name')}' = {expose_attributes(attributes)}" for name, attributes in entities_to_expose.items()]
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
