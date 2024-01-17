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

import homeassistant.components.conversation as ha_conversation
from homeassistant.components.conversation import ConversationInput, ConversationResult, AbstractConversationAgent
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
    CONF_USE_GBNF_GRAMMAR,
    CONF_TEXT_GEN_WEBUI_PRESET,
    CONF_OPENAI_API_KEY,
    CONF_TEXT_GEN_WEBUI_ADMIN_KEY,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_SERVICE_CALL_REGEX,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_BACKEND_TYPE,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_USE_GBNF_GRAMMAR,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_SERVICE_CALL_REGEX,
    DEFAULT_OPTIONS,
    BACKEND_TYPE_TEXT_GEN_WEBUI,
    BACKEND_TYPE_GENERIC_OPENAI,
    DOMAIN,
    GBNF_GRAMMAR_FILE,
    PROMPT_TEMPLATE_DESCRIPTIONS,
)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

def is_local_backend(backend):
    return backend not in [BACKEND_TYPE_TEXT_GEN_WEBUI, BACKEND_TYPE_GENERIC_OPENAI]

async def update_listener(hass, entry):
    """Handle options update."""
    hass.data[DOMAIN][entry.entry_id] = entry
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Local LLaMA Conversation from a config entry."""

    use_local_backend = is_local_backend(
        entry.data.get(CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE)
    )

    if use_local_backend:
        _LOGGER.info(
            "Using model file '%s'", entry.data.get(CONF_DOWNLOADED_MODEL_FILE)
        )

    def create_agent():
        return LLaMAAgent(hass, entry)

    # load the model in an executor job because it takes a while and locks up the UI otherwise
    agent = await hass.async_add_executor_job(create_agent)

    # handle updates to the options
    entry.async_on_unload(entry.add_update_listener(update_listener))

    ha_conversation.async_set_agent(hass, entry, agent)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Local LLaMA."""
    hass.data[DOMAIN].pop(entry.entry_id)
    ha_conversation.async_unset_agent(hass, entry)
    return True

async def async_migrate_entry(hass, config_entry: ConfigEntry):
    """Migrate old entry."""
    _LOGGER.debug("Migrating from version %s", config_entry.version)

    if config_entry.version > 1:
      # This means the user has downgraded from a future version
      return False

    # if config_entry.version < 2:
    #     # just ensure that the defaults are set
    #     new_options = dict(DEFAULT_OPTIONS)
    #     new_options.update(config_entry.options)

    #     config_entry.version = 2
    #     hass.config_entries.async_update_entry(config_entry, options=new_options)

    _LOGGER.debug("Migration to version %s successful", config_entry.version)

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

class LLaMAAgent(AbstractConversationAgent):
    """Local LLaMA conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry_id = entry.entry_id
        self.history: dict[str, list[dict]] = {}

        self.backend_type = entry.data.get(
            CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE
        )
        self.use_local_backend = is_local_backend(self.backend_type)

        self.api_host = None
        self.llm = None
        self.grammar = None

        self.model_path = entry.data.get(CONF_DOWNLOADED_MODEL_FILE)
        self.model_name = entry.data.get(CONF_CHAT_MODEL, self.model_path)

        if self.use_local_backend:
            self._load_local_model()
        else:
            host = entry.data[CONF_HOST]
            port = entry.data[CONF_PORT]
            self.api_host = f"http://{host}:{port}"

            # only load model if using text-generation-webui
            if self.backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI:
                api_key = entry.data.get(CONF_TEXT_GEN_WEBUI_ADMIN_KEY, entry.data.get(CONF_OPENAI_API_KEY))
                self._load_remote_model(api_key)

    @property
    def entry(self):
        return self.hass.data[DOMAIN][self.entry_id]

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process a sentence."""

        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        refresh_system_prompt = self.entry.options.get(CONF_REFRESH_SYSTEM_PROMPT, DEFAULT_REFRESH_SYSTEM_PROMPT)
        service_call_regex = self.entry.options.get(CONF_SERVICE_CALL_REGEX, DEFAULT_SERVICE_CALL_REGEX)

        try:
            service_call_pattern = re.compile(service_call_regex)
        except Exception as err:
            _LOGGER.exception("There was a problem compiling the service call regex")
            
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, there was a problem compiling the service call regex: {err}",
            )
            return ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            conversation = self.history[conversation_id]            
        else:
            conversation_id = ulid.ulid()
            conversation = []
        
        if len(conversation) == 0 or refresh_system_prompt:
            try:
                message = self._async_generate_system_prompt(raw_prompt)
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            
            system_prompt = { "role": "system", "message": message }
            
            if len(conversation) == 0:
                conversation.append(system_prompt)
            else:
                conversation[0] = system_prompt

        conversation.append({"role": "user", "message": user_input.text})

        # _LOGGER.debug("Prompt: %s", prompt)

        try:
            prompt = await self._async_format_prompt(conversation)

            _LOGGER.debug(prompt)
            response = await self._async_generate(prompt)
            _LOGGER.debug(response)

        except Exception as err:
            _LOGGER.exception("There was a problem talking to the backend")
            
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, there was a problem talking to the backend: {err}",
            )
            return ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        conversation.append({"role": "assistant", "message": response})
        self.history[conversation_id] = conversation

        exposed_entities = list(self._async_get_exposed_entities()[0].keys())
        
        to_say = service_call_pattern.sub("", response).strip()
        for block in service_call_pattern.findall(response.strip()):
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
                    try:
                        service = line.split("(")[0]
                        entity = line.split("(")[1][:-1]
                        domain, service = tuple(service.split("."))
                    except Exception:
                        to_say += f" Failed to parse call from '{line}'!"
                        continue

                # only acknowledge requests to exposed entities
                if entity not in exposed_entities:
                    to_say += f" Can't find device '{entity}'!"
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

        to_say = to_say.replace("<|im_end|>", "") # remove the eos token if it is returned (some backends + the old model does this)
        
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(to_say)
        return ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _load_remote_model(self, admin_key: str | None):
        try:
            currently_loaded_result = requests.get(f"{self.api_host}/v1/internal/model/info")
            currently_loaded_result.raise_for_status()

            loaded_model = currently_loaded_result.json()["model_name"]
            if loaded_model == self.model_name:
                _LOGGER.info(f"Model {self.model_name} is already loaded on the remote backend.")
            else:
                _LOGGER.info(f"Model is not {self.model_name} loaded on the remote backend. Loading it now...")
            
            headers = {}
            if admin_key:
                headers["Authorization"] = f"Basic {admin_key}"
            
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

    def _generate_remote(self, prompt: str) -> str:
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        timeout = self.entry.options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)

        request_params = {
            "prompt": prompt,
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        headers = {}
        api_key = self.entry.data.get(CONF_OPENAI_API_KEY)
        if api_key:
            headers["Authorization"] = f"Basic {api_key}"

        if self.backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI:
            preset = self.entry.options.get(CONF_TEXT_GEN_WEBUI_PRESET)
            if preset:
                request_params["preset"] = preset

        result = requests.post(
            f"{self.api_host}/v1/completions", 
            json=request_params,
            timeout=timeout,
            headers=headers,
        )

        try:
            result.raise_for_status()
        except requests.RequestException as err:
            _LOGGER.debug(f"Err was: {err}")
            _LOGGER.debug(f"Request was: {request_params}")
            _LOGGER.debug(f"Result was: {result.text}")
            return f"Failed to communicate with the API! {err}"

        choices = result.json()["choices"]

        if choices[0]["finish_reason"] != "stop":
            _LOGGER.warn("Model response did not end on a stop token (unfinished sentence)")

        return choices[0]["text"]
    
    def _load_local_model(self):
        if not self.model_path:
            raise Exception(f"Model was not found at '{self.model_path}'!")

        # don't import it until now because the wheel is installed by config_flow.py
        module = importlib.import_module("llama_cpp")
        Llama = getattr(module, "Llama")
        LlamaGrammar = getattr(module, "LlamaGrammar")

        _LOGGER.debug("Loading model...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_batch=2048,
            # TODO: expose arguments to the user in home assistant UI
            # n_threads=16,
            # n_threads_batch=4,
        )

        _LOGGER.debug("Loading grammar...")
        try:
            # TODO: make grammar configurable
            with open(os.path.join(os.path.dirname(__file__), GBNF_GRAMMAR_FILE)) as f:
                grammar_str = "".join(f.readlines())
            self.grammar = LlamaGrammar.from_string(grammar_str)
            _LOGGER.debug("Loaded grammar")
        except Exception:
            _LOGGER.exception("Failed to load grammar!")
            self.grammar = None


    def _generate_local(self, prompt: str) -> str:
        input_tokens = self.llm.tokenize(
            prompt.encode(), add_bos=False
        )

        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_k = int(self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K))
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        grammar = self.grammar if self.entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR) else None

        _LOGGER.debug(f"Options: {self.entry.options}")

        _LOGGER.debug(f"Processing {len(input_tokens)} input tokens...")
        output_tokens = self.llm.generate(
            input_tokens,
            temp=temperature,
            top_k=top_k,
            top_p=top_p,
            grammar=grammar
        )

        result_tokens = []
        for token in output_tokens:
            if token == self.llm.token_eos():
                break

            result_tokens.append(token)

            if len(result_tokens) >= max_tokens:
                break

        result = self.llm.detokenize(result_tokens).decode()

        return result

    async def _async_generate(self, prompt: str) -> str:
        if self.use_local_backend:
            return await self.hass.async_add_executor_job(
                self._generate_local, prompt
            )
        else:
            return await self.hass.async_add_executor_job(
                self._generate_remote, prompt
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

        # _LOGGER.debug(f"Exposed entities: {entity_states}")

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

    def _async_generate_system_prompt(self, prompt_template: str) -> str:
        """Generate a prompt for the user."""
        entities_to_expose, domains = self._async_get_exposed_entities()

        extra_attributes_to_expose = self.entry.options \
            .get(CONF_EXTRA_ATTRIBUTES_TO_EXPOSE, DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE)

        def expose_attributes(attributes):
            result = attributes["state"]
            for attribute_name in extra_attributes_to_expose:
                if attribute_name not in attributes:
                    continue

                _LOGGER.debug(f"{attribute_name} = {attributes[attribute_name]}")

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
                        value = f"vol={int(value*100)}"

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
