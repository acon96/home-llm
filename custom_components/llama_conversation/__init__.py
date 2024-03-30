"""The Local LLaMA Conversation integration."""
from __future__ import annotations

import logging
import threading
import importlib
from typing import Literal, Any, Callable

import requests
import re
import os
import json
import csv
import random
import time

import homeassistant.components.conversation as ha_conversation
from homeassistant.components.conversation import ConversationInput, ConversationResult, AbstractConversationAgent
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, CONF_HOST, CONF_PORT, CONF_SSL, MATCH_ALL
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady, ConfigEntryError, TemplateError
from homeassistant.helpers import config_validation as cv, intent, template, entity_registry as er
from homeassistant.helpers.event import async_track_state_change
from homeassistant.util import ulid

from .utils import closest_color, flatten_vol_schema, install_llama_cpp_python
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
    CONF_ALLOWED_SERVICE_CALL_ARGUMENTS,
    CONF_PROMPT_TEMPLATE,
    CONF_USE_GBNF_GRAMMAR,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    CONF_IN_CONTEXT_EXAMPLES_FILE,
    CONF_TEXT_GEN_WEBUI_PRESET,
    CONF_OPENAI_API_KEY,
    CONF_TEXT_GEN_WEBUI_ADMIN_KEY,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_PROMPT_CACHING_ENABLED,
    CONF_PROMPT_CACHING_INTERVAL,
    CONF_SERVICE_CALL_REGEX,
    CONF_REMOTE_USE_CHAT_ENDPOINT,
    CONF_TEXT_GEN_WEBUI_CHAT_MODE,
    CONF_OLLAMA_KEEP_ALIVE_MIN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_BACKEND_TYPE,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
    DEFAULT_ALLOWED_SERVICE_CALL_ARGUMENTS,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_USE_GBNF_GRAMMAR,
    DEFAULT_GBNF_GRAMMAR_FILE,
    DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    DEFAULT_IN_CONTEXT_EXAMPLES_FILE,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_NUM_INTERACTIONS,
    DEFAULT_PROMPT_CACHING_ENABLED,
    DEFAULT_PROMPT_CACHING_INTERVAL,
    DEFAULT_SERVICE_CALL_REGEX,
    DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
    DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE,
    DEFAULT_OPTIONS,
    DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
    BACKEND_TYPE_LLAMA_HF,
    BACKEND_TYPE_LLAMA_EXISTING,
    BACKEND_TYPE_TEXT_GEN_WEBUI,
    BACKEND_TYPE_GENERIC_OPENAI,
    BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER,
    BACKEND_TYPE_OLLAMA,
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT,
    TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT,
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT,
    DOMAIN,
    PROMPT_TEMPLATE_DESCRIPTIONS,
)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

async def update_listener(hass: HomeAssistant, entry: ConfigEntry):
    """Handle options update."""
    hass.data[DOMAIN][entry.entry_id] = entry
    
    # call update handler
    agent: LLaMAAgent = await ha_conversation._get_agent_manager(hass).async_get_agent(entry.entry_id)
    agent._update_options()

    backend_type = entry.data.get(CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE)
    if backend_type in [ BACKEND_TYPE_LLAMA_HF, BACKEND_TYPE_LLAMA_EXISTING ]:
        if entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED):
            hass.async_create_task(agent._async_cache_prompt(None, None, None))

    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Local LLaMA Conversation from a config entry."""

    def create_agent(backend_type):
        agent_cls = None

        if backend_type in [ BACKEND_TYPE_LLAMA_HF, BACKEND_TYPE_LLAMA_EXISTING ]:
            agent_cls = LocalLLaMAAgent
        elif backend_type == BACKEND_TYPE_GENERIC_OPENAI:
            agent_cls = GenericOpenAIAPIAgent
        elif backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI:
            agent_cls = TextGenerationWebuiAgent
        elif backend_type == BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER:
            agent_cls = LlamaCppPythonAPIAgent
        elif backend_type == BACKEND_TYPE_OLLAMA:
            agent_cls = OllamaAPIAgent
        
        return agent_cls(hass, entry)

    # load the model in an executor job because it takes a while and locks up the UI otherwise
    backend_type = entry.data.get(CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE)
    agent = await hass.async_add_executor_job(create_agent, backend_type)

    # handle updates to the options
    entry.async_on_unload(entry.add_update_listener(update_listener))

    ha_conversation.async_set_agent(hass, entry, agent)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry

    if backend_type in [ BACKEND_TYPE_LLAMA_HF, BACKEND_TYPE_LLAMA_EXISTING ]:
        if entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED):
            hass.async_create_task(agent._async_cache_prompt(None, None, None))

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

class LLaMAAgent(AbstractConversationAgent):
    """Local LLaMA conversation agent."""

    hass: Any
    entry_id: str
    history: dict[str, list[dict]]
    in_context_examples: list[dict]

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry_id = entry.entry_id
        self.history = {}

        self.backend_type = entry.data.get(
            CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE
        )

        self.in_context_examples = None
        if entry.options.get(CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES, DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES):
            self._load_icl_examples()

        self._load_model(entry)

    def _load_icl_examples(self):
        try:
            icl_file = self.entry.options.get(CONF_IN_CONTEXT_EXAMPLES_FILE, DEFAULT_IN_CONTEXT_EXAMPLES_FILE)
            icl_filename = os.path.join(os.path.dirname(__file__), icl_file)

            with open(icl_filename) as f:
                self.in_context_examples = list(csv.DictReader(f))

                if set(self.in_context_examples[0].keys()) != set(["service", "response" ]):
                    raise Exception("ICL csv file did not have 2 columns: service & response")

            _LOGGER.debug(f"Loaded {len(self.in_context_examples)} examples for ICL")
        except Exception:
            _LOGGER.exception("Failed to load in context learning examples!")
            self.in_context_examples = None

    def _update_options(self):
        if self.entry.options.get(CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES, DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES):
            self._load_icl_examples()
        else:
            self.in_context_examples = None

    @property
    def entry(self) -> ConfigEntry:
        return self.hass.data[DOMAIN][self.entry_id]

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL
    
    def _load_model(self, entry: ConfigEntry) -> None:
        raise NotImplementedError()
    
    def _generate(self, conversation: dict) -> str:
        raise NotImplementedError()

    async def _async_generate(self, conversation: dict) -> str:
        return await self.hass.async_add_executor_job(
            self._generate, conversation
        )

    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process a sentence."""

        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        prompt_template = self.entry.options.get(CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE)
        template_desc = PROMPT_TEMPLATE_DESCRIPTIONS[prompt_template]
        refresh_system_prompt = self.entry.options.get(CONF_REFRESH_SYSTEM_PROMPT, DEFAULT_REFRESH_SYSTEM_PROMPT)
        remember_conversation = self.entry.options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION)
        remember_num_interactions = self.entry.options.get(CONF_REMEMBER_NUM_INTERACTIONS, DEFAULT_REMEMBER_NUM_INTERACTIONS)
        service_call_regex = self.entry.options.get(CONF_SERVICE_CALL_REGEX, DEFAULT_SERVICE_CALL_REGEX)
        allowed_service_call_arguments = self.entry.options \
            .get(CONF_ALLOWED_SERVICE_CALL_ARGUMENTS, DEFAULT_ALLOWED_SERVICE_CALL_ARGUMENTS)

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
            conversation = self.history[conversation_id] if remember_conversation else [self.history[conversation_id][0]]
        else:
            conversation_id = ulid.ulid()
            conversation = []
        
        if len(conversation) == 0 or refresh_system_prompt:
            try:
                message = self._generate_system_prompt(raw_prompt)
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
                if not remember_conversation:
                    self.history[conversation_id] = conversation
            else:
                conversation[0] = system_prompt

        conversation.append({"role": "user", "message": user_input.text})

        try:
            _LOGGER.debug(conversation)
            response = await self._async_generate(conversation)
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
        if remember_conversation:
            if remember_num_interactions and len(conversation) > (remember_num_interactions * 2) + 1:
                for i in range(0,2):
                    conversation.pop(1)
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
                    if "to_say" in json_output:
                        to_say = to_say + json_output.pop("to_say")

                    extra_arguments = { k: v for k, v in json_output.items() if k not in [ "service", "target_device" ] }
                except Exception:
                    try:
                        service = line.split("(")[0]
                        entity = line.split("(")[1][:-1]
                        domain, service = tuple(service.split("."))
                        extra_arguments = {}
                    except Exception:
                        to_say += f" Failed to parse call from '{line}'!"
                        continue

                # fix certain arguments
                # make sure brightness is 0-255 and not a percentage
                if "brightness" in extra_arguments and 0.0 < extra_arguments["brightness"] <= 1.0:
                    extra_arguments["brightness"] = int(extra_arguments["brightness"] * 255)

                # convert string "tuple" to a list for RGB colors
                if "rgb_color" in extra_arguments and isinstance(extra_arguments["rgb_color"], str):
                    extra_arguments["rgb_color"] = [ int(x) for x in extra_arguments["rgb_color"][1:-1].split(",") ]

                # only acknowledge requests to exposed entities
                if entity not in exposed_entities:
                    to_say += f" Can't find device '{entity}'!"
                else:
                    # copy arguments to service call
                    service_data = {ATTR_ENTITY_ID: entity}
                    for attr in allowed_service_call_arguments:
                        if attr in extra_arguments.keys():
                            service_data[attr] = extra_arguments[attr]
                    
                    try:
                        _LOGGER.debug(f"service data: {service_data}")
                        await self.hass.services.async_call(
                            domain,
                            service,
                            service_data=service_data,
                            blocking=True,
                        )
                    except Exception as err:
                        to_say += f"\nFailed to run: {line}"
                        _LOGGER.exception(f"Failed to run: {line}")

        if template_desc["assistant"]["suffix"]:
            to_say = to_say.replace(template_desc["assistant"]["suffix"], "") # remove the eos token if it is returned (some backends + the old model does this)
        
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(to_say)
        return ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _async_get_exposed_entities(self) -> tuple[dict[str, str], list[str]]:
        """Gather exposed entity states"""
        entity_states = {}
        domains = set()
        entity_registry = er.async_get(self.hass)

        for state in self.hass.states.async_all():
            if not async_should_expose(self.hass, CONVERSATION_DOMAIN, state.entity_id):
                continue

            entity = entity_registry.async_get(state.entity_id)

            attributes = dict(state.attributes)
            attributes["state"] = state.state
            if entity and entity.aliases:
                attributes["aliases"] = entity.aliases
            entity_states[state.entity_id] = attributes
            domains.add(state.domain)

        _LOGGER.debug(f"Exposed entities: {entity_states}")

        return entity_states, list(domains)

    def _format_prompt(
        self, prompt: list[dict], include_generation_prompt: bool = True
    ) -> str:
        formatted_prompt = ""

        prompt_template = self.entry.options.get(CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE)
        template_desc = PROMPT_TEMPLATE_DESCRIPTIONS[prompt_template]

        # handle models without a system prompt
        if prompt[0]["role"] == "system" and "system" not in template_desc:
            system_prompt = prompt.pop(0)
            prompt[0]["message"] = system_prompt["message"] + prompt[0]["message"]

        for message in prompt:
            role = message["role"]
            message = message["message"]
            role_desc = template_desc[role]
            formatted_prompt = (
                formatted_prompt + f"{role_desc['prefix']}{message}{role_desc['suffix']}\n"
            )

        if include_generation_prompt:
            formatted_prompt = formatted_prompt + template_desc["generation_prompt"]

        _LOGGER.debug(formatted_prompt)
        return formatted_prompt

    def _generate_system_prompt(self, prompt_template: str) -> str:
        """Generate a prompt for the user."""
        entities_to_expose, domains = self._async_get_exposed_entities()

        extra_attributes_to_expose = self.entry.options \
            .get(CONF_EXTRA_ATTRIBUTES_TO_EXPOSE, DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE)
        allowed_service_call_arguments = self.entry.options \
            .get(CONF_ALLOWED_SERVICE_CALL_ARGUMENTS, DEFAULT_ALLOWED_SERVICE_CALL_ARGUMENTS)
        
        def icl_example_generator(num_examples, entity_names, service_names):
            entity_domains = set([x.split(".")[0] for x in entity_names])
            entity_names = entity_names[:]
            
            # filter out examples for disabled services
            selected_in_context_examples = []
            for x in self.in_context_examples:
                if x["service"] in service_names and x["service"].split(".")[0] in entity_domains:
                    selected_in_context_examples.append(x)

            random.shuffle(selected_in_context_examples)
            random.shuffle(entity_names)
            
            for x in range(num_examples):
                chosen_example = selected_in_context_examples.pop()
                chosen_service = chosen_example["service"]
                device = [ x for x in entity_names if x.split(".")[0] == chosen_service.split(".")[0] ][0]
                example = {
                    "to_say": chosen_example["response"],
                    "service": chosen_service,
                    "target_device": device,
                }
                yield json.dumps(example) + "\n"

        def expose_attributes(attributes):
            result = attributes["state"]
            for attribute_name in extra_attributes_to_expose:
                if attribute_name not in attributes:
                    continue

                _LOGGER.debug(f"{attribute_name} = {attributes[attribute_name]}")

                value = attributes[attribute_name]
                if value is not None:
                    if attribute_name == "temperature":
                        value = int(value)
                        if value > 50:
                            value = f"{value}F"
                        else:
                            value = f"{value}C"
                    elif attribute_name == "rgb_color":
                        value = F"{closest_color(value)} {value}"
                    elif attribute_name == "volume_level":
                        value = f"vol={int(value*100)}"
                    elif attribute_name == "brightness":
                        value = f"{int(value/255*100)}%"
                    elif attribute_name == "humidity":
                        value = f"{value}%"

                    result = result + ";" + str(value)
            return result

        device_states = [f"{name} '{attributes.get('friendly_name')}' = {expose_attributes(attributes)}" for name, attributes in entities_to_expose.items()]

        # expose devices as their alias as well
        for name, attributes in entities_to_expose.items():
            if "aliases" in attributes:
                for alias in attributes["aliases"]:
                    device_states.append(f"{name} '{alias}' = {expose_attributes(attributes)}")

        formatted_states = "\n".join(device_states) + "\n"

        service_dict = self.hass.services.async_services()
        all_services = []
        all_service_names = []
        for domain in domains:
            for name, service in service_dict.get(domain, {}).items():
                args = flatten_vol_schema(service.schema)
                args_to_expose = set(args).intersection(allowed_service_call_arguments)
                all_services.append(f"{domain}.{name}({','.join(args_to_expose)})")
                all_service_names.append(f"{domain}.{name}")
        formatted_services = ", ".join(all_services)

        render_variables = {
            "devices": formatted_states,
            "services": formatted_services,
        }

        if self.in_context_examples:
            # TODO: make number of examples configurable
            render_variables["response_examples"] = "\n".join(icl_example_generator(4, list(entities_to_expose.keys()), all_service_names))
        
        return template.Template(prompt_template, self.hass).async_render(
            render_variables,
            parse_result=False,
        )

class LocalLLaMAAgent(LLaMAAgent):
    model_path: str
    llm: Any
    grammar: Any
    llama_cpp_module: Any
    remove_prompt_caching_listener: Callable
    model_lock: threading.Lock
    last_cache_prime: float
    # last_updated_entities: list[str]

    def _load_model(self, entry: ConfigEntry) -> None:
        self.model_path = entry.data.get(CONF_DOWNLOADED_MODEL_FILE)

        _LOGGER.info(
            "Using model file '%s'", self.model_path
        )

        if not self.model_path:
            raise Exception(f"Model was not found at '{self.model_path}'!")

        # don't import it until now because the wheel is installed by config_flow.py
        try:
            self.llama_cpp_module = importlib.import_module("llama_cpp")
        except ModuleNotFoundError:
            # attempt to re-install llama-cpp-python if it was uninstalled for some reason
            install_result = install_llama_cpp_python(self.hass.config.config_dir)
            if not install_result == True:
                raise ConfigEntryError("llama-cpp-python was not installed on startup and re-installing it led to an error!")
            
            self.llama_cpp_module = importlib.import_module("llama_cpp")
            
        Llama = getattr(self.llama_cpp_module, "Llama")

        _LOGGER.debug("Loading model...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_batch=2048,
            # TODO: expose arguments to the user in home assistant UI
            # n_threads=16,
            # n_threads_batch=4,
        )

        self.grammar = None
        if entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            self._load_grammar(entry.options.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE))

        # TODO: check about disk caching
        # self.llm.set_cache(self.llama_cpp_module.LlamaDiskCache(
        #     capacity_bytes=(512 * 10e8),
        #     cache_dir=os.path.join(self.hass.config.media_dirs.get("local", self.hass.config.path("media")), "kv_cache")
        # ))
        
        self.remove_prompt_caching_listener = None
        self.last_cache_prime = None
        self.model_lock = threading.Lock()

        if entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED):
            self._set_prompt_caching(enabled=True)

    def _load_grammar(self, filename: str):
        LlamaGrammar = getattr(self.llama_cpp_module, "LlamaGrammar")
        _LOGGER.debug("Loading grammar...")
        try:
            with open(os.path.join(os.path.dirname(__file__), filename)) as f:
                grammar_str = "".join(f.readlines())
            self.grammar = LlamaGrammar.from_string(grammar_str)
            _LOGGER.debug("Loaded grammar")
        except Exception:
            _LOGGER.exception("Failed to load grammar!")
            self.grammar = None

    def _update_options(self):
        LLaMAAgent._update_options(self)
        if self.entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            self._load_grammar(self.entry.options.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE))
        else:
            self.grammar = None

        self._set_prompt_caching(enabled=self.entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED))

    @callback
    async def _async_cache_prompt(self, entity, old_state, new_state):
        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)

        # if self.last_updated_entities:
        #     self.last_updated_entities

        refresh_start = time.time()

        # TODO: track which entities are updated most often and sort them to the bottom of the system prompt
        prompt = self._format_prompt([
            { "role": "system", "message": self._generate_system_prompt(raw_prompt)},
            { "role": "user", "message": "" }
        ], include_generation_prompt=False)

        _LOGGER.debug(f"refreshing cached prompt because {entity} changed...")
        await self.hass.async_add_executor_job(self._prime_llamacpp_kv_cache, prompt)

        refresh_end = time.time()
        _LOGGER.debug(f"cache refresh took {(refresh_end - refresh_start):.2f} sec")

    def _set_prompt_caching(self, *, enabled=True):
        if enabled and not self.remove_prompt_caching_listener:
            _LOGGER.info("enabling prompt caching...")

            entity_ids = [ 
                state.entity_id for state in self.hass.states.async_all() \
                    if async_should_expose(self.hass, CONVERSATION_DOMAIN, state.entity_id) 
            ]

            self.remove_prompt_caching_listener = async_track_state_change(self.hass, entity_ids, self._async_cache_prompt)

        elif not enabled and self.remove_prompt_caching_listener:
            _LOGGER.info("disabling prompt caching...")
            self.remove_prompt_caching_listener()

    def _prime_llamacpp_kv_cache(self, prompt: str) -> None:
        current_time = time.time()
        
        fastest_prime_interval = self.entry.options.get(CONF_PROMPT_CACHING_INTERVAL, DEFAULT_PROMPT_CACHING_INTERVAL)
        if self.last_cache_prime and current_time - self.last_cache_prime < fastest_prime_interval:
            return
        
        with self.model_lock:
            input_tokens = self.llm.tokenize(
                prompt.encode(), add_bos=False
            )

            temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
            top_k = int(self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K))
            top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
            grammar = self.grammar if self.entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR) else None

            _LOGGER.debug(f"Options: {self.entry.options}")

            _LOGGER.debug(f"Processing {len(input_tokens)} input tokens...")

            # grab just one token. should prime the kv cache with the system prompt
            next(self.llm.generate(
                input_tokens,
                temp=temperature,
                top_k=top_k,
                top_p=top_p,
                grammar=grammar
            ))

        self.last_cache_prime = time.time()
    
    def _generate(self, conversation: dict) -> str:
        prompt = self._format_prompt(conversation)

        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_k = int(self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K))
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)

        _LOGGER.debug(f"Options: {self.entry.options}")

        with self.model_lock:
            input_tokens = self.llm.tokenize(
                prompt.encode(), add_bos=False
            )

            _LOGGER.debug(f"Processing {len(input_tokens)} input tokens...")
            output_tokens = self.llm.generate(
                input_tokens,
                temp=temperature,
                top_k=top_k,
                top_p=top_p,
                grammar=self.grammar
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
    
class GenericOpenAIAPIAgent(LLaMAAgent):
    api_host: str
    api_key: str
    model_name: str

    def _load_model(self, entry: ConfigEntry) -> None:
        self.api_host = f"{'https' if entry.data[CONF_SSL] else 'http'}://{entry.data[CONF_HOST]}:{entry.data[CONF_PORT]}"
        self.api_key = entry.data.get(CONF_OPENAI_API_KEY)
        self.model_name = entry.data.get(CONF_CHAT_MODEL)


    def _chat_completion_params(self, conversation: dict) -> (str, dict):
        request_params = {}

        endpoint = "/v1/chat/completions"
        request_params["messages"] = [ { "role": x["role"], "content": x["message"] } for x in conversation ]

        return endpoint, request_params

    def _completion_params(self, conversation: dict) -> (str, dict):
        request_params = {}

        endpoint = "/v1/completions"
        request_params["prompt"] = self._format_prompt(conversation)

        return endpoint, request_params
    
    def _extract_response(self, response_json: dict) -> str:
        choices = response_json["choices"]
        if choices[0]["finish_reason"] != "stop":
            _LOGGER.warn("Model response did not end on a stop token (unfinished sentence)")

        if response_json["object"] == "chat.completion":
            return choices[0]["message"]["content"]
        else:
            return choices[0]["text"]
    
    def _generate(self, conversation: dict) -> str:
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        timeout = self.entry.options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        use_chat_api = self.entry.options.get(CONF_REMOTE_USE_CHAT_ENDPOINT, DEFAULT_REMOTE_USE_CHAT_ENDPOINT)
        

        request_params = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if use_chat_api:
            endpoint, additional_params = self._chat_completion_params(conversation)
        else:
            endpoint, additional_params = self._completion_params(conversation)
        
        request_params.update(additional_params)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        result = requests.post(
            f"{self.api_host}{endpoint}", 
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

        _LOGGER.debug(result.json())

        return self._extract_response(result.json())
        
class TextGenerationWebuiAgent(GenericOpenAIAPIAgent):
    admin_key: str

    def _load_model(self, entry: ConfigEntry) -> None:
        super()._load_model(entry)
        self.admin_key = entry.data.get(CONF_TEXT_GEN_WEBUI_ADMIN_KEY, self.api_key)

        try:
            headers = {}
            if self.admin_key:
                headers["Authorization"] = f"Bearer {self.admin_key}"
                
            currently_loaded_result = requests.get(
                f"{self.api_host}/v1/internal/model/info",
                headers=headers,
            )
            currently_loaded_result.raise_for_status()

            loaded_model = currently_loaded_result.json()["model_name"]
            if loaded_model == self.model_name:
                _LOGGER.info(f"Model {self.model_name} is already loaded on the remote backend.")
                return
            else:
                _LOGGER.info(f"Model is not {self.model_name} loaded on the remote backend. Loading it now...")
            
            load_result = requests.post(
                f"{self.api_host}/v1/internal/model/load",
                json={
                    "model_name": self.model_name,
                    # TODO: expose arguments to the user in home assistant UI
                    # "args": {},
                },
                headers=headers
            )
            load_result.raise_for_status()

        except Exception as ex:
            _LOGGER.debug("Connection error was: %s", repr(ex))
            raise ConfigEntryNotReady("There was a problem connecting to the remote server") from ex

    def _chat_completion_params(self, conversation: dict) -> (str, dict):
        preset = self.entry.options.get(CONF_TEXT_GEN_WEBUI_PRESET)
        chat_mode = self.entry.options.get(CONF_TEXT_GEN_WEBUI_CHAT_MODE, DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE)

        endpoint, request_params = super()._chat_completion_params(conversation)

        request_params["mode"] = chat_mode
        if chat_mode == TEXT_GEN_WEBUI_CHAT_MODE_CHAT or chat_mode == TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT:
            if preset:
                request_params["character"] = preset
        elif chat_mode == TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT:
            request_params["instruction_template"] = self.entry.options.get(CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE)

        return endpoint, request_params
    
    def _completion_params(self, conversation: dict) -> (str, dict):
        preset = self.entry.options.get(CONF_TEXT_GEN_WEBUI_PRESET)

        endpoint, request_params = super()._completion_params(conversation)

        if preset:
            request_params["preset"] = preset

        return endpoint, request_params
    
    def _extract_response(self, response_json: dict) -> str:
        choices = response_json["choices"]
        if choices[0]["finish_reason"] != "stop":
            _LOGGER.warn("Model response did not end on a stop token (unfinished sentence)")

        # text-gen-webui has a typo where it is 'chat.completions' not 'chat.completion'
        if response_json["object"] == "chat.completions":
            return choices[0]["message"]["content"]
        else:
            return choices[0]["text"]
        
class LlamaCppPythonAPIAgent(GenericOpenAIAPIAgent):
    """https://llama-cpp-python.readthedocs.io/en/latest/server/"""
    grammar: str

    def _load_model(self, entry: ConfigEntry):
        super()._load_model(entry)

        with open(os.path.join(os.path.dirname(__file__), GBNF_GRAMMAR_FILE)) as f:
            self.grammar = "".join(f.readlines())

    def _chat_completion_params(self, conversation: dict) -> (str, dict):
        top_k = int(self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K))
        endpoint, request_params = super()._chat_completion_params(conversation)

        request_params["top_k"] = top_k

        if self.entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            request_params["grammar"] = self.grammar

        return endpoint, request_params
    
    def _completion_params(self, conversation: dict) -> (str, dict):
        top_k = int(self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K))
        endpoint, request_params = super()._completion_params(conversation)

        request_params["top_k"] = top_k
        
        if self.entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            request_params["grammar"] = self.grammar

        return endpoint, request_params

class OllamaAPIAgent(LLaMAAgent):
    api_host: str
    api_key: str
    model_name: str

    def _load_model(self, entry: ConfigEntry) -> None:
        self.api_host = f"{'https' if entry.data[CONF_SSL] else 'http'}://{entry.data[CONF_HOST]}:{entry.data[CONF_PORT]}"
        self.api_key = entry.data.get(CONF_OPENAI_API_KEY)
        self.model_name = entry.data.get(CONF_CHAT_MODEL)

        # ollama handles loading for us so just make sure the model is available
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            currently_downloaded_result = requests.get(
                f"{self.api_host}/api/tags",
                headers=headers,
            )
            currently_downloaded_result.raise_for_status()
                
        except Exception as ex:
            _LOGGER.debug("Connection error was: %s", repr(ex))
            raise ConfigEntryNotReady("There was a problem connecting to the remote server") from ex

        model_names = [ x["name"] for x in currently_downloaded_result.json()["models"] ]
        if ":" in self.model_name:
            if not any([ name == self.model_name for name in model_names]):
                raise ConfigEntryNotReady(f"Ollama server does not have the provided model: {self.model_name}")    
        elif not any([ name.split(":")[0] == self.model_name for name in model_names ]):
            raise ConfigEntryNotReady(f"Ollama server does not have the provided model: {self.model_name}")

    def _chat_completion_params(self, conversation: dict) -> (str, dict):
        request_params = {}

        endpoint = "/api/chat"
        request_params["messages"] = [ { "role": x["role"], "content": x["message"] } for x in conversation ]

        return endpoint, request_params

    def _completion_params(self, conversation: dict) -> (str, dict):
        request_params = {}

        endpoint = "/api/generate"
        request_params["prompt"] = self._format_prompt(conversation)

        return endpoint, request_params
    
    def _extract_response(self, response_json: dict) -> str:        
        if response_json["done"] != "true":
            _LOGGER.warn("Model response did not end on a stop token (unfinished sentence)")

        if "response" in response_json:
            return response_json["response"]
        else:
            return response_json["message"]["content"]
    
    def _generate(self, conversation: dict) -> str:
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        timeout = self.entry.options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        keep_alive = self.entry.options.get(CONF_OLLAMA_KEEP_ALIVE_MIN, DEFAULT_OLLAMA_KEEP_ALIVE_MIN)
        use_chat_api = self.entry.options.get(CONF_REMOTE_USE_CHAT_ENDPOINT, DEFAULT_REMOTE_USE_CHAT_ENDPOINT)
        

        request_params = {
            "model": self.model_name,
            "stream": False,
            "keep_alive": f"{keep_alive}m", # prevent ollama from unloading the model
            "options": {
                "top_p": top_p,
                "temperature": temperature,
                "num_predict": max_tokens,
            }   
        }
        
        if use_chat_api:
            endpoint, additional_params = self._chat_completion_params(conversation)
        else:
            endpoint, additional_params = self._completion_params(conversation)
        
        request_params.update(additional_params)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        result = requests.post(
            f"{self.api_host}{endpoint}", 
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
        
        _LOGGER.debug(result.json())

        return self._extract_response(result.json())