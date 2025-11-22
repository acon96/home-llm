"""Defines the llama cpp agent"""
from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from typing import Any, Callable, List, Generator, AsyncGenerator, Optional, cast

from homeassistant.components import conversation as conversation
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import callback, HomeAssistant
from homeassistant.exceptions import ConfigEntryError, HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.event import async_track_state_change, async_call_later

from custom_components.llama_conversation.utils import install_llama_cpp_python, validate_llama_cpp_python_installation, get_oai_formatted_messages, get_oai_formatted_tools
from custom_components.llama_conversation.const import (
    CONF_INSTALLED_LLAMACPP_VERSION,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_TYPICAL_P,
    CONF_MIN_P,
    CONF_DOWNLOADED_MODEL_FILE,
    CONF_LLAMACPP_ENABLE_FLASH_ATTENTION,
    CONF_USE_GBNF_GRAMMAR,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_PROMPT_CACHING_ENABLED,
    CONF_PROMPT_CACHING_INTERVAL,
    CONF_CONTEXT_LENGTH,
    CONF_LLAMACPP_BATCH_SIZE,
    CONF_LLAMACPP_THREAD_COUNT,
    CONF_LLAMACPP_BATCH_THREAD_COUNT,
    CONF_INSTALLED_LLAMACPP_VERSION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_MIN_P,
    DEFAULT_TYPICAL_P,
    DEFAULT_LLAMACPP_ENABLE_FLASH_ATTENTION,
    DEFAULT_USE_GBNF_GRAMMAR,
    DEFAULT_GBNF_GRAMMAR_FILE,
    DEFAULT_PROMPT_CACHING_ENABLED,
    DEFAULT_PROMPT_CACHING_INTERVAL,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_LLAMACPP_BATCH_SIZE,
    DEFAULT_LLAMACPP_THREAD_COUNT,
    DEFAULT_LLAMACPP_BATCH_THREAD_COUNT,
    DOMAIN,
)
from custom_components.llama_conversation.entity import LocalLLMClient, TextGenerationResult

# make type checking work for llama-cpp-python without importing it directly at runtime
from typing import TYPE_CHECKING
from types import ModuleType
if TYPE_CHECKING:
    from llama_cpp import Llama as LlamaType, LlamaGrammar as LlamaGrammarType
else:
    LlamaType = Any
    LlamaGrammarType = Any

_LOGGER = logging.getLogger(__name__)

def snapshot_settings(options: dict[str, Any]) -> dict[str, Any]:
    return {
        CONF_DOWNLOADED_MODEL_FILE: options.get(CONF_DOWNLOADED_MODEL_FILE, ""),
        CONF_CONTEXT_LENGTH: options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH),
        CONF_LLAMACPP_BATCH_SIZE: options.get(CONF_LLAMACPP_BATCH_SIZE, DEFAULT_LLAMACPP_BATCH_SIZE),
        CONF_LLAMACPP_THREAD_COUNT: options.get(CONF_LLAMACPP_THREAD_COUNT, DEFAULT_LLAMACPP_THREAD_COUNT),
        CONF_LLAMACPP_BATCH_THREAD_COUNT: options.get(CONF_LLAMACPP_BATCH_THREAD_COUNT, DEFAULT_LLAMACPP_BATCH_THREAD_COUNT),
        CONF_LLAMACPP_ENABLE_FLASH_ATTENTION: options.get(CONF_LLAMACPP_ENABLE_FLASH_ATTENTION, DEFAULT_LLAMACPP_ENABLE_FLASH_ATTENTION),
        CONF_INSTALLED_LLAMACPP_VERSION: options.get(CONF_INSTALLED_LLAMACPP_VERSION, ""),
        CONF_GBNF_GRAMMAR_FILE: options.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE),
        CONF_PROMPT_CACHING_ENABLED: options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED),
    }


class LlamaCppClient(LocalLLMClient):
    llama_cpp_module: ModuleType | None

    models: dict[str, LlamaType]
    grammars: dict[str, Any]
    loaded_model_settings: dict[str, dict[str, Any]]

    # caching properties
    model_lock: threading.Lock
    remove_prompt_caching_listener: Optional[Callable]
    last_cache_prime: float
    last_updated_entities: dict[str, float]
    cache_refresh_after_cooldown: bool

    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]) -> None:
        super().__init__(hass, client_options)

        self.llama_cpp_module = None
        self.models = {}
        self.grammars = {}
        self.loaded_model_settings = {}

        self.remove_prompt_caching_listener = None
        self.last_cache_prime = 0.0 
        self.last_updated_entities = {}
        self.cache_refresh_after_cooldown = False
        self.model_lock = threading.Lock()

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        return "Llama.cpp"

    async def async_get_available_models(self) -> List[str]:
        return [] # TODO: find available "huggingface_hub" models that have been downloaded

    def _load_model(self, entity_options: dict[str, Any]) -> None:
        model_name = entity_options.get(CONF_CHAT_MODEL, "")
        model_path = entity_options.get(CONF_DOWNLOADED_MODEL_FILE, "")

        if model_name in self.models:
            _LOGGER.info("Model %s is already loaded", model_name)
            return

        _LOGGER.info("Using model file '%s'", model_path)

        if not model_path or not os.path.isfile(model_path):
            raise Exception(f"Model was not found at '{model_path}'!")

        if not self.llama_cpp_module:
            validate_llama_cpp_python_installation()

            # don't import it until now because the wheel is installed by config_flow.py
            try:
                self.llama_cpp_module = importlib.import_module("llama_cpp")
            except ModuleNotFoundError:
                # attempt to re-install llama-cpp-python if it was uninstalled for some reason
                install_result = install_llama_cpp_python(self.hass.config.config_dir)
                if not install_result == True:
                    raise ConfigEntryError("llama-cpp-python was not installed on startup and re-installing it led to an error!")

                validate_llama_cpp_python_installation()
                self.llama_cpp_module = importlib.import_module("llama_cpp")

        Llama: type[LlamaType] = getattr(self.llama_cpp_module, "Llama")

        _LOGGER.debug(f"Loading model '{model_path}'...")
        model_settings = snapshot_settings(entity_options)

        llm = Llama(
            model_path=model_path,
            n_ctx=int(model_settings[CONF_CONTEXT_LENGTH]),
            n_batch=int(model_settings[CONF_LLAMACPP_BATCH_SIZE]),
            n_threads=int(model_settings[CONF_LLAMACPP_THREAD_COUNT]),
            n_threads_batch=int(model_settings[CONF_LLAMACPP_BATCH_THREAD_COUNT]),
            flash_attn=model_settings[CONF_LLAMACPP_ENABLE_FLASH_ATTENTION],
        )
        _LOGGER.debug("Model loaded")

        # TODO: check about disk caching
        # self.llm.set_cache(self.llama_cpp_module.LlamaDiskCache(
        #     capacity_bytes=(512 * 10e8),
        #     cache_dir=os.path.join(self.hass.config.media_dirs.get("local", self.hass.config.path("media")), "kv_cache")
        # ))

        if model_settings[CONF_PROMPT_CACHING_ENABLED]:
            @callback
            async def enable_caching_after_startup(_now) -> None:
                self._set_prompt_caching(entity_options, enabled=True)
                await self._async_cache_prompt(None, None, None, entity_options)
            async_call_later(self.hass, 5.0, enable_caching_after_startup)

        self.loaded_model_settings[model_name] = model_settings
        self.models[model_name] = llm

        self.grammars[model_name] = None
        if entity_options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            self._load_grammar(model_name, entity_options.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE))


    def _load_grammar(self, model_name: str, filename: str) -> Any:
        LlamaGrammar: type[LlamaGrammarType] = getattr(self.llama_cpp_module, "LlamaGrammar")
        _LOGGER.debug(f"Loading grammar {filename}...")
        try:
            with open(os.path.join(os.path.dirname(__file__), filename)) as f:
                grammar_str = "".join(f.readlines())
            self.grammars[model_name] = LlamaGrammar.from_string(grammar_str)
            self.loaded_model_settings[model_name][CONF_GBNF_GRAMMAR_FILE] = filename
            _LOGGER.debug("Loaded grammar")
        except Exception:
            _LOGGER.exception("Failed to load grammar!")
            self.grammars[model_name] = None

    def _update_options(self, entity_options: dict[str, Any]):
        LocalLLMClient._update_options(self, entity_options)

        loaded_options = self.loaded_model_settings.get(entity_options.get(CONF_CHAT_MODEL, ""), None)

        should_reload = False
        if not loaded_options:
            should_reload = True
        elif loaded_options[CONF_CONTEXT_LENGTH] != entity_options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH):
            should_reload = True
        elif loaded_options[CONF_LLAMACPP_BATCH_SIZE] != entity_options.get(CONF_LLAMACPP_BATCH_SIZE, DEFAULT_LLAMACPP_BATCH_SIZE):
            should_reload = True
        elif loaded_options[CONF_LLAMACPP_THREAD_COUNT] != entity_options.get(CONF_LLAMACPP_THREAD_COUNT, DEFAULT_LLAMACPP_THREAD_COUNT):
            should_reload = True
        elif loaded_options[CONF_LLAMACPP_BATCH_THREAD_COUNT] != entity_options.get(CONF_LLAMACPP_BATCH_THREAD_COUNT, DEFAULT_LLAMACPP_BATCH_THREAD_COUNT):
            should_reload = True
        elif loaded_options[CONF_LLAMACPP_ENABLE_FLASH_ATTENTION] != entity_options.get(CONF_LLAMACPP_ENABLE_FLASH_ATTENTION, DEFAULT_LLAMACPP_ENABLE_FLASH_ATTENTION):
            should_reload = True
        elif loaded_options[CONF_INSTALLED_LLAMACPP_VERSION] != entity_options.get(CONF_INSTALLED_LLAMACPP_VERSION):
            should_reload = True
            _LOGGER.debug(f"Reloading llama.cpp...")
            if self.llama_cpp_module:
                self.llama_cpp_module = importlib.reload(self.llama_cpp_module)

        model_path = entity_options.get(CONF_DOWNLOADED_MODEL_FILE, "")
        model_name = entity_options.get(CONF_CHAT_MODEL, "")

        if should_reload:

            _LOGGER.debug(f"Reloading model '{model_path}'...")
            model_settings = snapshot_settings(entity_options)

            Llama: type[LlamaType] = getattr(self.llama_cpp_module, "Llama")
            llm = Llama(
                model_path=model_path,
                n_ctx=int(model_settings[CONF_CONTEXT_LENGTH]),
                n_batch=int(model_settings[CONF_LLAMACPP_BATCH_SIZE]),
                n_threads=int(model_settings[CONF_LLAMACPP_THREAD_COUNT]),
                n_threads_batch=int(model_settings[CONF_LLAMACPP_BATCH_THREAD_COUNT]),
                flash_attn=model_settings[CONF_LLAMACPP_ENABLE_FLASH_ATTENTION],
            )
            _LOGGER.debug("Model loaded")

            self.loaded_model_settings[model_name] = model_settings
            self.models[model_name] = llm

        if entity_options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            current_grammar = entity_options.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE)
            if model_name not in self.grammars or self.loaded_model_settings[CONF_GBNF_GRAMMAR_FILE] != current_grammar:
                self._load_grammar(model_name, current_grammar)
        else:
            self.grammars[model_name] = None

        if entity_options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED):
            self._set_prompt_caching(entity_options, enabled=True)

            if self.loaded_model_settings[CONF_PROMPT_CACHING_ENABLED] != entity_options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED) or \
                should_reload:
                self.loaded_model_settings[CONF_PROMPT_CACHING_ENABLED] = entity_options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED)

                async def cache_current_prompt(_now):
                    await self._async_cache_prompt(None, None, None, entity_options)
                async_call_later(self.hass, 1.0, cache_current_prompt)
        else:
            self._set_prompt_caching(entity_options, enabled=False)

    def _async_get_exposed_entities(self) -> dict[str, dict[str, str]]:
        """Takes the super class function results and sorts the entities with the recently updated at the end"""
        entities = LocalLLMClient._async_get_exposed_entities(self)

        entity_order: dict[str, Optional[float]] = { name: None for name in entities.keys() }
        entity_order.update(self.last_updated_entities)

        def sort_key(item):
            item_name, last_updated = item
            # Handle cases where last updated is None
            if last_updated is None:
                return (False, '', item_name)
            else:
                return (True, last_updated, '')

        # Sort the items based on the sort_key function
        sorted_items = sorted(list(entity_order.items()), key=sort_key)

        _LOGGER.debug(f"sorted_items: {sorted_items}")

        sorted_entities: dict[str, dict[str, str]] = {}
        for item_name, _ in sorted_items:
            sorted_entities[item_name] = entities[item_name]

        return sorted_entities

    def _set_prompt_caching(self, entity_options: dict[str, Any], *, enabled=True):
        if enabled and not self.remove_prompt_caching_listener:
            _LOGGER.info("enabling prompt caching...")

            entity_ids = [
                state.entity_id for state in self.hass.states.async_all() \
                    if async_should_expose(self.hass, CONVERSATION_DOMAIN, state.entity_id)
            ]

            _LOGGER.debug(f"watching entities: {entity_ids}")

            self.remove_prompt_caching_listener = async_track_state_change(self.hass, entity_ids, lambda x, y, z: self._async_cache_prompt(x, y, z, entity_options))

        elif not enabled and self.remove_prompt_caching_listener:
            _LOGGER.info("disabling prompt caching...")
            self.remove_prompt_caching_listener()

    @callback
    async def _async_cache_prompt(self, entity, old_state, new_state, entity_options: dict[str, Any]):
        refresh_start = time.time()

        # track last update time so we can sort the context efficiently
        if entity:
            self.last_updated_entities[entity] = refresh_start

        llm_api: llm.APIInstance | None = None
        if entity_options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    self.hass, entity_options[CONF_LLM_HASS_API],
                    llm_context=llm.LLMContext(DOMAIN, context=None, language=None, assistant=None, device_id=None)
                )
            except HomeAssistantError:
                _LOGGER.exception("Failed to get LLM API when caching prompt!")
                return

        _LOGGER.debug(f"refreshing cached prompt because {entity} changed...")
        await self.hass.async_add_executor_job(self._cache_prompt, llm_api, entity_options)

        refresh_end = time.time()
        _LOGGER.debug(f"cache refresh took {(refresh_end - refresh_start):.2f} sec")

    def _cache_prompt(self, llm_api: llm.APIInstance | None, entity_options: dict[str, Any]) -> None:
        # if a refresh is already scheduled then exit
        if self.cache_refresh_after_cooldown:
            return

        # if we are inside the cooldown period, request a refresh and exit
        current_time = time.time()
        fastest_prime_interval = entity_options.get(CONF_PROMPT_CACHING_INTERVAL, DEFAULT_PROMPT_CACHING_INTERVAL)
        if self.last_cache_prime and current_time - self.last_cache_prime < fastest_prime_interval:
            self.cache_refresh_after_cooldown = True
            return

        # try to acquire the lock, if we are still running for some reason, request a refresh and exit
        lock_acquired = self.model_lock.acquire(False)
        if not lock_acquired:
            self.cache_refresh_after_cooldown = True
            return

        try:
            # Build system/user messages and use the chat-completion API to prime
            # the model. We request only a single token (max_tokens=1) and
            # discard the result. This avoids implementing any streaming logic
            # while still priming the KV cache with the system prompt.
            raw_prompt = entity_options.get(CONF_PROMPT, DEFAULT_PROMPT)
            system_prompt = self._generate_system_prompt(raw_prompt, llm_api, entity_options)

            messages = get_oai_formatted_messages([
                conversation.SystemContent(content=system_prompt),
                conversation.UserContent(content="")
            ])
            tools = None
            if llm_api:
                tools = get_oai_formatted_tools(llm_api, self._async_get_all_exposed_domains())

            model_name = entity_options.get(CONF_CHAT_MODEL, "")
            temperature = entity_options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
            top_k = int(entity_options.get(CONF_TOP_K, DEFAULT_TOP_K))
            top_p = entity_options.get(CONF_TOP_P, DEFAULT_TOP_P)
            min_p = entity_options.get(CONF_MIN_P, DEFAULT_MIN_P)
            typical_p = entity_options.get(CONF_TYPICAL_P, DEFAULT_TYPICAL_P)
            grammar = self.grammars.get(model_name) if entity_options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR) else None

            _LOGGER.debug("Priming model cache via chat completion API...")
            try:
                
                # avoid strict typing issues from the llama-cpp-python bindings
                self.models[model_name].create_chat_completion(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    typical_p=typical_p,
                    max_tokens=1,
                    grammar=grammar,
                    stream=False,
                )

                self.last_cache_prime = time.time()
            except Exception:
                _LOGGER.exception("Failed to prime model cache")
        finally:
            self.model_lock.release()

        # schedule a refresh using async_call_later
        # if the flag is set after the delay then we do another refresh

        @callback
        async def refresh_if_requested(_now):
            if self.cache_refresh_after_cooldown:
                self.cache_refresh_after_cooldown = False

                refresh_start = time.time()
                _LOGGER.debug(f"refreshing cached prompt after cooldown...")
                await self.hass.async_add_executor_job(self._cache_prompt, llm_api, entity_options)

                refresh_end = time.time()
                _LOGGER.debug(f"cache refresh took {(refresh_end - refresh_start):.2f} sec")

        refresh_delay = entity_options.get(CONF_PROMPT_CACHING_INTERVAL, DEFAULT_PROMPT_CACHING_INTERVAL)
        async_call_later(self.hass, float(refresh_delay), refresh_if_requested)

    def _generate_stream(self, 
                         conversation: List[conversation.Content],
                         llm_api: llm.APIInstance | None,
                         agent_id: str,
                         entity_options: dict[str, Any],
                        ) -> AsyncGenerator[TextGenerationResult, None]:
        """Async generator that yields TextGenerationResult as tokens are produced."""
        model_name = entity_options.get(CONF_CHAT_MODEL, "")
        max_tokens = entity_options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = entity_options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_k = int(entity_options.get(CONF_TOP_K, DEFAULT_TOP_K))
        top_p = entity_options.get(CONF_TOP_P, DEFAULT_TOP_P)
        min_p = entity_options.get(CONF_MIN_P, DEFAULT_MIN_P)
        typical_p = entity_options.get(CONF_TYPICAL_P, DEFAULT_TYPICAL_P)
        grammar = self.grammars.get(model_name) if entity_options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR) else None

        _LOGGER.debug(f"Options: {entity_options}")

        messages = get_oai_formatted_messages(conversation, user_content_as_list=True)
        tools = None
        if llm_api:
            tools = get_oai_formatted_tools(llm_api, self._async_get_all_exposed_domains())

        _LOGGER.debug(f"Generating completion with {len(messages)} messages and {len(tools) if tools else 0} tools...")

        chat_completion = self.models[model_name].create_chat_completion(
            messages,
            tools=tools,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            max_tokens=max_tokens,
            grammar=grammar,
            stream=True,
        )

        def next_token() -> Generator[tuple[Optional[str], Optional[List]]]:
            """Get the next token from the chat completion iterator."""
            for chunk in chat_completion:
                if isinstance(chunk, str):
                    yield chunk, []
                else:
                    content = chunk["choices"][0]["delta"].get("content")
                    tool_calls = chunk["choices"][0]["delta"].get("tool_calls")
                    yield content, tool_calls

        return self._async_parse_completion(llm_api, agent_id, entity_options, next_token=next_token())

