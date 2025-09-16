"""Defines the llama cpp agent"""
from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from typing import Any, Callable, List, Generator, AsyncGenerator, Optional

from homeassistant.components import conversation as conversation
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import callback
from homeassistant.exceptions import ConfigEntryError, HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.event import async_track_state_change, async_call_later

from custom_components.llama_conversation.utils import install_llama_cpp_python, validate_llama_cpp_python_installation, get_oai_formatted_messages, get_oai_formatted_tools, parse_raw_tool_call
from custom_components.llama_conversation.const import (
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_TYPICAL_P,
    CONF_MIN_P,
    CONF_DOWNLOADED_MODEL_FILE,
    CONF_ENABLE_FLASH_ATTENTION,
    CONF_USE_GBNF_GRAMMAR,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_PROMPT_CACHING_ENABLED,
    CONF_PROMPT_CACHING_INTERVAL,
    CONF_CONTEXT_LENGTH,
    CONF_BATCH_SIZE,
    CONF_THREAD_COUNT,
    CONF_BATCH_THREAD_COUNT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_MIN_P,
    DEFAULT_TYPICAL_P,
    DEFAULT_ENABLE_FLASH_ATTENTION,
    DEFAULT_USE_GBNF_GRAMMAR,
    DEFAULT_GBNF_GRAMMAR_FILE,
    DEFAULT_PROMPT_CACHING_ENABLED,
    DEFAULT_PROMPT_CACHING_INTERVAL,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_THREAD_COUNT,
    DEFAULT_BATCH_THREAD_COUNT,
)
from custom_components.llama_conversation.conversation import LocalLLMAgent, TextGenerationResult

# make type checking work for llama-cpp-python without importing it directly at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llama_cpp import Llama as LlamaType
else:
    LlamaType = Any

_LOGGER = logging.getLogger(__name__)


class LlamaCppAgent(LocalLLMAgent):
    model_path: str
    llm: LlamaType
    grammar: Any
    llama_cpp_module: Any
    remove_prompt_caching_listener: Callable
    model_lock: threading.Lock
    last_cache_prime: float
    last_updated_entities: dict[str, float]
    cache_refresh_after_cooldown: bool
    loaded_model_settings: dict[str, Any]

    _attr_supports_streaming = True

    def _load_model(self, entry: ConfigEntry) -> None:
        self.model_path = entry.data.get(CONF_DOWNLOADED_MODEL_FILE)

        _LOGGER.info(
            "Using model file '%s'", self.model_path
        )

        if not self.model_path:
            raise Exception(f"Model was not found at '{self.model_path}'!")

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

        Llama = getattr(self.llama_cpp_module, "Llama")

        _LOGGER.debug(f"Loading model '{self.model_path}'...")
        self.loaded_model_settings = {}
        self.loaded_model_settings[CONF_CONTEXT_LENGTH] = entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        self.loaded_model_settings[CONF_BATCH_SIZE] = entry.options.get(CONF_BATCH_SIZE, DEFAULT_BATCH_SIZE)
        self.loaded_model_settings[CONF_THREAD_COUNT] = entry.options.get(CONF_THREAD_COUNT, DEFAULT_THREAD_COUNT)
        self.loaded_model_settings[CONF_BATCH_THREAD_COUNT] = entry.options.get(CONF_BATCH_THREAD_COUNT, DEFAULT_BATCH_THREAD_COUNT)
        self.loaded_model_settings[CONF_ENABLE_FLASH_ATTENTION] = entry.options.get(CONF_ENABLE_FLASH_ATTENTION, DEFAULT_ENABLE_FLASH_ATTENTION)

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=int(self.loaded_model_settings[CONF_CONTEXT_LENGTH]),
            n_batch=int(self.loaded_model_settings[CONF_BATCH_SIZE]),
            n_threads=int(self.loaded_model_settings[CONF_THREAD_COUNT]),
            n_threads_batch=int(self.loaded_model_settings[CONF_BATCH_THREAD_COUNT]),
            flash_attn=self.loaded_model_settings[CONF_ENABLE_FLASH_ATTENTION],
        )
        _LOGGER.debug("Model loaded")

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
        self.last_updated_entities = {}
        self.cache_refresh_after_cooldown = False
        self.model_lock = threading.Lock()

        self.loaded_model_settings[CONF_PROMPT_CACHING_ENABLED] = entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED)
        if self.loaded_model_settings[CONF_PROMPT_CACHING_ENABLED]:
            @callback
            async def enable_caching_after_startup(_now) -> None:
                self._set_prompt_caching(enabled=True)
                await self._async_cache_prompt(None, None, None)
            async_call_later(self.hass, 5.0, enable_caching_after_startup)

    def _load_grammar(self, filename: str):
        LlamaGrammar = getattr(self.llama_cpp_module, "LlamaGrammar")
        _LOGGER.debug(f"Loading grammar {filename}...")
        try:
            with open(os.path.join(os.path.dirname(__file__), filename)) as f:
                grammar_str = "".join(f.readlines())
            self.grammar = LlamaGrammar.from_string(grammar_str)
            self.loaded_model_settings[CONF_GBNF_GRAMMAR_FILE] = filename
            _LOGGER.debug("Loaded grammar")
        except Exception:
            _LOGGER.exception("Failed to load grammar!")
            self.grammar = None

    def _update_options(self):
        LocalLLMAgent._update_options(self)

        model_reloaded = False
        if self.loaded_model_settings[CONF_CONTEXT_LENGTH] != self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH) or \
            self.loaded_model_settings[CONF_BATCH_SIZE] != self.entry.options.get(CONF_BATCH_SIZE, DEFAULT_BATCH_SIZE) or \
            self.loaded_model_settings[CONF_THREAD_COUNT] != self.entry.options.get(CONF_THREAD_COUNT, DEFAULT_THREAD_COUNT) or \
            self.loaded_model_settings[CONF_BATCH_THREAD_COUNT] != self.entry.options.get(CONF_BATCH_THREAD_COUNT, DEFAULT_BATCH_THREAD_COUNT) or \
            self.loaded_model_settings[CONF_ENABLE_FLASH_ATTENTION] != self.entry.options.get(CONF_ENABLE_FLASH_ATTENTION, DEFAULT_ENABLE_FLASH_ATTENTION):

            _LOGGER.debug(f"Reloading model '{self.model_path}'...")
            self.loaded_model_settings[CONF_CONTEXT_LENGTH] = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
            self.loaded_model_settings[CONF_BATCH_SIZE] = self.entry.options.get(CONF_BATCH_SIZE, DEFAULT_BATCH_SIZE)
            self.loaded_model_settings[CONF_THREAD_COUNT] = self.entry.options.get(CONF_THREAD_COUNT, DEFAULT_THREAD_COUNT)
            self.loaded_model_settings[CONF_BATCH_THREAD_COUNT] = self.entry.options.get(CONF_BATCH_THREAD_COUNT, DEFAULT_BATCH_THREAD_COUNT)
            self.loaded_model_settings[CONF_ENABLE_FLASH_ATTENTION] = self.entry.options.get(CONF_ENABLE_FLASH_ATTENTION, DEFAULT_ENABLE_FLASH_ATTENTION)

            Llama = getattr(self.llama_cpp_module, "Llama")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=int(self.loaded_model_settings[CONF_CONTEXT_LENGTH]),
                n_batch=int(self.loaded_model_settings[CONF_BATCH_SIZE]),
                n_threads=int(self.loaded_model_settings[CONF_THREAD_COUNT]),
                n_threads_batch=int(self.loaded_model_settings[CONF_BATCH_THREAD_COUNT]),
                flash_attn=self.loaded_model_settings[CONF_ENABLE_FLASH_ATTENTION],
            )
            _LOGGER.debug("Model loaded")
            model_reloaded = True

        if self.entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR):
            current_grammar = self.entry.options.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE)
            if not self.grammar or self.loaded_model_settings[CONF_GBNF_GRAMMAR_FILE] != current_grammar:
                self._load_grammar(current_grammar)
        else:
            self.grammar = None

        if self.entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED):
            self._set_prompt_caching(enabled=True)

            if self.loaded_model_settings[CONF_PROMPT_CACHING_ENABLED] != self.entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED) or \
                model_reloaded:
                self.loaded_model_settings[CONF_PROMPT_CACHING_ENABLED] = self.entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED)

                async def cache_current_prompt(_now):
                    await self._async_cache_prompt(None, None, None)
                async_call_later(self.hass, 1.0, cache_current_prompt)
        else:
            self._set_prompt_caching(enabled=False)

    def _async_get_exposed_entities(self) -> dict[str, str]:
        """Takes the super class function results and sorts the entities with the recently updated at the end"""
        entities = LocalLLMAgent._async_get_exposed_entities(self)

        # ignore sorting if prompt caching is disabled
        if not self.entry.options.get(CONF_PROMPT_CACHING_ENABLED, DEFAULT_PROMPT_CACHING_ENABLED):
            return entities

        entity_order = { name: None for name in entities.keys() }
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

        sorted_entities = {}
        for item_name, _ in sorted_items:
            sorted_entities[item_name] = entities[item_name]

        return sorted_entities

    def _set_prompt_caching(self, *, enabled=True):
        if enabled and not self.remove_prompt_caching_listener:
            _LOGGER.info("enabling prompt caching...")

            entity_ids = [
                state.entity_id for state in self.hass.states.async_all() \
                    if async_should_expose(self.hass, CONVERSATION_DOMAIN, state.entity_id)
            ]

            _LOGGER.debug(f"watching entities: {entity_ids}")

            self.remove_prompt_caching_listener = async_track_state_change(self.hass, entity_ids, self._async_cache_prompt)

        elif not enabled and self.remove_prompt_caching_listener:
            _LOGGER.info("disabling prompt caching...")
            self.remove_prompt_caching_listener()

    @callback
    async def _async_cache_prompt(self, entity, old_state, new_state):
        refresh_start = time.time()

        # track last update time so we can sort the context efficiently
        if entity:
            self.last_updated_entities[entity] = refresh_start

        llm_api: llm.APIInstance | None = None
        if self.entry.options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    self.hass, self.entry.options[CONF_LLM_HASS_API],
                )
            except HomeAssistantError:
                _LOGGER.exception("Failed to get LLM API when caching prompt!")
                return

        _LOGGER.debug(f"refreshing cached prompt because {entity} changed...")
        await self.hass.async_add_executor_job(self._cache_prompt, llm_api)

        refresh_end = time.time()
        _LOGGER.debug(f"cache refresh took {(refresh_end - refresh_start):.2f} sec")

    def _cache_prompt(self, llm_api: llm.APIInstance | None) -> None:
        # if a refresh is already scheduled then exit
        if self.cache_refresh_after_cooldown:
            return

        # if we are inside the cooldown period, request a refresh and exit
        current_time = time.time()
        fastest_prime_interval = self.entry.options.get(CONF_PROMPT_CACHING_INTERVAL, DEFAULT_PROMPT_CACHING_INTERVAL)
        if self.last_cache_prime and current_time - self.last_cache_prime < fastest_prime_interval:
            self.cache_refresh_after_cooldown = True
            return

        # try to acquire the lock, if we are still running for some reason, request a refresh and exit
        lock_acquired = self.model_lock.acquire(False)
        if not lock_acquired:
            self.cache_refresh_after_cooldown = True
            return

        try:
            raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
            prompt = self._format_prompt([
                { "role": "system", "message": self._generate_system_prompt(raw_prompt, llm_api)},
                { "role": "user", "message": "" }
            ], include_generation_prompt=False)

            input_tokens = self.llm.tokenize(
                prompt.encode(), add_bos=False
            )

            temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
            top_k = int(self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K))
            top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
            grammar = self.grammar if self.entry.options.get(CONF_USE_GBNF_GRAMMAR, DEFAULT_USE_GBNF_GRAMMAR) else None

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
                await self.hass.async_add_executor_job(self._cache_prompt, llm_api)

                refresh_end = time.time()
                _LOGGER.debug(f"cache refresh took {(refresh_end - refresh_start):.2f} sec")

        refresh_delay = self.entry.options.get(CONF_PROMPT_CACHING_INTERVAL, DEFAULT_PROMPT_CACHING_INTERVAL)
        async_call_later(self.hass, float(refresh_delay), refresh_if_requested)

    def _generate_stream(self, conversation: List[conversation.Content], llm_api: llm.APIInstance | None, user_input: conversation.ConversationInput) -> AsyncGenerator[TextGenerationResult, None]:
        """Async generator that yields TextGenerationResult as tokens are produced."""
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_k = int(self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K))
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        min_p = self.entry.options.get(CONF_MIN_P, DEFAULT_MIN_P)
        typical_p = self.entry.options.get(CONF_TYPICAL_P, DEFAULT_TYPICAL_P)

        _LOGGER.debug(f"Options: {self.entry.options}")

        # TODO: re-enable the context length check
        #     # FIXME: use the high level API so we can use the built-in prompt formatting
        #     input_tokens = self.llm.tokenize(
        #         prompt.encode(), add_bos=False
        #     )

        #     context_len = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        #     if len(input_tokens) >= context_len:
        #         num_entities = len(self._async_get_exposed_entities())
        #         context_size = self.entry.options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        #         self._warn_context_size()
        #         raise Exception(f"The model failed to produce a result because too many devices are exposed ({num_entities} devices) for the context size ({context_size} tokens)!")
        #     if len(input_tokens) + max_tokens >= context_len:
        #         self._warn_context_size()

        #     _LOGGER.debug(f"Processing {len(input_tokens)} input tokens...")

        messages = get_oai_formatted_messages(conversation)
        tools = None
        if llm_api:
            tools = get_oai_formatted_tools(llm_api, self._async_get_all_exposed_domains())

        _LOGGER.debug(f"Generating completion with {len(messages)} messages and {len(tools) if tools else 0} tools...")

        chat_completion = self.llm.create_chat_completion(
            messages,
            tools=tools,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            max_tokens=max_tokens,
            grammar=self.grammar,
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

        return self._async_parse_completion(llm_api, next_token=next_token())

