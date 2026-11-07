"""The Local LLM Conversation integration."""
from __future__ import annotations

import logging
import os
import shutil
from typing import Final

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import ATTR_ENTITY_ID, Platform, CONF_HOST, CONF_PORT, CONF_SSL, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm, device_registry as dr, entity_registry as er, issue_registry as ir
from homeassistant.util.json import JsonObjectType
from types import MappingProxyType

import voluptuous as vol

from .const import (
    ALLOWED_SERVICE_CALL_ARGUMENTS,
    DOMAIN,
    HOME_LLM_API_ID,
    SERVICE_TOOL_NAME,
    SERVICE_TOOL_ALLOWED_SERVICES,
    SERVICE_TOOL_ALLOWED_DOMAINS,
    CONF_BACKEND_TYPE,
    CONF_INSTALLED_LLAMACPP_VERSION,
    CONF_SELECTED_LANGUAGE,
    CONF_API_KEY,
    CONF_API_PATH,
    CONF_CHAT_MODEL, CONF_DOWNLOADED_MODEL_QUANTIZATION, CONF_DOWNLOADED_MODEL_FILE, CONF_REQUEST_TIMEOUT, CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_ENABLE_STREAMING, CONF_GBNF_GRAMMAR,
    CONF_PROMPT, CONF_TEMPERATURE, CONF_TOP_K, CONF_TOP_P, CONF_MIN_P, CONF_TYPICAL_P, CONF_MAX_TOKENS,
    CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES, CONF_IN_CONTEXT_EXAMPLES_FILE, CONF_NUM_IN_CONTEXT_EXAMPLES, CONF_EXTRA_ATTRIBUTES_TO_EXPOSE, 
    CONF_THINKING_PREFIX, CONF_THINKING_SUFFIX, CONF_TOOL_CALL_PREFIX, CONF_TOOL_CALL_SUFFIX,
    CONF_PROMPT_CACHING_ENABLED, CONF_PROMPT_CACHING_INTERVAL, CONF_CONTEXT_LENGTH,
    CONF_LLAMACPP_BATCH_SIZE, CONF_LLAMACPP_THREAD_COUNT, CONF_LLAMACPP_BATCH_THREAD_COUNT,
    CONF_LLAMACPP_ENABLE_FLASH_ATTENTION, CONF_ENABLE_LEGACY_TOOL_CALLING,
    CONF_OLLAMA_JSON_MODE, CONF_OLLAMA_KEEP_ALIVE_MIN,
    DEFAULT_BACKEND_TYPE,
    BACKEND_TYPE_LLAMA_CPP,
    BACKEND_TYPE_GENERIC_OPENAI,
    BACKEND_TYPE_GENERIC_OPENAI_RESPONSES,
    BACKEND_TYPE_LLAMA_CPP_SERVER,
    BACKEND_TYPE_OLLAMA,
    BACKEND_TYPE_ANTHROPIC,
    BACKEND_TYPE_LLAMA_EXISTING_OLD,
    BACKEND_TYPE_LLAMA_HF_OLD,
)
from .entity import LocalLLMClient, LocalLLMConfigEntry
from .backends.llamacpp import LlamaCppClient
from .backends.generic_openai import GenericOpenAIAPIClient, GenericOpenAIResponsesAPIClient
from .backends.tailored_openai import LlamaCppServerClient
from .backends.ollama import OllamaAPIClient
from .backends.anthropic import AnthropicAPIClient
from .utils import get_llama_cpp_python_version, download_model_from_hf

_LOGGER = logging.getLogger(__name__)

# Options and backends that no longer exist. Migration code still needs their old spellings to move or
# drop whatever an entry saved under them.
BACKEND_TYPE_TEXT_GEN_WEBUI: Final[str] = "text-generation-webui_api"
REMOVED_OPTION_KEYS: Final[tuple[str, ...]] = (
    "refresh_prompt_per_turn",
    "remember_conversation",
    "remember_num_interactions",
    "remember_conversation_time_minutes",
    "use_server_sampling_defaults",
    "gbnf_grammar",
    "gbnf_grammar_file",
    "text_generation_webui_preset",
    "text_generation_webui_admin_key",
    "text_generation_webui_chat_mode",
)
SAMPLING_OPTION_KEYS: Final[tuple[str, ...]] = (CONF_TEMPERATURE, CONF_TOP_K, CONF_TOP_P, CONF_MIN_P, CONF_TYPICAL_P)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

PLATFORMS = (Platform.CONVERSATION,  Platform.AI_TASK)

BACKEND_TO_CLS: dict[str, type[LocalLLMClient]] = {
    BACKEND_TYPE_LLAMA_CPP: LlamaCppClient,
    BACKEND_TYPE_GENERIC_OPENAI: GenericOpenAIAPIClient,
    BACKEND_TYPE_GENERIC_OPENAI_RESPONSES: GenericOpenAIResponsesAPIClient,
    BACKEND_TYPE_LLAMA_CPP_SERVER: LlamaCppServerClient,
    BACKEND_TYPE_OLLAMA: OllamaAPIClient,
    BACKEND_TYPE_ANTHROPIC: AnthropicAPIClient,
}

async def async_setup_entry(hass: HomeAssistant, entry: LocalLLMConfigEntry) -> bool:

    # make sure the API is registered
    if not any([x.id == HOME_LLM_API_ID for x in llm.async_get_apis(hass)]):
        llm.async_register_api(hass, HomeLLMAPI(hass))

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry

    def create_client(backend_type):
        _LOGGER.debug("Creating Local LLM client of type %s", backend_type)
        # Merge entry.data and entry.options - data has connection info, options has model settings
        client_options = {**dict(entry.data), **dict(entry.options)}
        return BACKEND_TO_CLS[backend_type](hass, client_options)

    # create the agent in an executor job because the constructor calls `open()`
    backend_type = entry.data.get(CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE)
    entry.runtime_data = await hass.async_add_executor_job(create_client, backend_type)

    try:
        await entry.runtime_data.async_validate_startup(entry)

        # forward setup to platform to register the entity
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except Exception:
        hass.data[DOMAIN].pop(entry.entry_id, None)
        raise

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    return True


async def _async_update_listener(hass: HomeAssistant, entry: LocalLLMConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)

async def async_unload_entry(hass: HomeAssistant, entry: LocalLLMConfigEntry) -> bool:
    """Unload the integration."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    if entry.data[CONF_BACKEND_TYPE] == BACKEND_TYPE_LLAMA_CPP:
        # clean up any disk cache resources
        def cleanup_cache_dir():
            model_name = entry.data.get(CONF_CHAT_MODEL) or entry.options.get(CONF_CHAT_MODEL)
            if not model_name:
                return
            cache_dir = model_name.strip().replace(" ", "_").lower()
            full_path = os.path.join(hass.config.media_dirs.get("local", hass.config.path("media")), "kv_cache", cache_dir)
            shutil.rmtree(full_path, ignore_errors=True)

        await hass.async_add_executor_job(cleanup_cache_dir)
    hass.data[DOMAIN].pop(entry.entry_id)
    return True


async def async_migrate_entry(hass: HomeAssistant, config_entry: LocalLLMConfigEntry):
    """Migrate old entry."""
    _LOGGER.debug("Migrating from version %s", config_entry.version)

    # 1 -> 2: This was a breaking change so force users to re-create entries
    if config_entry.version == 1:
        _LOGGER.error("Cannot upgrade models that were created prior to v0.3. Please delete and re-create them.")
        return False

    # Migrate the config_entry to be an entry + sub-entry
    if config_entry.version == 2:
        ENTRY_DATA_KEYS = [CONF_BACKEND_TYPE]
        ENTRY_OPTIONS_KEYS = [CONF_SELECTED_LANGUAGE, CONF_HOST, CONF_PORT, CONF_SSL, CONF_API_KEY, CONF_API_PATH]
        SUBENTRY_KEYS = [
            CONF_CHAT_MODEL, CONF_DOWNLOADED_MODEL_QUANTIZATION, CONF_DOWNLOADED_MODEL_FILE, CONF_REQUEST_TIMEOUT, CONF_MAX_TOOL_CALL_ITERATIONS,
            CONF_ENABLE_STREAMING, CONF_GBNF_GRAMMAR,
            CONF_LLM_HASS_API, CONF_PROMPT, CONF_TEMPERATURE, CONF_TOP_K, CONF_TOP_P, CONF_MIN_P, CONF_TYPICAL_P, CONF_MAX_TOKENS,
            CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES, CONF_IN_CONTEXT_EXAMPLES_FILE, CONF_NUM_IN_CONTEXT_EXAMPLES, CONF_EXTRA_ATTRIBUTES_TO_EXPOSE, 
            CONF_THINKING_PREFIX, CONF_THINKING_SUFFIX, CONF_TOOL_CALL_PREFIX, CONF_TOOL_CALL_SUFFIX,
            CONF_PROMPT_CACHING_ENABLED, CONF_PROMPT_CACHING_INTERVAL, CONF_CONTEXT_LENGTH,
            CONF_LLAMACPP_BATCH_SIZE, CONF_LLAMACPP_THREAD_COUNT, CONF_LLAMACPP_BATCH_THREAD_COUNT,
            CONF_LLAMACPP_ENABLE_FLASH_ATTENTION, CONF_ENABLE_LEGACY_TOOL_CALLING,
            CONF_OLLAMA_JSON_MODE, CONF_OLLAMA_KEEP_ALIVE_MIN,
            *REMOVED_OPTION_KEYS,
        ]

        # Build entry data/options & subentry data from existing options and model info
        source_data = {**config_entry.data}
        source_data.update(config_entry.options)

        entry_data = { k: v for k, v in source_data.items() if k in ENTRY_DATA_KEYS }
        entry_options = { k: v for k, v in source_data.items() if k in ENTRY_OPTIONS_KEYS }
        subentry_data = { k: v for k, v in source_data.items() if k in SUBENTRY_KEYS }

        backend = config_entry.data[CONF_BACKEND_TYPE]
        if backend == BACKEND_TYPE_LLAMA_EXISTING_OLD or backend == BACKEND_TYPE_LLAMA_HF_OLD:
            backend = BACKEND_TYPE_LLAMA_CPP
            entry_data[CONF_BACKEND_TYPE] = BACKEND_TYPE_LLAMA_CPP
            entry_options[CONF_INSTALLED_LLAMACPP_VERSION] = await hass.async_add_executor_job(get_llama_cpp_python_version)
        else:
            # ensure all remote backends have a path set
            entry_options[CONF_API_PATH] = entry_options.get(CONF_API_PATH, "")
        
        entry_title = BACKEND_TO_CLS[backend].get_name(entry_options)

        subentry = ConfigSubentry(
            data=MappingProxyType(subentry_data),
            subentry_type="conversation",
            title=config_entry.title.split("'")[-2],
            unique_id=None,
        )

        # create sub-entry
        hass.config_entries.async_add_subentry(config_entry, subentry)

        # update the parent entry
        hass.config_entries.async_update_entry(config_entry, title=entry_title, data=entry_data, options=entry_options, version=3)

        _LOGGER.debug("Migration to subentries complete")
    
    if config_entry.version == 3 and config_entry.minor_version == 0:
        # add the downloaded model file to options if missing
        if config_entry.data.get(CONF_BACKEND_TYPE) == BACKEND_TYPE_LLAMA_CPP:
            for subentry in config_entry.subentries.values():
                if subentry.data.get(CONF_DOWNLOADED_MODEL_FILE) is None:
                    model_name = subentry.data[CONF_CHAT_MODEL]
                    quantization_type = subentry.data[CONF_DOWNLOADED_MODEL_QUANTIZATION]
                    storage_folder = os.path.join(hass.config.media_dirs.get("local", hass.config.path("media")), "models")

                    new_options = dict(subentry.data)
                    file_name = await hass.async_add_executor_job(download_model_from_hf, model_name, quantization_type, storage_folder, True)
                    new_options[CONF_DOWNLOADED_MODEL_FILE] = file_name

                    hass.config_entries.async_update_subentry(
                        config_entry, subentry, data=MappingProxyType(new_options)
                    )

        hass.config_entries.async_update_entry(config_entry, minor_version=1)
        
        _LOGGER.debug("Migration to add downloaded model file complete") 

    if config_entry.version == 3 and config_entry.minor_version == 1:
        # convert selected APIs from single value to list
        api_to_convert = config_entry.options.get(CONF_LLM_HASS_API)
        new_options = dict(config_entry.options)
        if api_to_convert is not None:
            new_options[CONF_LLM_HASS_API] = [api_to_convert]
        else:
            new_options[CONF_LLM_HASS_API] = []

        hass.config_entries.async_update_entry(
            config_entry, options=MappingProxyType(new_options)
        )
        hass.config_entries.async_update_entry(config_entry, minor_version=2)

    if config_entry.version == 3 and config_entry.minor_version == 2:
        # v3.2 recorded whether to keep the server's sampling defaults. v4 dropped that option, so
        # whatever sampling parameters an entry has are simply left alone from here on.
        hass.config_entries.async_update_entry(config_entry, minor_version=3)

    if config_entry.version == 3:
        await async_migrate_to_v4(hass, config_entry)

    return True


async def async_migrate_to_v4(hass: HomeAssistant, config_entry: LocalLLMConfigEntry):
    """Move a v3 entry onto the v4 option set.

    v4 drops the options that the model defaults or Home Assistant handle better than a setting does:
    refreshing the system prompt per turn, remembering a fixed number of messages, deferring to the
    server sampling defaults, the grammar file, and the text-generation-webui backend.
    """
    options = dict(config_entry.options)
    data = dict(config_entry.data)
    # read before the removed keys are dropped below
    used_server_sampling_defaults = options.get("use_server_sampling_defaults") is True or any(
        subentry.data.get("use_server_sampling_defaults") is True for subentry in config_entry.subentries.values()
    )
    issues: list[str] = []

    if data.get(CONF_BACKEND_TYPE) == BACKEND_TYPE_TEXT_GEN_WEBUI:
        data[CONF_BACKEND_TYPE] = BACKEND_TYPE_GENERIC_OPENAI
        for subentry in config_entry.subentries.values():
            preset = subentry.data.get("text_generation_webui_preset")
            chat_mode = subentry.data.get("text_generation_webui_chat_mode", "chat")
            if preset or chat_mode != "chat":
                issues.append(f"'{subentry.title}' used the preset '{preset or 'none'}' and the '{chat_mode}' chat mode")

    for key in REMOVED_OPTION_KEYS:
        options.pop(key, None)
        # the text-generation-webui admin key was stored with the connection settings
        data.pop(key, None)

    # the configured sampling parameters were being ignored, so drop them instead of starting to send them
    if used_server_sampling_defaults:
        for key in SAMPLING_OPTION_KEYS:
            options.pop(key, None)

    hass.config_entries.async_update_entry(
        config_entry,
        data=MappingProxyType(data),
        options=MappingProxyType(options),
        version=4,
        minor_version=0,
    )

    for subentry in config_entry.subentries.values():
        new_options = dict(subentry.data)

        # the grammar lived in a file inside custom_components, where an update would overwrite it
        grammar_file = new_options.pop("gbnf_grammar_file", None)
        if new_options.pop("gbnf_grammar", False) and grammar_file:
            path = os.path.join(os.path.dirname(__file__), grammar_file)
            try:
                with open(path) as grammar:
                    new_options[CONF_GBNF_GRAMMAR] = "".join(grammar.readlines())
            except OSError:
                _LOGGER.warning("Could not read %s while migrating %s, disabling its grammar", path, subentry.title)

        # the configured sampling parameters were being ignored, so drop them instead of starting to send them
        if used_server_sampling_defaults:
            for key in SAMPLING_OPTION_KEYS:
                new_options.pop(key, None)

        for key in REMOVED_OPTION_KEYS:
            new_options.pop(key, None)

        if new_options != dict(subentry.data):
            hass.config_entries.async_update_subentry(config_entry, subentry, data=MappingProxyType(new_options))

    if issues:
        ir.async_create_issue(
            hass,
            DOMAIN,
            "text_generation_webui_removed",
            is_fixable=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key="text_generation_webui_removed",
            translation_placeholders={"details": "\n".join(issues)},
        )

    _LOGGER.debug("Migration to v4 complete")

class HassServiceTool(llm.Tool):
    """Tool to get the current time."""

    name: Final[str] = SERVICE_TOOL_NAME
    description: Final[str] = "Executes a Home Assistant service"

    # Optional. A voluptuous schema of the input parameters.
    parameters = vol.Schema({
        vol.Required('service'): str,
        vol.Required('target_device'): str,
        vol.Optional('rgb_color'): str,
        vol.Optional('brightness'): float,
        vol.Optional('temperature'): float,
        vol.Optional('humidity'): float,
        vol.Optional('fan_mode'): str,
        vol.Optional('hvac_mode'): str,
        vol.Optional('preset_mode'): str,
        vol.Optional('duration'): str,
        vol.Optional('item'): str,
    })

    ALLOWED_SERVICES: Final[list[str]] = SERVICE_TOOL_ALLOWED_SERVICES
    ALLOWED_DOMAINS: Final[list[str]] = SERVICE_TOOL_ALLOWED_DOMAINS

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        """Call the tool."""
        try:
            domain, service = tuple(tool_input.tool_args["service"].split("."))
        except ValueError:
            return { "result": "unknown service" }

        target_device = tool_input.tool_args["target_device"]

        if domain not in self.ALLOWED_DOMAINS or service not in self.ALLOWED_SERVICES:
            return { "result": "unknown service" }

        if domain == "script" and service not in ["reload", "turn_on", "turn_off", "toggle"]:
            return { "result": "unknown service" }

        service_data = {ATTR_ENTITY_ID: target_device}
        for attr in ALLOWED_SERVICE_CALL_ARGUMENTS:
            if attr in tool_input.tool_args.keys():
                service_data[attr] = tool_input.tool_args[attr]
        try:
            await hass.services.async_call(
                domain,
                service,
                service_data=service_data,
                blocking=True,
            )
        except Exception:
            _LOGGER.exception("Failed to execute service for model")
            return { "result": "failed" }

        return { "result": "success" }

class HomeLLMAPI(llm.API):
    """
    An API that allows calling Home Assistant services to maintain compatibility
    with the older (v3 and older) Home LLM models
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Init the class."""
        super().__init__(
            hass=hass,
            id=HOME_LLM_API_ID,
            name="Home Assistant Services",
        )

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        """Return the instance of the API."""
        return llm.APIInstance(
            api=self,
            api_prompt="Call services in Home Assistant by passing the service name and the device to control. Designed for Home-LLM Models (v1-v3)",
            llm_context=llm_context,
            tools=[HassServiceTool()],
        )
