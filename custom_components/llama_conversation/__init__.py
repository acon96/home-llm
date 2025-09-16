"""The Local LLM Conversation integration."""
from __future__ import annotations

import logging
from typing import Final

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import ATTR_ENTITY_ID, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm, device_registry as dr, entity_registry as er
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
    CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    CONF_IN_CONTEXT_EXAMPLES_FILE,
    DEFAULT_BACKEND_TYPE,
    DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    DEFAULT_IN_CONTEXT_EXAMPLES_FILE,
    BACKEND_TYPE_LLAMA_HF,
    BACKEND_TYPE_LLAMA_EXISTING,
    BACKEND_TYPE_TEXT_GEN_WEBUI,
    BACKEND_TYPE_GENERIC_OPENAI,
    BACKEND_TYPE_GENERIC_OPENAI_RESPONSES,
    BACKEND_TYPE_LLAMA_CPP_SERVER,
    BACKEND_TYPE_OLLAMA,
    CONF_CHAT_MODEL,
    CONF_DOWNLOADED_MODEL_FILE,
)
from .entity import LocalLLMClient, LocalLLMConfigEntry
from .backends.llamacpp import LlamaCppClient
from .backends.generic_openai import GenericOpenAIAPIClient, GenericOpenAIResponsesAPIClient
from .backends.tailored_openai import TextGenerationWebuiClient, LlamaCppServerClient
from .backends.ollama import OllamaAPIClient

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

PLATFORMS = (Platform.CONVERSATION,)

async def update_listener(hass: HomeAssistant, entry: LocalLLMConfigEntry):
    """Handle options update."""
    hass.data[DOMAIN][entry.entry_id] = entry

    # call update handler
    client: LocalLLMClient = entry.runtime_data
    await hass.async_add_executor_job(client._update_options, dict(entry.options))

async def async_setup_entry(hass: HomeAssistant, entry: LocalLLMConfigEntry) -> bool:

    # make sure the API is registered
    if not any([x.id == HOME_LLM_API_ID for x in llm.async_get_apis(hass)]):
        llm.async_register_api(hass, HomeLLMAPI(hass))

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry

    icl_examples_filename = None
    if entry.options.get(CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES, DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES):
        icl_examples_filename = entry.options.get(CONF_IN_CONTEXT_EXAMPLES_FILE, DEFAULT_IN_CONTEXT_EXAMPLES_FILE)

    def create_client(backend_type):
        client_cls = None

        if backend_type in [ BACKEND_TYPE_LLAMA_HF, BACKEND_TYPE_LLAMA_EXISTING ]:
            client_cls = LlamaCppClient
        elif backend_type == BACKEND_TYPE_GENERIC_OPENAI:
            client_cls = GenericOpenAIAPIClient
        elif backend_type == BACKEND_TYPE_GENERIC_OPENAI_RESPONSES:
            client_cls = GenericOpenAIResponsesAPIClient
        elif backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI:
            client_cls = TextGenerationWebuiClient
        elif backend_type == BACKEND_TYPE_LLAMA_CPP_SERVER:
            client_cls = LlamaCppServerClient
        elif backend_type == BACKEND_TYPE_OLLAMA:
            client_cls = OllamaAPIClient

        if client_cls is None:
            raise ValueError(f"Unknown backend type {backend_type}")
        return client_cls(hass, icl_examples_filename)

    # create the agent in an executor job because the constructor calls `open()`
    backend_type = entry.data.get(CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE)
    entry.runtime_data = await hass.async_add_executor_job(create_client, backend_type)

    # handle updates to the options
    entry.async_on_unload(entry.add_update_listener(update_listener))

    # call load model
    await entry.runtime_data._async_load_model(entry)

    # forward setup to platform to register the entity
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: LocalLLMConfigEntry) -> bool:
    """Unload Ollama."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False
    hass.data[DOMAIN].pop(entry.entry_id)
    return True

# TODO: split out which options are per-model and which ones are conversation-specific
# and only migrate the conversation-specific ones to the subentry
ENTRY_KEYS = []
SUBENTRY_KEYS = []

async def async_migrate_entry(hass: HomeAssistant, config_entry: LocalLLMConfigEntry):
    """Migrate old entry."""
    _LOGGER.debug("Migrating from version %s", config_entry.version)

    # 1 -> 2: This was a breaking change so force users to re-create entries
    if config_entry.version == 1:
        _LOGGER.error("Cannot upgrade models that were created prior to v0.3. Please delete and re-create them.")
        return False
    
    # If already at or above the target version nothing to do
    if config_entry.version >= 3:
        _LOGGER.debug("Entry already migrated (version %s)", config_entry.version)
        return True

    # Migrate each existing config entry to use subentries for conversations
    # We will create a conversation subentry using the entry.options plus any
    # model identifier stored in entry.data (CONF_CHAT_MODEL / CONF_DOWNLOADED_MODEL_FILE)

    entries = sorted(
        hass.config_entries.async_entries(DOMAIN),
        key=lambda e: e.disabled_by is not None,
    )

    for entry in entries:
        # Skip entries that already have subentries
        if entry.subentries:
            continue

        # Build subentry data from existing options and model info
        subentry_data = { k: v for k, v in (entry.options or {}).items() if k in SUBENTRY_KEYS }
        entry_data = { k: v for k, v in (entry.data or {}).items() if k in ENTRY_KEYS }

        subentry = ConfigSubentry(
            data=MappingProxyType(subentry_data),
            subentry_type="conversation",
            title=entry.title,
            unique_id=None,
        )

        hass.config_entries.async_add_subentry(entry, subentry)

        # Move entity/device registry associations to the new subentry where applicable
        entity_registry = er.async_get(hass)
        device_registry = dr.async_get(hass)

        conversation_entity_id = entity_registry.async_get_entity_id(
            "conversation",
            DOMAIN,
            entry.entry_id,
        )
        device = device_registry.async_get_device(identifiers={(DOMAIN, entry.entry_id)})

        if conversation_entity_id is not None:
            conversation_entity_entry = entity_registry.entities[conversation_entity_id]
            entity_disabled_by = conversation_entity_entry.disabled_by
            # Keep a sensible disabled flag when migrating
            if (
                entity_disabled_by is er.RegistryEntryDisabler.CONFIG_ENTRY
                and not all(e.disabled_by is not None for e in entries if e.entry_id != entry.entry_id)
            ):
                entity_disabled_by = (
                    er.RegistryEntryDisabler.DEVICE if device else er.RegistryEntryDisabler.USER
                )
            entity_registry.async_update_entity(
                conversation_entity_id,
                config_entry_id=entry.entry_id,
                config_subentry_id=subentry.subentry_id,
                disabled_by=entity_disabled_by,
                new_unique_id=subentry.subentry_id,
            )

        if device is not None:
            # Adjust device registry identifiers to point to the subentry
            device_disabled_by = device.disabled_by
            if (
                device.disabled_by is dr.DeviceEntryDisabler.CONFIG_ENTRY
            ):
                device_disabled_by = dr.DeviceEntryDisabler.USER
            device_registry.async_update_device(
                device.id,
                disabled_by=device_disabled_by,
                new_identifiers={(DOMAIN, subentry.subentry_id)},
                add_config_subentry_id=subentry.subentry_id,
                add_config_entry_id=entry.entry_id,
            )

        # Update the parent entry to remove model-level fields and clear options
        hass.config_entries.async_update_entry(entry, data=entry_data, options={}, version=3)

    _LOGGER.debug("Migration to subentries complete")

    return True

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
            name="Home-LLM (v1-v3)",
        )

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        """Return the instance of the API."""
        return llm.APIInstance(
            api=self,
            api_prompt="Call services in Home Assistant by passing the service name and the device to control.",
            llm_context=llm_context,
            tools=[HassServiceTool()],
        )
