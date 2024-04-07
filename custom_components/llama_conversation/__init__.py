"""The Local LLaMA Conversation integration."""
from __future__ import annotations

import logging

import homeassistant.components.conversation as ha_conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv

from .agent import (
    LLaMAAgent,
    LocalLLaMAAgent,
    GenericOpenAIAPIAgent,
    TextGenerationWebuiAgent,
    LlamaCppPythonAPIAgent,
    OllamaAPIAgent,
)

from .const import (
    CONF_BACKEND_TYPE,
    DEFAULT_BACKEND_TYPE,
    BACKEND_TYPE_LLAMA_HF,
    BACKEND_TYPE_LLAMA_EXISTING,
    BACKEND_TYPE_TEXT_GEN_WEBUI,
    BACKEND_TYPE_GENERIC_OPENAI,
    BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER,
    BACKEND_TYPE_OLLAMA,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

async def update_listener(hass: HomeAssistant, entry: ConfigEntry):
    """Handle options update."""
    hass.data[DOMAIN][entry.entry_id] = entry
    
    # call update handler
    agent: LLaMAAgent = await ha_conversation._get_agent_manager(hass).async_get_agent(entry.entry_id)
    agent._update_options()

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
