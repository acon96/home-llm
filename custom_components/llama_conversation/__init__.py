"""The Local LLM Conversation integration."""
from __future__ import annotations

import logging
from typing import Final

import homeassistant.components.conversation as ha_conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.util.json import JsonObjectType

import voluptuous as vol

from .agent import (
    LocalLLMAgent,
    LlamaCppAgent,
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
    ALLOWED_SERVICE_CALL_ARGUMENTS,
    DOMAIN,
    HOME_LLM_API_ID,
    SERVICE_TOOL_NAME,
)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

async def update_listener(hass: HomeAssistant, entry: ConfigEntry):
    """Handle options update."""
    hass.data[DOMAIN][entry.entry_id] = entry
    
    # call update handler
    agent: LocalLLMAgent = ha_conversation.get_agent_manager(hass).async_get_agent(entry.entry_id)
    await hass.async_add_executor_job(agent._update_options)

    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Local LLM Conversation from a config entry."""

    # make sure the API is registered
    if not any([x.id == HOME_LLM_API_ID for x in llm.async_get_apis(hass)]):
        llm.async_register_api(hass, HomeLLMAPI(hass))

    def create_agent(backend_type):
        agent_cls = None

        if backend_type in [ BACKEND_TYPE_LLAMA_HF, BACKEND_TYPE_LLAMA_EXISTING ]:
            agent_cls = LlamaCppAgent
        elif backend_type == BACKEND_TYPE_GENERIC_OPENAI:
            agent_cls = GenericOpenAIAPIAgent
        elif backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI:
            agent_cls = TextGenerationWebuiAgent
        elif backend_type == BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER:
            agent_cls = LlamaCppPythonAPIAgent
        elif backend_type == BACKEND_TYPE_OLLAMA:
            agent_cls = OllamaAPIAgent
        
        return agent_cls(hass, entry)

    # create the agent in an executor job because the constructor calls `open()`
    backend_type = entry.data.get(CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE)
    agent = await hass.async_add_executor_job(create_agent, backend_type)

    # call load model
    await agent._async_load_model(entry)

    # handle updates to the options
    entry.async_on_unload(entry.add_update_listener(update_listener))

    ha_conversation.async_set_agent(hass, entry, agent)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Local LLM."""
    hass.data[DOMAIN].pop(entry.entry_id)
    ha_conversation.async_unset_agent(hass, entry)
    return True

async def async_migrate_entry(hass, config_entry: ConfigEntry):
    """Migrate old entry."""
    _LOGGER.debug("Migrating from version %s", config_entry.version)

    # 1 -> 2: This was a breaking change so force users to re-create entries
    if config_entry.version == 1:
        _LOGGER.error("Cannot upgrade models that were created prior to v0.3. Please delete and re-create them.")
        return False

    _LOGGER.debug("Migration to version %s successful", config_entry.version)

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

    ALLOWED_SERVICES: Final[list[str]] = [
        "turn_on", "turn_off", "toggle", "press", "increase_speed", "decrease_speed", "open_cover", "close_cover", "stop_cover",
        "lock", "unlock", "start", "stop", "return_to_base", "pause", "cancel", "add_item"
    ]
    ALLOWED_DOMAINS: Final[list[str]] = [
        "light", "switch", "button", "fan", "cover", "lock", "media_player", "climate", "vacuum", "todo", "timer", "script",
    ]

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        """Call the tool."""
        domain, service = tuple(tool_input.tool_args["service"].split("."))
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
