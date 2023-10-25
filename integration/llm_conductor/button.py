"""Support for buttons which integrates with other components."""
from __future__ import annotations

import logging

# import transformers

from homeassistant.config_entries import ConfigEntry
from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    entity = LLMConductor(hass, entry)
    hass.data[DOMAIN][f"{entry.entry_id}-entity"] = entity

    async_add_entities([entity])


class LLMConductor(ButtonEntity):
    """Representation of an Awesome Light."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize an LLMConductor"""
        self._name = entry.title
        self._state = None
        self._model = None
        self._hass = hass

        self._attr_unique_id = f"{entry.entry_id}-llm-conductor"

    @property
    def name(self) -> str:
        """Return the display name of this light."""
        return self._name

    def press(self) -> None:
        """Presses the button"""
        _LOGGER.info("Button was pressed")

        for domain_name, services in self._hass.services.services.items():
            _LOGGER.info(f"{domain_name} - {', '.join(services.keys())}")

        check_domains = [
            "light",
            "fan",
            "cover",
        ]
        for domain in check_domains:
            for entity_id in self._hass.states.entity_ids(domain):
                state = self._hass.states.get(entity_id)
                _LOGGER.info(f"{entity_id} = {state.state}")

    async def unload_model(self):
        _LOGGER.info("Unloading Model")
