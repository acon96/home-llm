"""Defines the various LLM Backend Agents"""
from __future__ import annotations
from typing import Literal
import logging

from homeassistant.components.conversation import ConversationInput, ConversationResult, ConversationEntity
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.helpers import chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .entity import LocalLLMEntity, LocalLLMClient, LocalLLMConfigEntry
from .const import (
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry: LocalLLMConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> bool:
    """Set up Local LLM Conversation from a config entry."""

    for subentry in entry.subentries.values():
        if subentry.subentry_type != conversation.DOMAIN:
            continue

        # create one agent entity per conversation subentry
        agent_entity = LocalLLMAgent(hass, entry, subentry, entry.runtime_data)

        # make sure model is loaded
        await entry.runtime_data._async_load_model(dict(subentry.data))

        # register the agent entity
        async_add_entities([agent_entity], config_subentry_id=subentry.subentry_id,)

    return True

class LocalLLMAgent(ConversationEntity, AbstractConversationAgent, LocalLLMEntity):
    """Base Local LLM conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, subentry: ConfigSubentry, client: LocalLLMClient) -> None:
        super().__init__(hass, entry, subentry, client)

        if subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process a sentence."""
        with (
            chat_session.async_get_chat_session(
                self.hass, user_input.conversation_id
            ) as session,
            conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self.client._async_handle_message(user_input, chat_log, self.runtime_options)
