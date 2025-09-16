"""Defines the various LLM Backend Agents"""
from __future__ import annotations

import logging
from typing import List, Optional
from dataclasses import dataclass

from homeassistant.components.conversation import ConversationInput, ConversationResult, ConversationEntity
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm, chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .entity import LocalLLMEntity, LocalLLMClient
from .const import (
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


async def update_listener(hass: HomeAssistant, entry: ConfigEntry):
    """Handle options update."""
    hass.data[DOMAIN][entry.entry_id] = entry

    # call update handler
    agent: LocalLLMAgent = entry.runtime_data
    await hass.async_add_executor_job(agent.client._update_options, entry.options)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> bool:
    """Set up Local LLM Conversation from a config entry."""

    # handle updates to the options
    entry.async_on_unload(entry.add_update_listener(update_listener))

    # register the agent entity
    async_add_entities([entry.runtime_data])

    return True

class LocalLLMAgent(LocalLLMEntity, ConversationEntity, AbstractConversationAgent):
    """Base Local LLM conversation agent."""

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

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
            return await self.client._async_handle_message(user_input, chat_log, self.entry.options)
