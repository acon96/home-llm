"""Defines the various LLM Backend Agents"""
from __future__ import annotations
from collections import OrderedDict
from typing import Any, Literal, List, Tuple, Dict
import logging

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationEntity, ConversationInput, ConversationResult
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import TemplateError, HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from custom_components.llama_conversation.utils import MalformedToolCallException, PreparedLLMAPI, prompt_as_literal, strip_thinking_blocks

from .entity import LocalLLMEntity, LocalLLMClient, LocalLLMConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_CONTEXT_LENGTH,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_PROMPT,
    CONF_THINKING_PREFIX,
    CONF_THINKING_SUFFIX,
    CONTEXT_BUDGET_MARGIN_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_CALL_ITERATIONS,
    DEFAULT_THINKING_PREFIX,
    DEFAULT_THINKING_SUFFIX,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

# A conversation keeps its rendered system prompt for as long as it lasts, so that a backend can reuse
# its key/value cache. Bounded because nothing else about a conversation says how many there will be.
MAX_CACHED_SYSTEM_PROMPTS = 16

# What Assist hears when a turn produced nothing worth speaking. Saying nothing at all looks like a
# broken speaker, so the integration has always admitted it had nothing to report.
NO_SPEECH_MESSAGE = "I don't have anything to say right now"


async def async_setup_entry(hass: HomeAssistant, entry: LocalLLMConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> bool:
    """Set up Local LLM Conversation from a config entry."""

    for subentry in entry.subentries.values():
        if subentry.subentry_type != conversation.DOMAIN:
            continue

        if CONF_CHAT_MODEL not in subentry.data:
            _LOGGER.warning("Conversation subentry %s missing required config key %s, You must delete the model and re-create it.", subentry.subentry_id, CONF_CHAT_MODEL)
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

        # the API can also come from the entry options (migrated entries), so use the runtime options
        if self.runtime_options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

        # conversation id -> (prompt template, rendered prompt)
        self._system_prompts: OrderedDict[str, Tuple[str, str]] = OrderedDict()

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

    @property
    def supports_streaming(self) -> bool:
        """Forward streaming support from the backend client."""
        return getattr(self.client, '_attr_supports_streaming', False)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> ConversationResult:
        """Answer a user message that the chat log has already been given.

        Home Assistant owns the chat session and the chat log. This method only configures the log with
        the prompt and tools, streams the model's answer (plus the tool calls it triggers) back into the
        log, and turns the result into an intent response.
        """
        options = self.runtime_options

        # A configured value below 1 means "answer in a single shot": the model is expected to produce
        # its final answer together with the tool call, so running out of iterations is not an error.
        requested_iterations = int(options.get(CONF_MAX_TOOL_CALL_ITERATIONS, DEFAULT_MAX_TOOL_CALL_ITERATIONS))
        iterations = max(1, requested_iterations)

        llm_context = user_input.as_llm_context(DOMAIN)

        # Resolve the API here (rather than letting the chat log do it) because the tool list is rendered
        # into the prompt template. PreparedLLMAPI makes sure it is only resolved once per turn.
        llm_api: llm.APIInstance | None = None
        api_id = options.get(CONF_LLM_HASS_API)
        if api_id:
            try:
                llm_api = await llm.async_get_api(self.hass, api_id, llm_context)
            except HomeAssistantError as err:
                _LOGGER.error("Error getting LLM API %s: %s", api_id, err)
                return self._error_result(user_input, chat_log, intent.IntentResponseErrorCode.UNKNOWN, f"Error preparing LLM API: {err}")

        try:
            system_prompt = self._get_system_prompt(chat_log, llm_api, options)
        except TemplateError as err:
            _LOGGER.error("Error rendering prompt: %s", err)
            return self._error_result(user_input, chat_log, intent.IntentResponseErrorCode.UNKNOWN, f"Sorry, I had a problem with my template: {err}")

        try:
            # This composes the system prompt, registers the tools with the chat log so it can execute
            # the tool calls it is handed, and notifies subscribers/traces about the update.
            await chat_log.async_provide_llm_data(
                llm_context,
                PreparedLLMAPI(llm_api) if llm_api else None,
                prompt_as_literal(system_prompt),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        message_history = self._build_message_history(chat_log, options)

        for _iteration in range(iterations):
            _LOGGER.debug("Generating response for %s (%s tool call iterations configured)", user_input.text, requested_iterations)
            try:
                generation = await self.client._async_generate(
                    message_history,
                    user_input.agent_id,
                    chat_log,
                    options,
                )
                async for message in generation:
                    message_history.append(message)
                    _LOGGER.debug("Added message to history: %s", message)
            except MalformedToolCallException as err:
                # Ask the model to correct itself. These helper messages exist only to be fed back to the
                # model, so they stay out of the chat log.
                message_history.extend(err.as_tool_messages())
                _LOGGER.debug("Malformed tool call produced", exc_info=err)
                continue
            except Exception as err:
                _LOGGER.exception("There was a problem talking to the backend")
                return self._error_result(
                    user_input,
                    chat_log,
                    intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                    f"Sorry, there was a problem talking to the backend: {repr(err)}",
                )

            # The chat log runs tool calls as they stream in, so an unresponded tool result means the
            # model has not been given the chance to say anything about what just happened yet.
            if not chat_log.unresponded_tool_results:
                break
        else:
            if requested_iterations > 0:
                return self._error_result(
                    user_input,
                    chat_log,
                    intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                    "Sorry, I ran out of attempts to handle your request",
                )

        return self._build_result(user_input, chat_log, options)

    def _get_system_prompt(
        self,
        chat_log: conversation.ChatLog,
        llm_api: llm.APIInstance | None,
        options: Dict[str, Any],
    ) -> str:
        """Render the system prompt for a conversation, or reuse the one it was started with.

        The prompt is rendered on the first turn of a conversation and reused byte for byte afterwards,
        which is what lets a backend reuse its key/value cache instead of processing the device list
        again. The exposed states therefore age for the life of the conversation, and the model is
        expected to act on what its tools report rather than on what the prompt once said. A conversation
        whose prompt template changed underneath it is rendered again.
        """
        raw_prompt = options.get(CONF_PROMPT, DEFAULT_PROMPT)
        cache_key = chat_log.conversation_id or ""

        cached = self._system_prompts.get(cache_key)
        if cached is not None and cached[0] == raw_prompt:
            self._system_prompts.move_to_end(cache_key)
            return cached[1]

        prompt = self.client._generate_system_prompt(raw_prompt, llm_api, options)

        self._system_prompts[cache_key] = (raw_prompt, prompt)
        while len(self._system_prompts) > MAX_CACHED_SYSTEM_PROMPTS:
            self._system_prompts.popitem(last=False)

        return prompt

    def _build_message_history(
        self,
        chat_log: conversation.ChatLog,
        options: Dict[str, Any],
    ) -> List[conversation.Content]:
        """Collect the messages the model gets to see for this turn, trimmed to fit its context.

        The chat log is the source of truth: it holds the composed system prompt, the conversation so far
        and the user message that is being answered. Trimming happens on this copy only, so the log keeps
        the whole exchange for the UI, other agents and the next turn.

        Everything that fits the model context is sent, newest first. The system prompt and the message
        being answered are never trimmed, and a tool call is never separated from its results.
        """
        system_prompt, *turn = chat_log.content
        groups = self._group_tool_calls(turn)
        budget = self._context_token_budget(options) - self.client._count_tokens([system_prompt], options)

        kept: List[List[conversation.Content]] = []
        used = 0
        for index, group in reversed(list(enumerate(groups))):
            cost = self.client._count_tokens(group, options)
            if kept and used + cost > budget:
                _LOGGER.debug("Dropping the %s oldest messages to fit the model context", index)
                break
            kept.insert(0, group)
            used += cost

        return [system_prompt, *[content for group in kept for content in group]]

    @staticmethod
    def _context_token_budget(options: Dict[str, Any]) -> int:
        """How much of the model context is left for the conversation itself."""
        context_length = int(options.get(CONF_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH))
        max_tokens = int(options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        return max(0, context_length - max_tokens - CONTEXT_BUDGET_MARGIN_TOKENS)

    @staticmethod
    def _group_tool_calls(content: List[conversation.Content]) -> List[List[conversation.Content]]:
        """Group each tool call with the results that follow it, so trimming keeps them together."""
        groups: List[List[conversation.Content]] = []
        for item in content:
            if isinstance(item, conversation.ToolResultContent) and groups:
                groups[-1].append(item)
            else:
                groups.append([item])
        return groups

    def _build_result(
        self,
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
        options: Dict[str, Any],
    ) -> ConversationResult:
        """Turn the finished chat log into the response for Home Assistant."""
        speech = self._get_speech(chat_log, options)

        if isinstance(chat_log.content[-1], conversation.AssistantContent):
            result = conversation.async_get_result_from_chat_log(user_input, chat_log)
            result.response.async_set_speech(speech or NO_SPEECH_MESSAGE)
            return result

        # Single shot models answer with the tool call itself, which leaves the turn ending on a tool
        # result instead of the assistant message the built in helper insists upon. The tool may have
        # answered with a card or its own text, which is what should be shown in that case.
        result = ConversationResult(
            response=self._get_tool_response(user_input, chat_log),
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )
        if speech is not None:
            result.response.async_set_speech(speech)
        elif not result.response.speech:
            result.response.async_set_speech(NO_SPEECH_MESSAGE)
        return result

    @staticmethod
    def _get_tool_response(user_input: ConversationInput, chat_log: conversation.ChatLog) -> intent.IntentResponse:
        """Return the response of the last tool that produced one, if any.

        Tools answering with an intent response (scripts, assistants, actions) carry their own card and
        response type, which is what Home Assistant should show rather than a plain text answer.
        """
        response: intent.IntentResponse | None = None
        for content in chat_log.content[chat_log.llm_input_provided_index:]:
            if isinstance(content, conversation.ToolResultContent) and isinstance(content.tool_result, llm.IntentResponseDict):
                response = getattr(content.tool_result, "original", None) or response

        return response or intent.IntentResponse(language=user_input.language)

    def _get_speech(self, chat_log: conversation.ChatLog, options: Dict[str, Any]) -> str | None:
        """Return what the model said this turn with any hidden thinking block removed.

        The model may answer with speech next to a tool call (single shot setups), wrap its reasoning in
        thinking blocks, or stay quiet after running tools, so the newest assistant message of this turn
        that carries text is what gets spoken.
        """
        think_prefix = options.get(CONF_THINKING_PREFIX, DEFAULT_THINKING_PREFIX)
        think_suffix = options.get(CONF_THINKING_SUFFIX, DEFAULT_THINKING_SUFFIX)

        for content in reversed(chat_log.content[chat_log.llm_input_provided_index:]):
            if isinstance(content, conversation.AssistantContent) and content.content:
                speech = strip_thinking_blocks(content.content, think_prefix, think_suffix)
                if speech:
                    return speech

        return None

    @staticmethod
    def _error_result(
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
        code: intent.IntentResponseErrorCode,
        message: str,
    ) -> ConversationResult:
        """Build an error response.

        The conversation id comes from the chat log because it is already assigned for the first message
        of a brand new conversation, where the user input does not carry one yet.
        """
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_error(code, message)
        return ConversationResult(response=response, conversation_id=chat_log.conversation_id)
