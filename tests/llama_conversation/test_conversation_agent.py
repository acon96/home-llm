"""Tests for LocalLLMAgent async_process."""

import pytest
from contextlib import contextmanager

from homeassistant.components.conversation import ConversationInput, SystemContent, AssistantContent, UserContent
from homeassistant.const import MATCH_ALL

from custom_components.llama_conversation.conversation import LocalLLMAgent
from custom_components.llama_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    DEFAULT_PROMPT,
    DOMAIN,
)


class DummyClient:
    def __init__(self, hass):
        self.hass = hass
        self.generated_prompts = []

    def _generate_system_prompt(self, prompt_template, llm_api, entity_options):
        self.generated_prompts.append(prompt_template)
        return "rendered-system-prompt"

    async def _async_generate(self, conv, agent_id, chat_log, entity_options):
        async def gen():
            yield AssistantContent(agent_id=agent_id, content="hello from llm")
        return gen()


class DummySubentry:
    def __init__(self, subentry_id="sub1", title="Test Agent", chat_model="model"):
        self.subentry_id = subentry_id
        self.title = title
        self.subentry_type = DOMAIN
        self.data = {CONF_CHAT_MODEL: chat_model}


class DummyEntry:
    def __init__(self, entry_id="entry1", options=None, subentry=None, runtime_data=None):
        self.entry_id = entry_id
        self.options = options or {}
        self.subentries = {subentry.subentry_id: subentry}
        self.runtime_data = runtime_data

    def add_update_listener(self, _cb):
        return lambda: None


class FakeChatLog:
    def __init__(self):
        self.content = []
        self.llm_api = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeChatSession:
    def __enter__(self):
        return {}

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_async_process_generates_response(monkeypatch, hass):
    client = DummyClient(hass)
    subentry = DummySubentry()
    entry = DummyEntry(subentry=subentry, runtime_data=client)

    # Make entry discoverable through hass data as LocalLLMEntity expects.
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry

    @contextmanager
    def fake_chat_session(_hass, _conversation_id):
        yield FakeChatSession()

    @contextmanager
    def fake_chat_log(_hass, _session, _user_input):
        yield FakeChatLog()

    monkeypatch.setattr(
        "custom_components.llama_conversation.conversation.chat_session.async_get_chat_session",
        fake_chat_session,
    )
    monkeypatch.setattr(
        "custom_components.llama_conversation.conversation.conversation.async_get_chat_log",
        fake_chat_log,
    )

    agent = LocalLLMAgent(hass, entry, subentry, client)

    result = await agent.async_process(
        ConversationInput(
            text="turn on the lights",
            context=None,
            conversation_id="conv-id",
            device_id=None,
            language="en",
            agent_id="agent-1",
        )
    )

    assert result.response.speech["plain"]["speech"] == "hello from llm"
    # System prompt should be rendered once when message history is empty.
    assert client.generated_prompts == [DEFAULT_PROMPT]
    assert agent.supported_languages == MATCH_ALL


def _make_user_input(text="turn on the lights"):
    return ConversationInput(
        text=text, context=None, conversation_id="conv-id",
        device_id=None, language="en", agent_id="agent-1",
    )


def _patch_chat(monkeypatch, chat_log):
    """Patch chat_session and chat_log context managers for agent tests."""
    @contextmanager
    def fake_chat_session(_hass, _conversation_id):
        yield FakeChatSession()

    @contextmanager
    def fake_chat_log_cm(_hass, _session, _user_input):
        yield chat_log

    monkeypatch.setattr(
        "custom_components.llama_conversation.conversation.chat_session.async_get_chat_session",
        fake_chat_session,
    )
    monkeypatch.setattr(
        "custom_components.llama_conversation.conversation.conversation.async_get_chat_log",
        fake_chat_log_cm,
    )


def _make_agent(hass, subentry_data_overrides=None):
    client = DummyClient(hass)
    subentry = DummySubentry()
    if subentry_data_overrides:
        subentry.data = {**subentry.data, **subentry_data_overrides}
    entry = DummyEntry(subentry=subentry, runtime_data=client)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry
    return LocalLLMAgent(hass, entry, subentry, client), client


@pytest.mark.asyncio
async def test_system_prompt_generated_when_chat_log_has_user_content(monkeypatch, hass):
    """System prompt must be generated on the first turn even when chat_log
    already contains a UserContent (added by HA before async_process runs)
    and refresh_system_prompt is False."""
    agent, client = _make_agent(hass, {
        CONF_REFRESH_SYSTEM_PROMPT: False, CONF_REMEMBER_CONVERSATION: True,
    })
    chat_log = FakeChatLog()
    chat_log.content.append(UserContent(content="turn on the lights"))
    _patch_chat(monkeypatch, chat_log)

    result = await agent.async_process(_make_user_input())

    assert result.response.speech["plain"]["speech"] == "hello from llm"
    assert len(client.generated_prompts) == 1
    assert isinstance(chat_log.content[0], SystemContent)


@pytest.mark.asyncio
async def test_system_prompt_regenerated_when_refresh_enabled(monkeypatch, hass):
    """When refresh_system_prompt is True, the system prompt should be
    regenerated even if one already exists in the history."""
    agent, client = _make_agent(hass, {
        CONF_REFRESH_SYSTEM_PROMPT: True, CONF_REMEMBER_CONVERSATION: True,
    })
    chat_log = FakeChatLog()
    chat_log.content.append(SystemContent(content="old-system-prompt"))
    chat_log.content.append(UserContent(content="turn on the lights"))
    _patch_chat(monkeypatch, chat_log)

    result = await agent.async_process(_make_user_input())

    assert result.response.speech["plain"]["speech"] == "hello from llm"
    assert len(client.generated_prompts) == 1
    assert client.generated_prompts[0] == DEFAULT_PROMPT


@pytest.mark.asyncio
async def test_system_prompt_not_regenerated_when_refresh_disabled(monkeypatch, hass):
    """When refresh_system_prompt is False and a SystemContent already exists,
    the system prompt should NOT be regenerated."""
    agent, client = _make_agent(hass, {
        CONF_REFRESH_SYSTEM_PROMPT: False, CONF_REMEMBER_CONVERSATION: True,
    })
    chat_log = FakeChatLog()
    chat_log.content.append(SystemContent(content="existing-system-prompt"))
    chat_log.content.append(UserContent(content="turn on the lights"))
    _patch_chat(monkeypatch, chat_log)

    result = await agent.async_process(_make_user_input())

    assert result.response.speech["plain"]["speech"] == "hello from llm"
    assert len(client.generated_prompts) == 0
    assert isinstance(chat_log.content[0], SystemContent)
    assert chat_log.content[0].content == "existing-system-prompt"
