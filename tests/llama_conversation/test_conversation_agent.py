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
@pytest.mark.parametrize(
    "refresh, initial_content, expect_generated, expect_first_content",
    [
        pytest.param(
            False,
            [UserContent(content="turn on the lights")],
            1,
            "rendered-system-prompt",
            id="first_turn_no_system_prompt",
        ),
        pytest.param(
            True,
            [SystemContent(content="old-system-prompt"), UserContent(content="turn on the lights")],
            1,
            "rendered-system-prompt",
            id="refresh_replaces_existing",
        ),
        pytest.param(
            False,
            [SystemContent(content="existing-system-prompt"), UserContent(content="turn on the lights")],
            0,
            "existing-system-prompt",
            id="no_refresh_keeps_existing",
        ),
    ],
)
async def test_system_prompt_injection(
    monkeypatch, hass, refresh, initial_content, expect_generated, expect_first_content,
):
    """Verify system prompt generation/replacement based on refresh setting
    and whether a SystemContent already exists in chat history."""
    agent, client = _make_agent(hass, {
        CONF_REFRESH_SYSTEM_PROMPT: refresh, CONF_REMEMBER_CONVERSATION: True,
    })
    chat_log = FakeChatLog()
    chat_log.content.extend(initial_content)
    _patch_chat(monkeypatch, chat_log)

    result = await agent.async_process(_make_user_input())

    assert result.response.speech["plain"]["speech"] == "hello from llm"
    assert len(client.generated_prompts) == expect_generated
    assert isinstance(chat_log.content[0], SystemContent)
    assert chat_log.content[0].content == expect_first_content
