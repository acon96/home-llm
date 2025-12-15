"""Tests for LocalLLMClient helpers in entity.py."""

import inspect
import json
import pytest
from json import JSONDecodeError

from custom_components.llama_conversation.entity import LocalLLMClient
from custom_components.llama_conversation.const import (
    CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
    CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    DEFAULT_TOOL_CALL_PREFIX,
    DEFAULT_TOOL_CALL_SUFFIX,
    DEFAULT_THINKING_PREFIX,
    DEFAULT_THINKING_SUFFIX,
)


class DummyLocalClient(LocalLLMClient):
    @staticmethod
    def get_name(_client_options):
        return "dummy"


class DummyLLMApi:
    def __init__(self):
        self.tools = []


@pytest.fixture
def client(hass):
    # Disable ICL loading during tests to avoid filesystem access.
    return DummyLocalClient(hass, {CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: False})


@pytest.mark.asyncio
async def test_async_parse_completion_parses_tool_call(client):
    raw_tool = '{"name":"light.turn_on","arguments":{"brightness":0.5,"to_say":" acknowledged"}}'
    completion = (
        f"{DEFAULT_THINKING_PREFIX}internal{DEFAULT_THINKING_SUFFIX}"
        f"hello {DEFAULT_TOOL_CALL_PREFIX}{raw_tool}{DEFAULT_TOOL_CALL_SUFFIX}"
    )

    result = await client._async_parse_completion(DummyLLMApi(), "agent-id", {}, completion)

    assert result.response.strip().startswith("hello")
    assert "acknowledged" in result.response
    assert result.tool_calls
    tool_call = result.tool_calls[0]
    assert tool_call.tool_name == "light.turn_on"
    assert tool_call.tool_args["brightness"] == 127


@pytest.mark.asyncio
async def test_async_parse_completion_ignores_tools_without_llm_api(client):
    raw_tool = '{"name":"light.turn_on","arguments":{"brightness":1}}'
    completion = f"hello {DEFAULT_TOOL_CALL_PREFIX}{raw_tool}{DEFAULT_TOOL_CALL_SUFFIX}"

    result = await client._async_parse_completion(None, "agent-id", {}, completion)

    assert result.tool_calls == []
    assert result.response.strip() == "hello"


@pytest.mark.asyncio
async def test_async_parse_completion_malformed_tool_raises(client):
    bad_tool = f"{DEFAULT_TOOL_CALL_PREFIX}{{not-json{DEFAULT_TOOL_CALL_SUFFIX}"

    with pytest.raises(JSONDecodeError):
        await client._async_parse_completion(DummyLLMApi(), "agent-id", {}, bad_tool)


@pytest.mark.asyncio
async def test_async_stream_parse_completion_handles_streamed_tool_call(client):
    async def token_generator():
        yield ("Hi", None)
        yield (
            None,
            [
                {
                    "function": {
                        "name": "light.turn_on",
                        "arguments": {"brightness": 0.25, "to_say": " ok"},
                    }
                }
            ],
        )

    stream = client._async_stream_parse_completion(
        DummyLLMApi(), "agent-id", {}, anext_token=token_generator()
    )

    results = [chunk async for chunk in stream]

    assert results[0].response == "Hi"
    assert results[1].response.strip() == "ok"
    assert results[1].tool_calls[0].tool_args["brightness"] == 63


@pytest.mark.asyncio
async def test_async_stream_parse_completion_malformed_tool_raises(client):
    async def token_generator():
        yield ("Hi", None)
        yield (None, ["{not-json"])

    with pytest.raises(JSONDecodeError):
        async for _chunk in client._async_stream_parse_completion(
            DummyLLMApi(), "agent-id", {}, anext_token=token_generator()
        ):
            pass


@pytest.mark.asyncio
async def test_async_stream_parse_completion_ignores_tools_without_llm_api(client):
    async def token_generator():
        yield ("Hi", None)
        yield (None, ["{}"])

    results = [chunk async for chunk in client._async_stream_parse_completion(
        None, "agent-id", {}, anext_token=token_generator()
    )]

    assert results[0].response == "Hi"
    assert results[1].tool_calls is None


@pytest.mark.asyncio
async def test_async_get_exposed_entities_respects_exposure(monkeypatch, client, hass):
    hass.states.async_set("light.exposed", "on", {"friendly_name": "Lamp"})
    hass.states.async_set("switch.hidden", "off", {"friendly_name": "Hidden"})

    monkeypatch.setattr(
        "custom_components.llama_conversation.entity.async_should_expose",
        lambda _hass, _domain, entity_id: not entity_id.endswith("hidden"),
    )

    exposed = client._async_get_exposed_entities()

    assert "light.exposed" in exposed
    assert "switch.hidden" not in exposed
    assert exposed["light.exposed"]["friendly_name"] == "Lamp"
    assert exposed["light.exposed"]["state"] == "on"


@pytest.mark.asyncio
async def test_generate_system_prompt_renders(monkeypatch, client, hass):
    hass.states.async_set("light.kitchen", "on", {"friendly_name": "Kitchen"})
    monkeypatch.setattr(
        "custom_components.llama_conversation.entity.async_should_expose",
        lambda _hass, _domain, _entity_id: True,
    )

    rendered = client._generate_system_prompt(
        "Devices:\n{{ formatted_devices }}",
        llm_api=None,
        entity_options={CONF_EXTRA_ATTRIBUTES_TO_EXPOSE: []},
    )
    if inspect.iscoroutine(rendered):
        rendered = await rendered

    assert isinstance(rendered, str)
    assert "light.kitchen" in rendered
