"""Regression tests for config entry migration in __init__.py."""

import dataclasses

import pytest

from homeassistant.const import CONF_LLM_HASS_API, CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.config_entries import ConfigSubentry
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.llama_conversation import async_migrate_entry
from custom_components.llama_conversation.const import (
    BACKEND_TYPE_LLAMA_CPP,
    BACKEND_TYPE_GENERIC_OPENAI,
    BACKEND_TYPE_LLAMA_CPP_SERVER,
    CONF_BACKEND_TYPE,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_LENGTH,
    CONF_DOWNLOADED_MODEL_FILE,
    CONF_DOWNLOADED_MODEL_QUANTIZATION,
    CONF_API_PATH,
    CONF_GBNF_GRAMMAR,
    CONF_PROMPT,
    CONF_REQUEST_TIMEOUT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
)


@pytest.mark.asyncio
async def test_migrate_v1_is_rejected(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_CPP}, version=1)
    entry.add_to_hass(hass)

    result = await async_migrate_entry(hass, entry)

    assert result is False


@pytest.mark.asyncio
async def test_migrate_v2_creates_subentry_and_updates_entry(monkeypatch, hass):
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="llama 'Test Agent' entry",
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI},
        options={
            CONF_HOST: "localhost",
            CONF_PORT: "8080",
            CONF_SSL: False,
            CONF_API_PATH: "v1",
            CONF_PROMPT: "hello",
            CONF_REQUEST_TIMEOUT: 90,
            CONF_CHAT_MODEL: "model-x",
            CONF_CONTEXT_LENGTH: 1024,
        },
        version=2,
    )
    entry.add_to_hass(hass)

    added_subentries = []
    update_calls = []

    def fake_add_subentry(cfg_entry, subentry):
        added_subentries.append((cfg_entry, subentry))

    def fake_update_entry(cfg_entry, **kwargs):
        update_calls.append(kwargs)

    monkeypatch.setattr(hass.config_entries, "async_add_subentry", fake_add_subentry)
    monkeypatch.setattr(hass.config_entries, "async_update_entry", fake_update_entry)

    result = await async_migrate_entry(hass, entry)

    assert result is True
    assert added_subentries, "Subentry should be added"
    subentry = added_subentries[0][1]
    assert isinstance(subentry, ConfigSubentry)
    assert subentry.subentry_type == "conversation"
    assert subentry.data[CONF_CHAT_MODEL] == "model-x"
    # Entry should be updated to version 3 with data/options separated
    assert any(call.get("version") == 3 for call in update_calls)
    last_options = [c["options"] for c in update_calls if "options" in c][-1]
    assert last_options[CONF_HOST] == "localhost"
    assert CONF_PROMPT not in last_options  # moved to subentry


@pytest.mark.asyncio
async def test_migrate_v3_minor0_downloads_model(monkeypatch, hass):
    sub_data = {
        CONF_CHAT_MODEL: "model-a",
        CONF_DOWNLOADED_MODEL_QUANTIZATION: "Q4_K_M",
        CONF_REQUEST_TIMEOUT: 30,
    }
    subentry = ConfigSubentry(data=sub_data, subentry_type="conversation", title="sub", unique_id=None)
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_CPP},
        options={},
        version=3,
        minor_version=0,
    )
    entry.subentries = {"sub": subentry}
    entry.add_to_hass(hass)

    updated_subentries = []
    update_calls = []

    def fake_update_subentry(cfg_entry, old_sub, *, data=None, **_kwargs):
        updated_subentries.append((cfg_entry, old_sub, data))

    def fake_update_entry(cfg_entry, **kwargs):
        update_calls.append(kwargs)

    monkeypatch.setattr(
        "custom_components.llama_conversation.download_model_from_hf", lambda *_args, **_kw: "file.gguf"
    )
    monkeypatch.setattr(hass.config_entries, "async_update_subentry", fake_update_subentry)
    monkeypatch.setattr(hass.config_entries, "async_update_entry", fake_update_entry)

    result = await async_migrate_entry(hass, entry)

    assert result is True
    assert updated_subentries, "Subentry should be updated with downloaded file"
    new_data = updated_subentries[0][2]
    assert new_data[CONF_DOWNLOADED_MODEL_FILE] == "file.gguf"
    assert any(call.get("minor_version") == 1 for call in update_calls)


@pytest.mark.parametrize(
    "api_value,expected_list",
    [("api-1", ["api-1"]), (None, [])],
)
@pytest.mark.asyncio
async def test_migrate_v3_minor1_converts_api_to_list(monkeypatch, hass, api_value, expected_list):
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI},
        options={CONF_LLM_HASS_API: api_value},
        version=3,
        minor_version=1,
    )
    entry.add_to_hass(hass)

    calls = []

    def fake_update_entry(cfg_entry, **kwargs):
        calls.append(kwargs)
        # apply it, like the real one does, so the next migration step sees the change
        for field, value in kwargs.items():
            object.__setattr__(cfg_entry, field, value)

    monkeypatch.setattr(hass.config_entries, "async_update_entry", fake_update_entry)

    result = await async_migrate_entry(hass, entry)

    assert result is True
    options_calls = [c for c in calls if "options" in c]
    assert options_calls, "async_update_entry should be called with options"
    assert options_calls[-1]["options"][CONF_LLM_HASS_API] == expected_list

    # the entry is stamped with v4 on the way, which keeps the converted list
    version_calls = [c for c in calls if c.get("version") == 4]
    assert version_calls, "the entry should end up on v4"
    assert version_calls[-1]["options"][CONF_LLM_HASS_API] == expected_list


def _fake_config_entries(monkeypatch, hass):
    """Make the migration able to chain: apply every update like the real thing does."""

    def fake_update_entry(cfg_entry, **kwargs):
        for field, value in kwargs.items():
            object.__setattr__(cfg_entry, field, value)

    def fake_update_subentry(cfg_entry, subentry, *, data=None, **_kwargs):
        cfg_entry.subentries[subentry.subentry_id] = dataclasses.replace(subentry, data=data)

    monkeypatch.setattr(hass.config_entries, "async_update_entry", fake_update_entry)
    monkeypatch.setattr(hass.config_entries, "async_update_subentry", fake_update_subentry)


@pytest.mark.asyncio
async def test_migrate_v4_drops_removed_options_and_reads_the_grammar(monkeypatch, hass):
    subentry = ConfigSubentry(
        data={
            CONF_CHAT_MODEL: "model-a",
            "gbnf_grammar": True,
            "gbnf_grammar_file": "output.gbnf",
            "refresh_prompt_per_turn": True,
            "remember_num_interactions": 5,
            CONF_CONTEXT_LENGTH: 2048,
        },
        subentry_id="sub",
        subentry_type="conversation",
        title="sub",
        unique_id=None,
    )
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_CPP},
        options={"remember_conversation": True, "use_server_sampling_defaults": False},
        version=3,
        minor_version=3,
    )
    entry.subentries = {"sub": subentry}
    entry.add_to_hass(hass)
    _fake_config_entries(monkeypatch, hass)

    assert await async_migrate_entry(hass, entry) is True

    assert (entry.version, entry.minor_version) == (4, 0)
    assert dict(entry.options) == {}

    new_data = dict(entry.subentries["sub"].data)
    # the grammar is the text itself now, taken from the file the entry pointed at
    assert new_data[CONF_GBNF_GRAMMAR].startswith("root") and "homeassistant" in new_data[CONF_GBNF_GRAMMAR]
    for key in ("gbnf_grammar", "gbnf_grammar_file", "refresh_prompt_per_turn", "remember_num_interactions"):
        assert key not in new_data
    assert new_data[CONF_CONTEXT_LENGTH] == 2048  # everything else is left alone

    # nothing was dropped silently, so there is nothing to tell the user about
    assert ir.async_get(hass).async_get_issue(DOMAIN, "text_generation_webui_removed") is None


@pytest.mark.asyncio
async def test_migrate_v4_converts_text_generation_webui(monkeypatch, hass):
    subentry = ConfigSubentry(
        data={
            CONF_CHAT_MODEL: "model-a",
            "text_generation_webui_preset": "my-preset",
            "text_generation_webui_chat_mode": "instruct",
            "text_generation_webui_admin_key": "secret",
        },
        subentry_id="sub",
        subentry_type="conversation",
        title="sub",
        unique_id=None,
    )
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BACKEND_TYPE: "text-generation-webui_api"},
        options={},
        version=3,
        minor_version=3,
    )
    entry.subentries = {"sub": subentry}
    entry.add_to_hass(hass)
    _fake_config_entries(monkeypatch, hass)

    assert await async_migrate_entry(hass, entry) is True

    assert entry.data[CONF_BACKEND_TYPE] == BACKEND_TYPE_GENERIC_OPENAI
    new_data = dict(entry.subentries["sub"].data)
    assert not any(key.startswith("text_generation_webui") for key in new_data)

    issue = ir.async_get(hass).async_get_issue(DOMAIN, "text_generation_webui_removed")
    assert issue is not None
    assert "my-preset" in issue.translation_placeholders["details"]
    assert "instruct" in issue.translation_placeholders["details"]


@pytest.mark.asyncio
async def test_migrate_v4_drops_sampling_the_entry_was_ignoring(monkeypatch, hass):
    subentry = ConfigSubentry(
        data={
            CONF_CHAT_MODEL: "model-a",
            CONF_TEMPERATURE: 0.2,
            CONF_TOP_P: 0.9,
            "use_server_sampling_defaults": True,
        },
        subentry_id="sub",
        subentry_type="conversation",
        title="sub",
        unique_id=None,
    )
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI},
        options={},
        version=3,
        minor_version=3,
    )
    entry.subentries = {"sub": subentry}
    entry.add_to_hass(hass)
    _fake_config_entries(monkeypatch, hass)

    assert await async_migrate_entry(hass, entry) is True

    # the entry never sent them, so it must not start sending them after the upgrade
    new_data = dict(entry.subentries["sub"].data)
    assert CONF_TEMPERATURE not in new_data
    assert CONF_TOP_P not in new_data
    assert new_data[CONF_CHAT_MODEL] == "model-a"
