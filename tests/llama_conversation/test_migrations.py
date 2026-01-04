"""Regression tests for config entry migration in __init__.py."""

import pytest

from homeassistant.const import CONF_LLM_HASS_API, CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.config_entries import ConfigSubentry
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
    CONF_PROMPT,
    CONF_REQUEST_TIMEOUT,
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
        if "options" in kwargs:
            cfg_entry._options = kwargs["options"]  # type: ignore[attr-defined]
        if "minor_version" in kwargs:
            cfg_entry._minor_version = kwargs["minor_version"]  # type: ignore[attr-defined]

    monkeypatch.setattr(hass.config_entries, "async_update_entry", fake_update_entry)

    result = await async_migrate_entry(hass, entry)

    assert result is True
    options_calls = [c for c in calls if "options" in c]
    assert options_calls, "async_update_entry should be called with options"
    assert options_calls[-1]["options"][CONF_LLM_HASS_API] == expected_list

    minor_calls = [c for c in calls if c.get("minor_version")]
    assert minor_calls and minor_calls[-1]["minor_version"] == 2
