"""Tests for integration entry setup and unload behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from homeassistant.const import CONF_HOST
from homeassistant.exceptions import ConfigEntryError

from custom_components.llama_conversation import async_setup_entry, async_unload_entry
from custom_components.llama_conversation.const import (
    BACKEND_TYPE_GENERIC_OPENAI,
    BACKEND_TYPE_LLAMA_CPP,
    CONF_BACKEND_TYPE,
    CONF_CHAT_MODEL,
    DOMAIN,
)


class DummyEntry:
    def __init__(self, *, entry_id: str, data: dict, options: dict):
        self.entry_id = entry_id
        self.data = data
        self.options = options
        self.runtime_data = None
        self.update_listener = None
        self.unload_callbacks = []

    def add_update_listener(self, callback):
        self.update_listener = callback
        return lambda: None

    def async_on_unload(self, callback):
        self.unload_callbacks.append(callback)


@pytest.mark.asyncio
async def test_async_setup_entry_registers_api_and_creates_client(monkeypatch, hass):
    created = {}
    registered = []
    forwarded = []

    class DummyClient:
        def __init__(self, _hass, options):
            created["options"] = options

        async def async_validate_startup(self, _entry):
            return None

    entry = DummyEntry(
        entry_id="entry-1",
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI},
        options={CONF_HOST: "localhost", CONF_CHAT_MODEL: "demo-model"},
    )

    async def fake_forward_entry_setups(cfg_entry, platforms):
        forwarded.append((cfg_entry, tuple(platforms)))

    monkeypatch.setattr(
        "custom_components.llama_conversation.llm.async_get_apis",
        lambda _hass: [],
    )
    monkeypatch.setattr(
        "custom_components.llama_conversation.llm.async_register_api",
        lambda _hass, api: registered.append(api),
    )
    monkeypatch.setattr(
        "custom_components.llama_conversation.BACKEND_TO_CLS",
        {BACKEND_TYPE_GENERIC_OPENAI: DummyClient},
    )
    monkeypatch.setattr(hass.config_entries, "async_forward_entry_setups", fake_forward_entry_setups)

    result = await async_setup_entry(hass, entry)

    assert result is True
    assert registered
    assert created["options"] == {
        CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI,
        CONF_HOST: "localhost",
        CONF_CHAT_MODEL: "demo-model",
    }
    assert hass.data[DOMAIN][entry.entry_id] is entry
    assert isinstance(entry.runtime_data, DummyClient)
    assert forwarded and forwarded[0][0] is entry
    assert entry.update_listener is not None
    assert entry.unload_callbacks


@pytest.mark.asyncio
async def test_async_setup_entry_raises_before_forwarding_when_startup_validation_fails(monkeypatch, hass):
    forwarded = []

    class DummyClient:
        def __init__(self, _hass, _options):
            pass

        async def async_validate_startup(self, _entry):
            raise ConfigEntryError("Unable to install package wheel: unexpected BufError")

    entry = DummyEntry(
        entry_id="entry-validate-fail",
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI},
        options={CONF_HOST: "localhost", CONF_CHAT_MODEL: "demo-model"},
    )

    async def fake_forward_entry_setups(cfg_entry, platforms):
        forwarded.append((cfg_entry, tuple(platforms)))

    monkeypatch.setattr(
        "custom_components.llama_conversation.llm.async_get_apis",
        lambda _hass: [],
    )
    monkeypatch.setattr(
        "custom_components.llama_conversation.llm.async_register_api",
        lambda _hass, api: None,
    )
    monkeypatch.setattr(
        "custom_components.llama_conversation.BACKEND_TO_CLS",
        {BACKEND_TYPE_GENERIC_OPENAI: DummyClient},
    )
    monkeypatch.setattr(hass.config_entries, "async_forward_entry_setups", fake_forward_entry_setups)

    with pytest.raises(ConfigEntryError, match="unexpected BufError"):
        await async_setup_entry(hass, entry)

    assert not forwarded
    assert entry.entry_id not in hass.data.get(DOMAIN, {})


@pytest.mark.asyncio
async def test_async_unload_entry_cleans_llama_cpp_cache(monkeypatch, hass, tmp_path):
    removed_paths = []
    hass.data.setdefault(DOMAIN, {})["entry-1"] = object()
    hass.config.media_dirs = {"local": str(tmp_path)}
    entry = DummyEntry(
        entry_id="entry-1",
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_CPP},
        options={CONF_CHAT_MODEL: "My Model"},
    )

    async def fake_unload_platforms(_entry, _platforms):
        return True

    monkeypatch.setattr(hass.config_entries, "async_unload_platforms", fake_unload_platforms)
    monkeypatch.setattr(
        "custom_components.llama_conversation.shutil.rmtree",
        lambda path, ignore_errors=True: removed_paths.append((path, ignore_errors)),
    )

    result = await async_unload_entry(hass, entry)

    assert result is True
    assert removed_paths == [(str(tmp_path / "kv_cache" / "my_model"), True)]
    assert entry.entry_id not in hass.data[DOMAIN]


@pytest.mark.asyncio
async def test_async_unload_entry_leaves_data_when_platform_unload_fails(monkeypatch, hass):
    sentinel = object()
    hass.data.setdefault(DOMAIN, {})["entry-2"] = sentinel
    entry = DummyEntry(
        entry_id="entry-2",
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI},
        options={},
    )

    async def fake_unload_platforms(_entry, _platforms):
        return False

    monkeypatch.setattr(hass.config_entries, "async_unload_platforms", fake_unload_platforms)

    result = await async_unload_entry(hass, entry)

    assert result is False
    assert hass.data[DOMAIN][entry.entry_id] is sentinel