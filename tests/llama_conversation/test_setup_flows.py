"""Behavior-driven tests for options flows and subentry setup flows."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_HOST, CONF_LLM_HASS_API, CONF_PORT, CONF_SSL
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.llama_conversation.config_flow import LocalLLMSubentryFlowHandler, OptionsFlow
from custom_components.llama_conversation.const import (
    BACKEND_TYPE_GENERIC_OPENAI,
    BACKEND_TYPE_LLAMA_CPP,
    CONF_AI_TASK_RETRIES,
    CONF_BACKEND_TYPE,
    CONF_CHAT_MODEL,
    CONF_DOWNLOADED_MODEL_FILE,
    CONF_DOWNLOADED_MODEL_QUANTIZATION,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_INSTALLED_LLAMACPP_VERSION,
    CONF_LLAMACPP_REINSTALL,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_PROMPT,
    CONF_REQUEST_TIMEOUT,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_SELECTED_LANGUAGE,
    CONF_USE_GBNF_GRAMMAR,
    DOMAIN,
)


def _build_options_flow(hass, entry) -> OptionsFlow:
    flow = OptionsFlow()
    flow.hass = hass
    flow.handler = entry.entry_id
    flow.context = {"source": "options"}
    entry.add_to_hass(hass)
    return flow


def _build_subentry_flow(
    hass,
    entry,
    *,
    source: str = "user",
    subentry_type: str = conversation.DOMAIN,
    reconfigure_subentry=None,
) -> LocalLLMSubentryFlowHandler:
    flow = LocalLLMSubentryFlowHandler()
    flow.hass = hass
    flow.context = {"source": source}
    flow.handler = (DOMAIN, subentry_type)
    flow._get_entry = lambda: entry
    if reconfigure_subentry is not None:
        flow._get_reconfigure_subentry = lambda: reconfigure_subentry
    return flow


class DummyRuntimeData:
    def __init__(self, models: list[str] | None = None):
        self.models = models or ["demo-model"]

    async def async_get_available_models(self) -> list[str]:
        return self.models


@pytest.mark.asyncio
async def test_options_flow_remote_backend_surfaces_connection_error(monkeypatch, hass):
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI},
        options={CONF_HOST: "old-host", CONF_PORT: "8080", CONF_SSL: False},
    )
    flow = _build_options_flow(hass, entry)

    async def fake_validate_connection(_hass, _config):
        return RuntimeError("refused")

    monkeypatch.setattr(
        "custom_components.llama_conversation.config_flow.BACKEND_TO_CLS",
        {BACKEND_TYPE_GENERIC_OPENAI: type("Backend", (), {"async_validate_connection": staticmethod(fake_validate_connection)})},
    )

    result = await flow.async_step_init(
        {CONF_HOST: "new-host", CONF_PORT: "8081", CONF_SSL: True}
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "failed_to_connect"
    assert result["description_placeholders"]["exception"] == "refused"


@pytest.mark.asyncio
async def test_options_flow_remote_backend_saves_valid_connection(monkeypatch, hass):
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI},
        options={CONF_HOST: "old-host", CONF_PORT: "8080", CONF_SSL: False},
    )
    flow = _build_options_flow(hass, entry)

    async def fake_validate_connection(_hass, _config):
        return None

    monkeypatch.setattr(
        "custom_components.llama_conversation.config_flow.BACKEND_TO_CLS",
        {BACKEND_TYPE_GENERIC_OPENAI: type("Backend", (), {"async_validate_connection": staticmethod(fake_validate_connection)})},
    )

    result = await flow.async_step_init(
        {CONF_HOST: "new-host", CONF_PORT: "8081", CONF_SSL: True}
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_HOST] == "new-host"
    assert result["data"][CONF_PORT] == "8081"
    assert result["data"][CONF_SSL] is True


@pytest.mark.asyncio
async def test_options_flow_llama_cpp_reinstall_failure_aborts(monkeypatch, hass):
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_CPP},
        options={CONF_INSTALLED_LLAMACPP_VERSION: "0.2.0"},
    )
    flow = _build_options_flow(hass, entry)

    async def fake_versions(_hass):
        return [("0.2.0", True)]

    monkeypatch.setattr(
        "custom_components.llama_conversation.config_flow.get_available_llama_cpp_versions",
        fake_versions,
    )
    monkeypatch.setattr(
        "custom_components.llama_conversation.config_flow.install_llama_cpp_python",
        lambda _config_dir, _force, _version: False,
    )
    monkeypatch.setattr(
        hass,
        "async_create_background_task",
        lambda coro, *, name=None: asyncio.create_task(coro, name=name),
    )

    initial = await flow.async_step_init()
    assert initial["type"] == FlowResultType.FORM

    started = await flow.async_step_reinstall(
        {CONF_LLAMACPP_REINSTALL: True, CONF_INSTALLED_LLAMACPP_VERSION: "0.2.0"}
    )
    assert started["type"] == FlowResultType.SHOW_PROGRESS

    await flow.reinstall_task
    finished = await flow.async_step_reinstall()
    assert finished["type"] == FlowResultType.SHOW_PROGRESS_DONE

    result = await flow.async_step_init()
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "pip_wheel_error"


@pytest.mark.asyncio
async def test_options_flow_llama_cpp_reinstall_success_updates_version(monkeypatch, hass):
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_CPP},
        options={CONF_INSTALLED_LLAMACPP_VERSION: "0.2.0"},
    )
    flow = _build_options_flow(hass, entry)

    async def fake_versions(_hass):
        return [("0.2.0", True), ("0.3.0", False)]

    monkeypatch.setattr(
        "custom_components.llama_conversation.config_flow.get_available_llama_cpp_versions",
        fake_versions,
    )
    monkeypatch.setattr(
        "custom_components.llama_conversation.config_flow.install_llama_cpp_python",
        lambda _config_dir, _force, _version: True,
    )
    monkeypatch.setattr(
        "custom_components.llama_conversation.config_flow.get_llama_cpp_python_version",
        lambda: "0.3.0",
    )
    monkeypatch.setattr(
        hass,
        "async_create_background_task",
        lambda coro, *, name=None: asyncio.create_task(coro, name=name),
    )

    await flow.async_step_init()
    await flow.async_step_reinstall(
        {CONF_LLAMACPP_REINSTALL: True, CONF_INSTALLED_LLAMACPP_VERSION: "0.3.0"}
    )
    await flow.reinstall_task
    await flow.async_step_reinstall()

    result = await flow.async_step_init()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_INSTALLED_LLAMACPP_VERSION] == "0.3.0"


@pytest.mark.asyncio
async def test_subentry_flow_rejects_unloaded_parent_entry(hass):
    entry = SimpleNamespace(state=ConfigEntryState.SETUP_ERROR)
    flow = _build_subentry_flow(hass, entry)

    result = await flow.async_step_user()

    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "entry_not_loaded"


@pytest.mark.asyncio
async def test_subentry_flow_local_missing_model_file_returns_error(hass):
    entry = SimpleNamespace(
        state=ConfigEntryState.LOADED,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_CPP},
        options={CONF_SELECTED_LANGUAGE: "en"},
        runtime_data=DummyRuntimeData(),
    )
    flow = _build_subentry_flow(hass, entry)

    result = await flow.async_step_user(
        {
            CONF_CHAT_MODEL: "",
            CONF_DOWNLOADED_MODEL_QUANTIZATION: "Q4_K_M",
            CONF_DOWNLOADED_MODEL_FILE: "/tmp/missing-model.gguf",
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "pick_model"
    assert result["errors"]["base"] == "missing_model_file"


@pytest.mark.asyncio
async def test_subentry_flow_model_parameters_rejects_missing_gbnf_file(hass):
    entry = SimpleNamespace(
        state=ConfigEntryState.LOADED,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_LLAMA_CPP},
        options={CONF_SELECTED_LANGUAGE: "en"},
        runtime_data=DummyRuntimeData(),
    )
    flow = _build_subentry_flow(hass, entry)
    flow.model_config = {
        CONF_CHAT_MODEL: "demo-model",
        CONF_DOWNLOADED_MODEL_FILE: "/tmp/demo-model.gguf",
    }

    result = await flow.async_step_model_parameters(
        {
            CONF_USE_GBNF_GRAMMAR: True,
            CONF_GBNF_GRAMMAR_FILE: "missing.gbnf",
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "model_parameters"
    assert result["errors"]["base"] == "missing_gbnf_file"
    assert result["description_placeholders"]["filename"] == "missing.gbnf"


@pytest.mark.asyncio
async def test_subentry_flow_model_parameters_normalizes_int_fields(hass):
    entry = SimpleNamespace(
        state=ConfigEntryState.LOADED,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI},
        options={CONF_SELECTED_LANGUAGE: "en"},
        runtime_data=DummyRuntimeData(["model-a"]),
    )
    flow = _build_subentry_flow(hass, entry)
    flow.model_config = {CONF_CHAT_MODEL: "model-a"}

    result = await flow.async_step_model_parameters(
        {
            CONF_LLM_HASS_API: [],
            CONF_REMEMBER_NUM_INTERACTIONS: "3",
            CONF_MAX_TOOL_CALL_ITERATIONS: "2",
            CONF_REQUEST_TIMEOUT: "45",
        }
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_REMEMBER_NUM_INTERACTIONS] == 3
    assert result["data"][CONF_MAX_TOOL_CALL_ITERATIONS] == 2
    assert result["data"][CONF_REQUEST_TIMEOUT] == 45


@pytest.mark.asyncio
async def test_subentry_reconfigure_updates_existing_subentry(monkeypatch, hass):
    entry = SimpleNamespace(
        state=ConfigEntryState.LOADED,
        data={CONF_BACKEND_TYPE: BACKEND_TYPE_GENERIC_OPENAI},
        options={CONF_SELECTED_LANGUAGE: "en"},
        runtime_data=DummyRuntimeData(["model-a"]),
    )
    existing_subentry = SimpleNamespace(
        data={
            CONF_CHAT_MODEL: "model-a",
            CONF_PROMPT: "old prompt",
            CONF_REQUEST_TIMEOUT: 15,
            CONF_AI_TASK_RETRIES: 0,
        }
    )
    flow = _build_subentry_flow(
        hass,
        entry,
        source="reconfigure",
        subentry_type=ai_task.DOMAIN,
        reconfigure_subentry=existing_subentry,
    )
    captured = {}

    def fake_update_and_abort(cfg_entry, subentry, **kwargs):
        captured["entry"] = cfg_entry
        captured["subentry"] = subentry
        captured.update(kwargs)
        return {"type": FlowResultType.ABORT, "reason": "reconfigured"}

    monkeypatch.setattr(flow, "async_update_and_abort", fake_update_and_abort)

    result = await flow.async_step_reconfigure(
        {
            CONF_PROMPT: "new prompt",
            CONF_REQUEST_TIMEOUT: "30",
            CONF_AI_TASK_RETRIES: "2",
        }
    )

    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reconfigured"
    assert captured["entry"] is entry
    assert captured["subentry"] is existing_subentry
    assert captured["data"][CONF_PROMPT] == "new prompt"
    assert captured["data"][CONF_REQUEST_TIMEOUT] == 30
    assert captured["data"][CONF_AI_TASK_RETRIES] == 2