"""Config flow for Local LLaMA Conversation integration."""
from __future__ import annotations

import os
import logging
import types
from types import MappingProxyType
from typing import Any
from abc import ABC, abstractmethod

from huggingface_hub import hf_hub_download

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.data_entry_flow import (
    AbortFlow,
    FlowHandler,
    FlowManager,
    FlowResult,
)
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    TemplateSelector,
)

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_USE_LOCAL_BACKEND,
    CONF_DOWNLOADED_MODEL_FILE,
    CONF_DOWNLOAD_MODEL_FROM_HF,
    CONF_DOWNLOADED_MODEL_QUANTIZATION,
    CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_USE_LOCAL_BACKEND,
    DEFAULT_DOWNLOAD_MODEL_FROM_HF,
    DEFAULT_DOWNLOADED_MODEL_QUANTIZATION,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

STEP_INIT_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_CHAT_MODEL, default=DEFAULT_CHAT_MODEL): str,
        vol.Required(CONF_USE_LOCAL_BACKEND, default=DEFAULT_USE_LOCAL_BACKEND): bool,
    }
)

STEP_LOCAL_SETUP_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_DOWNLOAD_MODEL_FROM_HF, default=DEFAULT_DOWNLOAD_MODEL_FROM_HF): bool,
        vol.Required(CONF_DOWNLOADED_MODEL_QUANTIZATION, default=DEFAULT_DOWNLOADED_MODEL_QUANTIZATION): vol.In(CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS),
        vol.Optional(CONF_DOWNLOADED_MODEL_FILE, description="Leave blank if downloading from HF. Otherwise set to the path of the model file."): str,
    }
)
STEP_REMOTE_SETUP_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_HOST, default=DEFAULT_HOST): str,
        vol.Optional(CONF_PORT, default=DEFAULT_PORT): str,
    }
)

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_TOP_K: DEFAULT_TOP_K,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
    }
)

def download_model_from_hf(
    model_name: str, quantization_type: str, storage_folder: str
):
    try:
        expected_filename = (
            model_name.split("/")[1].removesuffix("-GGUF") + f".{quantization_type}.gguf"
        )
        os.makedirs(storage_folder, exist_ok=True)

        return hf_hub_download(
            repo_id=model_name,
            repo_type="model",
            filename=expected_filename,
            resume_download=True,
            cache_dir=storage_folder,
        )
    except Exception as ex:
        return ex

class BaseLlamaConversationConfigFlow(FlowHandler, ABC):
    """Represent the base config flow for Z-Wave JS."""

    @property
    @abstractmethod
    def flow_manager(self) -> FlowManager:
        """Return the flow manager of the flow."""

    @abstractmethod
    async def async_step_local_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """ Configure a local model """

    @abstractmethod
    async def async_step_remote_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """ Configure a remote model """

    @abstractmethod
    async def async_step_download(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """ Download a model from HF """

    @abstractmethod
    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """ Finish configuration """

class ConfigFlow(BaseLlamaConversationConfigFlow, config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Local LLaMA Conversation."""

    VERSION = 1
    download_task = None
    download_error = None
    model_options: dict[str, Any] = {}

    @property
    def flow_manager(self) -> config_entries.ConfigEntriesFlowManager:
        """Return the correct flow manager."""
        return self.hass.config_entries.flow

    async def _async_do_task(self, task):
        result = await task  # A task that take some time to complete.

        # Continue the flow after show progress when the task is done.
        # To avoid a potential deadlock we create a new task that continues the flow.
        # The task must be completely done so the flow can await the task
        # if needed and get the task result.
        self.hass.async_create_task(
            self.hass.config_entries.flow.async_configure(flow_id=self.flow_id, user_input={"result": result })
        )

    def async_remove(self) -> None:
        self.download_task.cancel()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_INIT_DATA_SCHEMA
            )

        errors = {}

        try:
            local_backend = user_input[CONF_USE_LOCAL_BACKEND]
            self.model_options.update(user_input)

        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            if local_backend:
                return await self.async_step_local_model()
            else:
                return await self.async_step_remote_model()

        return self.async_show_form(
            step_id="user", data_schema=STEP_INIT_DATA_SCHEMA, errors=errors
        )

    async def async_step_local_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""

        errors = {}

        if self.download_error:
            errors["base"] = "download_failed"

        if user_input is None:
            return self.async_show_form(
                step_id="local_model", data_schema=STEP_LOCAL_SETUP_DATA_SCHEMA, errors=errors
            )

        try:
            self.model_options.update(user_input)

        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            if self.model_options[CONF_DOWNLOAD_MODEL_FROM_HF]:
                return await self.async_step_download()
            elif not self.model_options.__contains__(CONF_DOWNLOADED_MODEL_FILE):
                errors["base"] = "missing_file_name"
            else:
                return await self.async_step_finish()


        return self.async_show_form(
            step_id="local_model", data_schema=STEP_LOCAL_SETUP_DATA_SCHEMA, errors=errors
        )

    async def async_step_download(
      self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if not user_input:
            if self.download_task:
                return self.async_show_progress(
                    step_id="download",
                    progress_action="download",
                )

            model_name = self.model_options[CONF_CHAT_MODEL]
            quantization_type = self.model_options[CONF_DOWNLOADED_MODEL_QUANTIZATION]

            storage_folder = os.path.join(self.hass.config.media_dirs["local"], "models")
            executor_job = self.hass.async_add_executor_job(
                download_model_from_hf, model_name, quantization_type, storage_folder
            )

            self.download_task = self.hass.async_create_task(self._async_do_task(executor_job))

            return self.async_show_progress(
                step_id="download",
                progress_action="download",
            )

        download_result = user_input["result"]
        if isinstance(download_result, Exception):
            _LOGGER.info("Failed to download model: %s", repr(download_result))
            self.download_error = download_result
            return self.async_show_progress_done(next_step_id="local_model")
        else:
            self.model_options[CONF_DOWNLOADED_MODEL_FILE] = download_result
            return self.async_show_progress_done(next_step_id="finish")


    async def async_step_remote_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="remote_model", data_schema=STEP_REMOTE_SETUP_DATA_SCHEMA
            )

        errors = {}

        try:
            self.model_options.update(user_input)

        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return await self.async_step_finish()

        return self.async_show_form(
            step_id="remote_model", data_schema=STEP_REMOTE_SETUP_DATA_SCHEMA, errors=errors
        )

    async def async_step_finish(self, user_input=None):

        model_name = self.model_options[CONF_CHAT_MODEL]
        location = "llama.cpp" if self.model_options[CONF_USE_LOCAL_BACKEND] else "remote"

        return self.async_create_entry(
            title=f"LLM Model '{model_name}' ({location})",
            description="A Transformers Model Agent",
            data=self.model_options,
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlow(config_entry)


class OptionsFlow(config_entries.OptionsFlow):
    """Local LLaMA config flow options handler."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="LLaMA Conversation", data=user_input)
        schema = local_llama_config_option_schema(self.config_entry.options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


def local_llama_config_option_schema(options: MappingProxyType[str, Any]) -> dict:
    """Return a schema for Local LLaMA completion options."""
    if not options:
        options = DEFAULT_OPTIONS
    return {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options[CONF_PROMPT]},
            default=DEFAULT_PROMPT,
        ): TemplateSelector(),
        vol.Optional(
            CONF_MAX_TOKENS,
            description={"suggested_value": options[CONF_MAX_TOKENS]},
            default=DEFAULT_MAX_TOKENS,
        ): int,
        vol.Optional(
            CONF_TOP_K,
            description={"suggested_value": options[CONF_TOP_K]},
            default=DEFAULT_TOP_K,
        ): NumberSelector(NumberSelectorConfig(min=1, max=256, step=1)),
        vol.Optional(
            CONF_TOP_P,
            description={"suggested_value": options[CONF_TOP_P]},
            default=DEFAULT_TOP_P,
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        vol.Optional(
            CONF_TEMPERATURE,
            description={"suggested_value": options[CONF_TEMPERATURE]},
            default=DEFAULT_TEMPERATURE,
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
    }
