"""Config flow for Local LLaMA Conversation integration."""
from __future__ import annotations

import os
import logging
import types
import requests
import platform
from types import MappingProxyType
from typing import Any
from abc import ABC, abstractmethod

from huggingface_hub import hf_hub_download

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.requirements import pip_kwargs
from homeassistant.util.package import install_package, is_installed
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
    CONF_REQUEST_TIMEOUT,
    CONF_BACKEND_TYPE,
    CONF_BACKEND_TYPE_OPTIONS,
    CONF_DOWNLOADED_MODEL_FILE,
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
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_BACKEND_TYPE,
    BACKEND_TYPE_LLAMA_HF,
    BACKEND_TYPE_LLAMA_EXISTING,
    BACKEND_TYPE_REMOTE,
    DEFAULT_DOWNLOADED_MODEL_QUANTIZATION,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

def STEP_INIT_DATA_SCHEMA(backend_type=None):
    return vol.Schema(
        {
            vol.Required(CONF_BACKEND_TYPE, default=backend_type if backend_type else DEFAULT_BACKEND_TYPE): vol.In(CONF_BACKEND_TYPE_OPTIONS),
        }
    )

def STEP_LOCAL_SETUP_EXISTING_DATA_SCHEMA(model_file=None):
    return vol.Schema(
        {
            vol.Required(CONF_DOWNLOADED_MODEL_FILE, default=model_file if model_file else ""): str,
        }
    )

def STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA(*, chat_model=None, downloaded_model_quantization=None):
    return vol.Schema(
        {
            vol.Required(CONF_CHAT_MODEL, default=chat_model if chat_model else DEFAULT_CHAT_MODEL): str,
            vol.Required(CONF_DOWNLOADED_MODEL_QUANTIZATION, default=downloaded_model_quantization if downloaded_model_quantization else DEFAULT_DOWNLOADED_MODEL_QUANTIZATION): vol.In(CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS),
        }
    )

def STEP_REMOTE_SETUP_DATA_SCHEMA(*, host=None, port=None, chat_model=None):
    return vol.Schema(
        {
            vol.Required(CONF_HOST, default=host if host else DEFAULT_HOST): str,
            vol.Required(CONF_PORT, default=port if port else DEFAULT_PORT): str,
            vol.Required(CONF_CHAT_MODEL, default=chat_model if chat_model else DEFAULT_CHAT_MODEL): str,
        }
    )

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_TOP_K: DEFAULT_TOP_K,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_REQUEST_TIMEOUT: DEFAULT_REQUEST_TIMEOUT,
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
    
def install_llama_cpp_python(config_dir: str):
    try:
        if not is_installed("llama-cpp-python"):
            _LOGGER.info("Installing llama-cpp-python from wheel")
            platform_suffix = platform.machine()
            if platform_suffix == "arm64":
                platform_suffix = "aarch64"
            folder = os.path.dirname(__file__)
            potential_wheels = [ path for path in os.listdir(folder) if path.endswith(f"{platform_suffix}.whl") ]
            if len(potential_wheels) == 1:
                wheel_to_install = potential_wheels[0]
            else:
                _LOGGER.info("There are multiple potential wheels to install... Using the latest one")
                wheel_to_install = sorted(potential_wheels, reverse=True)[0]

            _LOGGER.debug("Wheel location: ", wheel_to_install)
            return install_package(os.path.join(folder, wheel_to_install), pip_kwargs(config_dir))
        else:
            _LOGGER.info("llama-cpp-python is already installed")
        return True
    except Exception as ex:
        _LOGGER.exception("Install failed!")
        return ex


class BaseLlamaConversationConfigFlow(FlowHandler, ABC):
    """Represent the base config flow for Z-Wave JS."""

    @property
    @abstractmethod
    def flow_manager(self) -> FlowManager:
        """Return the flow manager of the flow."""

    @abstractmethod
    async def async_step_install_local_wheels(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """ Install pre-built wheels """
    
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
    install_wheel_task = None
    install_wheel_error = None
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
        if self.download_task:
            self.download_task.cancel()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors = {}

        schema = STEP_INIT_DATA_SCHEMA()

        if user_input:
            try:
                local_backend = user_input[CONF_BACKEND_TYPE] != BACKEND_TYPE_REMOTE
                self.model_options.update(user_input)

            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                if local_backend:
                    for key, value in self.hass.data.get(DOMAIN, {}).items():
                        other_backend_type = value.data.get(CONF_BACKEND_TYPE)
                        if other_backend_type == BACKEND_TYPE_LLAMA_HF or \
                            other_backend_type == BACKEND_TYPE_LLAMA_EXISTING:
                            errors["base"] = "other_existing_local"
                            schema = STEP_INIT_DATA_SCHEMA(
                                backend_type=user_input[CONF_BACKEND_TYPE],
                            )

                    if "base" not in errors:
                        return await self.async_step_install_local_wheels()
                else:
                    return await self.async_step_remote_model()

        return self.async_show_form(
            step_id="user", data_schema=schema, errors=errors
        )
    
    async def async_step_install_local_wheels(
      self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if not user_input:
            if self.install_wheel_task:
                return self.async_show_progress(
                    step_id="install_local_wheels",
                    progress_action="install_local_wheels",
                )

            _LOGGER.debug("Queuing install task...")
            self.install_wheel_task = self.hass.async_add_executor_job(
                install_llama_cpp_python, self.hass.config.config_dir
            )

            self.hass.async_create_task(self._async_do_task(self.install_wheel_task))

            return self.async_show_progress(
                step_id="install_local_wheels",
                progress_action="install_local_wheels",
            )

        wheel_install_result = user_input["result"]
        if isinstance(wheel_install_result, Exception):
            _LOGGER.warning("Failed to install wheel: %s", repr(wheel_install_result))
            self.wheel_install_error = wheel_install_result
            return self.async_show_progress_done(next_step_id="user")
        else:
            _LOGGER.debug(f"Finished install")
            return self.async_show_progress_done(next_step_id="local_model")

    async def async_step_local_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        errors = {}

        backend_type = self.model_options[CONF_BACKEND_TYPE]
        if backend_type == BACKEND_TYPE_LLAMA_HF:
            schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA()
        elif backend_type == BACKEND_TYPE_LLAMA_EXISTING:
            schema = STEP_LOCAL_SETUP_EXISTING_DATA_SCHEMA()
        else:
            raise ValueError()

        if self.download_error:
            errors["base"] = "download_failed"
            schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA(
                chat_model=self.model_options[CONF_CHAT_MODEL],
                downloaded_model_quantization=self.model_options[CONF_DOWNLOADED_MODEL_QUANTIZATION]
            )

        if user_input:
            try:
                self.model_options.update(user_input)

            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                if backend_type == BACKEND_TYPE_LLAMA_HF:
                    return await self.async_step_download()
                else:
                    model_file = self.model_options[CONF_DOWNLOADED_MODEL_FILE]
                    if os.path.exists(model_file):
                        return await self.async_step_finish()
                    else:
                        errors["base"] = "missing_model_file"
                        schema = STEP_LOCAL_SETUP_EXISTING_DATA_SCHEMA(model_file)


        return self.async_show_form(
            step_id="local_model", data_schema=schema, errors=errors
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
            self.download_task = self.hass.async_add_executor_job(
                download_model_from_hf, model_name, quantization_type, storage_folder
            )

            self.hass.async_create_task(self._async_do_task(self.download_task))

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


    def _validate_remote_api(self) -> str:
        try:
            models_result = requests.get(f"http://{self.model_options[CONF_HOST]}:{self.model_options[CONF_PORT]}/v1/internal/model/list")
            models_result.raise_for_status()

            models = models_result.json()

            for model in models["model_names"]:
                if model == self.model_options[CONF_CHAT_MODEL].replace("/", "_"):
                    return ""

            return "missing_model_api"

        except Exception as ex:
            _LOGGER.info("Connection error was: %s", repr(ex))
            return "failed_to_connect"

    async def async_step_remote_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        errors = {}
        schema = STEP_REMOTE_SETUP_DATA_SCHEMA()

        if user_input:
            try:
                self.model_options.update(user_input)

                error_reason = await self.hass.async_add_executor_job(self._validate_remote_api)
                if error_reason:
                    errors["base"] = error_reason
                    schema = STEP_REMOTE_SETUP_DATA_SCHEMA(
                        host=user_input[CONF_HOST],
                        port=user_input[CONF_PORT],
                        chat_model=user_input[CONF_CHAT_MODEL],
                    )
                else:
                    return await self.async_step_finish()

            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="remote_model", data_schema=schema, errors=errors
        )

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:

        model_name = self.model_options.get(CONF_CHAT_MODEL)
        if not model_name:
            model_name = os.path.basename(self.model_options.get(CONF_DOWNLOADED_MODEL_FILE))
        location = "remote" if self.model_options[CONF_BACKEND_TYPE] == BACKEND_TYPE_REMOTE else "llama.cpp"

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
        is_local_backend = self.config_entry.data[CONF_BACKEND_TYPE] != BACKEND_TYPE_REMOTE
        schema = local_llama_config_option_schema(self.config_entry.options, is_local_backend)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


def local_llama_config_option_schema(options: MappingProxyType[str, Any], is_local_backend: bool) -> dict:
    """Return a schema for Local LLaMA completion options."""
    if not options:
        options = DEFAULT_OPTIONS
    result = {
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

    if not is_local_backend:
        result[vol.Optional(
            CONF_REQUEST_TIMEOUT,
            description={"suggested_value": options[CONF_REQUEST_TIMEOUT]},
            default=DEFAULT_REQUEST_TIMEOUT,
        )] = int

    return result
