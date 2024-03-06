"""Config flow for Local LLaMA Conversation integration."""
from __future__ import annotations

import os
import sys
import logging
import requests
from types import MappingProxyType
from typing import Any
from abc import ABC, abstractmethod

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL, UnitOfTime
from homeassistant.data_entry_flow import (
    AbortFlow,
    FlowHandler,
    FlowManager,
    FlowResult,
)
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TemplateSelector,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
)

from .utils import download_model_from_hf, install_llama_cpp_python
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_REQUEST_TIMEOUT,
    CONF_BACKEND_TYPE,
    CONF_DOWNLOADED_MODEL_FILE,
    CONF_DOWNLOADED_MODEL_QUANTIZATION,
    CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS,
    CONF_PROMPT_TEMPLATE,
    CONF_USE_GBNF_GRAMMAR,
    CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
    CONF_TEXT_GEN_WEBUI_PRESET,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_OPENAI_API_KEY,
    CONF_TEXT_GEN_WEBUI_ADMIN_KEY,
    CONF_SERVICE_CALL_REGEX,
    CONF_REMOTE_USE_CHAT_ENDPOINT,
    CONF_TEXT_GEN_WEBUI_CHAT_MODE,
    CONF_OLLAMA_KEEP_ALIVE_MIN,
    DEFAULT_CHAT_MODEL,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_SSL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_BACKEND_TYPE,
    DEFAULT_DOWNLOADED_MODEL_QUANTIZATION,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_USE_GBNF_GRAMMAR,
    DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_SERVICE_CALL_REGEX,
    DEFAULT_OPTIONS,
    DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
    DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE,
    DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
    BACKEND_TYPE_LLAMA_HF,
    BACKEND_TYPE_LLAMA_EXISTING,
    BACKEND_TYPE_TEXT_GEN_WEBUI,
    BACKEND_TYPE_GENERIC_OPENAI,
    BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER,
    BACKEND_TYPE_OLLAMA,
    PROMPT_TEMPLATE_DESCRIPTIONS,
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT,
    TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT,
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

def is_local_backend(backend):
    return backend in [BACKEND_TYPE_LLAMA_EXISTING, BACKEND_TYPE_LLAMA_HF]

def STEP_INIT_DATA_SCHEMA(backend_type=None):
    return vol.Schema(
        {
            vol.Required(
                CONF_BACKEND_TYPE,
                default=backend_type if backend_type else DEFAULT_BACKEND_TYPE
            ): SelectSelector(SelectSelectorConfig(
                options=[ 
                    BACKEND_TYPE_LLAMA_HF, BACKEND_TYPE_LLAMA_EXISTING,
                    BACKEND_TYPE_TEXT_GEN_WEBUI,
                    BACKEND_TYPE_GENERIC_OPENAI,
                    BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER,
                    BACKEND_TYPE_OLLAMA
                ],
                translation_key=CONF_BACKEND_TYPE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            ))
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

def STEP_REMOTE_SETUP_DATA_SCHEMA(backend_type: str, *, host=None, port=None, ssl=None, chat_model=None, use_chat_endpoint=None, webui_preset="", webui_chat_mode=""):

    extra1, extra2 = ({}, {})
    default_port = DEFAULT_PORT

    if backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI: 
        extra1[vol.Optional(CONF_TEXT_GEN_WEBUI_PRESET, default=webui_preset)] = str
        extra1[vol.Optional(CONF_TEXT_GEN_WEBUI_CHAT_MODE, default=webui_chat_mode)] = SelectSelector(SelectSelectorConfig(
            options=["", TEXT_GEN_WEBUI_CHAT_MODE_CHAT, TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT, TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT],
            translation_key=CONF_TEXT_GEN_WEBUI_CHAT_MODE,
            multiple=False,
            mode=SelectSelectorMode.DROPDOWN,
        ))
        extra2[vol.Optional(CONF_TEXT_GEN_WEBUI_ADMIN_KEY)] = TextSelector(TextSelectorConfig(type="password"))

    elif backend_type == BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER:
        default_port = "8000"

    return vol.Schema(
        {
            vol.Required(CONF_HOST, default=host if host else DEFAULT_HOST): str,
            vol.Required(CONF_PORT, default=port if port else default_port): str,
            vol.Required(CONF_SSL, default=ssl if ssl else DEFAULT_SSL): bool,
            vol.Required(CONF_CHAT_MODEL, default=chat_model if chat_model else DEFAULT_CHAT_MODEL): str,
            vol.Required(CONF_REMOTE_USE_CHAT_ENDPOINT, default=use_chat_endpoint if use_chat_endpoint else DEFAULT_REMOTE_USE_CHAT_ENDPOINT): bool,
            **extra1,
            vol.Optional(CONF_OPENAI_API_KEY): TextSelector(TextSelectorConfig(type="password")),
            **extra2
        }
    )


class BaseLlamaConversationConfigFlow(FlowHandler, ABC):
    """Represent the base config flow for Z-Wave JS."""

    @property
    @abstractmethod
    def flow_manager(self) -> FlowManager:
        """Return the flow manager of the flow."""

    @abstractmethod
    async def async_step_pick_backend(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """ Select backend """

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
    model_config: dict[str, Any] = {}

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
        return await self.async_step_pick_backend()

    async def async_step_pick_backend(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors = {}

        schema = STEP_INIT_DATA_SCHEMA()

        if user_input:
            local_backend = is_local_backend(user_input[CONF_BACKEND_TYPE])
            self.model_config.update(user_input)
            if local_backend:
                return await self.async_step_install_local_wheels()
                # this check isn't working right now
                # for key, value in self.hass.data.get(DOMAIN, {}).items():
                #     other_backend_type = value.data.get(CONF_BACKEND_TYPE)
                #     if other_backend_type == BACKEND_TYPE_LLAMA_HF or \
                #         other_backend_type == BACKEND_TYPE_LLAMA_EXISTING:
                #         errors["base"] = "other_existing_local"
                #         schema = STEP_INIT_DATA_SCHEMA(
                #             backend_type=user_input[CONF_BACKEND_TYPE],
                #         )
                # if "base" not in errors:
                #     return await self.async_step_install_local_wheels()
            else:
                return await self.async_step_remote_model()
        elif self.install_wheel_error:
            errors["base"] = str(self.install_wheel_error)
            self.install_wheel_error = None

        return self.async_show_form(
            step_id="pick_backend", data_schema=schema, errors=errors
        )

    async def async_step_install_local_wheels(
      self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if not user_input:
            if self.install_wheel_task:
                return self.async_show_progress(
                    progress_task=self.install_wheel_task,
                    step_id="install_local_wheels",
                    progress_action="install_local_wheels",
                )

            _LOGGER.debug("Queuing install task...")
            self.install_wheel_task = self.hass.async_add_executor_job(
                install_llama_cpp_python, self.hass.config.config_dir
            )

            self.hass.async_create_task(self._async_do_task(self.install_wheel_task))

            return self.async_show_progress(
                progress_task=self.install_wheel_task,
                step_id="install_local_wheels",
                progress_action="install_local_wheels",
            )

        wheel_install_result = user_input["result"]
        if isinstance(wheel_install_result, Exception):
            _LOGGER.warning("Failed to install wheel: %s", repr(wheel_install_result))
            self.install_wheel_error = wheel_install_result
            self.install_wheel_task = None
            return self.async_show_progress_done(next_step_id="pick_backend")
        elif wheel_install_result == False:
            _LOGGER.warning("Failed to install wheel: %s", repr(wheel_install_result))
            self.install_wheel_error = "pip_wheel_error"
            self.install_wheel_task = None
            return self.async_show_progress_done(next_step_id="pick_backend")
        else:
            _LOGGER.debug(f"Finished install: {wheel_install_result}")
            self.install_wheel_task = None
            return self.async_show_progress_done(next_step_id="local_model")

    async def async_step_local_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        errors = {}

        backend_type = self.model_config[CONF_BACKEND_TYPE]
        if backend_type == BACKEND_TYPE_LLAMA_HF:
            schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA()
        elif backend_type == BACKEND_TYPE_LLAMA_EXISTING:
            schema = STEP_LOCAL_SETUP_EXISTING_DATA_SCHEMA()
        else:
            raise ValueError()

        if self.download_error:
            errors["base"] = "download_failed"
            schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA(
                chat_model=self.model_config[CONF_CHAT_MODEL],
                downloaded_model_quantization=self.model_config[CONF_DOWNLOADED_MODEL_QUANTIZATION]
            )

        if user_input and "result" not in user_input:
            self.model_config.update(user_input)

            if backend_type == BACKEND_TYPE_LLAMA_HF:
                return await self.async_step_download()
            else:
                model_file = self.model_config[CONF_DOWNLOADED_MODEL_FILE]
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
                    progress_task=self.download_task,
                    step_id="download",
                    progress_action="download",
                )

            model_name = self.model_config[CONF_CHAT_MODEL]
            quantization_type = self.model_config[CONF_DOWNLOADED_MODEL_QUANTIZATION]

            storage_folder = os.path.join(self.hass.config.media_dirs["local"], "models")
            self.download_task = self.hass.async_add_executor_job(
                download_model_from_hf, model_name, quantization_type, storage_folder
            )

            self.hass.async_create_task(self._async_do_task(self.download_task))

            return self.async_show_progress(
                progress_task=self.download_task,
                step_id="download",
                progress_action="download",
            )

        download_result = user_input["result"]
        self.download_task = None

        if isinstance(download_result, Exception):
            _LOGGER.info("Failed to download model: %s", repr(download_result))
            self.download_error = download_result
            return self.async_show_progress_done(next_step_id="local_model")
        else:
            self.model_config[CONF_DOWNLOADED_MODEL_FILE] = download_result
            return self.async_show_progress_done(next_step_id="finish")


    def _validate_text_generation_webui(self, user_input: dict) -> str:
        try:
            headers = {}
            api_key = user_input.get(CONF_TEXT_GEN_WEBUI_ADMIN_KEY, user_input.get(CONF_OPENAI_API_KEY))
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            models_result = requests.get(
                f"{'https' if self.model_config[CONF_SSL] else 'http'}://{self.model_config[CONF_HOST]}:{self.model_config[CONF_PORT]}/v1/internal/model/list",
                headers=headers
            )
            models_result.raise_for_status()

            models = models_result.json()

            for model in models["model_names"]:
                if model == self.model_config[CONF_CHAT_MODEL].replace("/", "_"):
                    return ""

            return "missing_model_api"

        except Exception as ex:
            _LOGGER.info("Connection error was: %s", repr(ex))
            return "failed_to_connect"
        
    def _validate_ollama(self, user_input: dict) -> str:
        try:
            headers = {}
            api_key = user_input.get(CONF_OPENAI_API_KEY)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            models_result = requests.get(
                f"{'https' if self.model_config[CONF_SSL] else 'http'}://{self.model_config[CONF_HOST]}:{self.model_config[CONF_PORT]}/api/tags",
                headers=headers
            )
            models_result.raise_for_status()

            models = models_result.json()["models"]

            for model in models:
                model_name = self.model_config[CONF_CHAT_MODEL]
                if ":" in model_name:
                    if model["name"] == model_name:
                        return ""
                elif model["name"].split(":")[0] == model_name:
                    return ""
                

            return "missing_model_api"

        except Exception as ex:
            _LOGGER.info("Connection error was: %s", repr(ex))
            return "failed_to_connect"

    async def async_step_remote_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        errors = {}
        backend_type = self.model_config[CONF_BACKEND_TYPE]
        schema = STEP_REMOTE_SETUP_DATA_SCHEMA(backend_type)

        if user_input:
            try:
                self.model_config.update(user_input)
                error_reason = None

                # validate and load when using text-generation-webui or ollama
                if backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI:
                    error_reason = await self.hass.async_add_executor_job(
                        self._validate_text_generation_webui, user_input
                    )
                elif backend_type == BACKEND_TYPE_OLLAMA:
                    error_reason = await self.hass.async_add_executor_job(
                        self._validate_ollama, user_input
                    )

                if error_reason:
                    errors["base"] = error_reason
                    schema = STEP_REMOTE_SETUP_DATA_SCHEMA(
                        backend_type,
                        host=user_input[CONF_HOST],
                        port=user_input[CONF_PORT],
                        ssl=user_input[CONF_SSL],
                        chat_model=user_input[CONF_CHAT_MODEL],
                        use_chat_endpoint=user_input[CONF_REMOTE_USE_CHAT_ENDPOINT],
                        webui_preset=user_input.get(CONF_TEXT_GEN_WEBUI_PRESET),
                        webui_chat_mode=user_input.get(CONF_TEXT_GEN_WEBUI_CHAT_MODE),
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

        model_name = self.model_config.get(CONF_CHAT_MODEL)
        backend = self.model_config[CONF_BACKEND_TYPE]
        if backend == BACKEND_TYPE_LLAMA_EXISTING:
            model_name = os.path.basename(self.model_config.get(CONF_DOWNLOADED_MODEL_FILE))
        location = "llama.cpp" if is_local_backend(backend) else "remote"

        return self.async_create_entry(
            title=f"LLM Model '{model_name}' ({location})",
            description="A Large Language Model Chat Agent",
            data=self.model_config,
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
        schema = local_llama_config_option_schema(
            self.config_entry.options,
            self.config_entry.data[CONF_BACKEND_TYPE],
        )
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


def insert_after_key(input_dict: dict, key_name: str, other_dict: dict):
    # if we want to insert them into the above list we need to re-build the dictionary
    result = {}
    for key in input_dict.keys():
        result[key] = input_dict[key]

        if key.schema == key_name:
            for other_key in other_dict.keys():
                result[other_key] = other_dict[other_key]

    return result

def local_llama_config_option_schema(options: MappingProxyType[str, Any], backend_type: str) -> dict:
    """Return a schema for Local LLaMA completion options."""
    if not options:
        options = DEFAULT_OPTIONS

    result = {
        vol.Required(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT)},
            default=DEFAULT_PROMPT,
        ): TemplateSelector(),
        vol.Required(
            CONF_PROMPT_TEMPLATE,
            description={"suggested_value": options.get(CONF_PROMPT_TEMPLATE)},
            default=DEFAULT_PROMPT_TEMPLATE,
        ): SelectSelector(SelectSelectorConfig(
            options=list(PROMPT_TEMPLATE_DESCRIPTIONS.keys()),
            translation_key=CONF_PROMPT_TEMPLATE,
            multiple=False,
            mode=SelectSelectorMode.DROPDOWN,
        )),
        vol.Required(
            CONF_MAX_TOKENS,
            description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            default=DEFAULT_MAX_TOKENS,
        ): int,
        vol.Required(
            CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
            description={"suggested_value": options.get(CONF_EXTRA_ATTRIBUTES_TO_EXPOSE)},
            default=DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
        ): TextSelector(TextSelectorConfig(multiple=True)),
        vol.Required(
            CONF_SERVICE_CALL_REGEX,
            description={"suggested_value": options.get(CONF_SERVICE_CALL_REGEX)},
            default=DEFAULT_SERVICE_CALL_REGEX,
        ): str,
        vol.Required(
            CONF_REFRESH_SYSTEM_PROMPT,
            description={"suggested_value": options.get(CONF_REFRESH_SYSTEM_PROMPT)},
            default=DEFAULT_REFRESH_SYSTEM_PROMPT,
        ): bool,
        vol.Required(
            CONF_REMEMBER_CONVERSATION,
            description={"suggested_value": options.get(CONF_REMEMBER_CONVERSATION)},
            default=DEFAULT_REMEMBER_CONVERSATION,
        ): bool,
        vol.Optional(
            CONF_REMEMBER_NUM_INTERACTIONS,
            description={"suggested_value": options.get(CONF_REMEMBER_NUM_INTERACTIONS)},
        ): int,
    }

    if is_local_backend(backend_type):
        result = insert_after_key(result, CONF_MAX_TOKENS, {
            vol.Required(
                CONF_TOP_K,
                description={"suggested_value": options.get(CONF_TOP_K)},
                default=DEFAULT_TOP_K,
            ): NumberSelector(NumberSelectorConfig(min=1, max=256, step=1)),
            vol.Required(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=DEFAULT_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_USE_GBNF_GRAMMAR,
                description={"suggested_value": options.get(CONF_USE_GBNF_GRAMMAR)},
                default=DEFAULT_USE_GBNF_GRAMMAR,
            ): bool
        })
    elif backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI:
        result = insert_after_key(result, CONF_MAX_TOKENS, {
            vol.Required(
                CONF_REQUEST_TIMEOUT,
                description={"suggested_value": options.get(CONF_REQUEST_TIMEOUT)},
                default=DEFAULT_REQUEST_TIMEOUT,
            ): NumberSelector(NumberSelectorConfig(min=5, max=900, step=1, unit_of_measurement=UnitOfTime.SECONDS, mode=NumberSelectorMode.BOX)),
            vol.Required(
                CONF_REMOTE_USE_CHAT_ENDPOINT,
                description={"suggested_value": options.get(CONF_REMOTE_USE_CHAT_ENDPOINT)},
                default=DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
            ): bool,
            vol.Optional(
                CONF_TEXT_GEN_WEBUI_PRESET,
                description={"suggested_value": options.get(CONF_TEXT_GEN_WEBUI_PRESET)},
            ): str,
            vol.Required(
                CONF_TEXT_GEN_WEBUI_CHAT_MODE,
                description={"suggested_value": options.get(CONF_TEXT_GEN_WEBUI_CHAT_MODE)},
                default=DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE,
            ): SelectSelector(SelectSelectorConfig(
                options=[TEXT_GEN_WEBUI_CHAT_MODE_CHAT, TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT, TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT],
                translation_key=CONF_TEXT_GEN_WEBUI_CHAT_MODE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
        })
    elif backend_type == BACKEND_TYPE_GENERIC_OPENAI:
        result = insert_after_key(result, CONF_MAX_TOKENS, {
            vol.Required(
                CONF_REQUEST_TIMEOUT,
                description={"suggested_value": options.get(CONF_REQUEST_TIMEOUT)},
                default=DEFAULT_REQUEST_TIMEOUT,
            ): NumberSelector(NumberSelectorConfig(min=5, max=900, step=1, unit_of_measurement=UnitOfTime.SECONDS, mode=NumberSelectorMode.BOX)),
            vol.Required(
                CONF_REMOTE_USE_CHAT_ENDPOINT,
                description={"suggested_value": options.get(CONF_REMOTE_USE_CHAT_ENDPOINT)},
                default=DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
            ): bool,
            vol.Required(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=DEFAULT_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        })
    elif backend_type == BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER:
        result = insert_after_key(result, CONF_MAX_TOKENS, {
            vol.Required(
                CONF_REQUEST_TIMEOUT,
                description={"suggested_value": options.get(CONF_REQUEST_TIMEOUT)},
                default=DEFAULT_REQUEST_TIMEOUT,
            ): NumberSelector(NumberSelectorConfig(min=5, max=900, step=1, unit_of_measurement=UnitOfTime.SECONDS, mode=NumberSelectorMode.BOX)),
            vol.Required(
                CONF_REMOTE_USE_CHAT_ENDPOINT,
                description={"suggested_value": options.get(CONF_REMOTE_USE_CHAT_ENDPOINT)},
                default=DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
            ): bool,
            vol.Required(
                CONF_TOP_K,
                description={"suggested_value": options.get(CONF_TOP_K)},
                default=DEFAULT_TOP_K,
            ): NumberSelector(NumberSelectorConfig(min=1, max=256, step=1)),
            vol.Required(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=DEFAULT_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_USE_GBNF_GRAMMAR,
                description={"suggested_value": options.get(CONF_USE_GBNF_GRAMMAR)},
                default=DEFAULT_USE_GBNF_GRAMMAR,
            ): bool
        })
    elif backend_type == BACKEND_TYPE_OLLAMA:
        result = insert_after_key(result, CONF_MAX_TOKENS, {
            vol.Required(
                CONF_REQUEST_TIMEOUT,
                description={"suggested_value": options.get(CONF_REQUEST_TIMEOUT)},
                default=DEFAULT_REQUEST_TIMEOUT,
            ): NumberSelector(NumberSelectorConfig(min=5, max=900, step=1, unit_of_measurement=UnitOfTime.SECONDS, mode=NumberSelectorMode.BOX)),
            vol.Required(
                CONF_OLLAMA_KEEP_ALIVE_MIN,
                description={"suggested_value": options.get(CONF_OLLAMA_KEEP_ALIVE_MIN)},
                default=DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
            ): NumberSelector(NumberSelectorConfig(min=-1, max=1440, step=1, unit_of_measurement=UnitOfTime.MINUTES, mode=NumberSelectorMode.BOX)),
            vol.Required(
                CONF_REMOTE_USE_CHAT_ENDPOINT,
                description={"suggested_value": options.get(CONF_REMOTE_USE_CHAT_ENDPOINT)},
                default=DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
            ): bool,
            vol.Required(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=DEFAULT_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        })

    return result
