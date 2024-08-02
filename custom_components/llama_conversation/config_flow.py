"""Config flow for Local LLM Conversation integration."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL, CONF_LLM_HASS_API, UnitOfTime
from homeassistant.data_entry_flow import (
    AbortFlow,
    FlowHandler,
    FlowManager,
    FlowResult,
)
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TemplateSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    BooleanSelector,
    BooleanSelectorConfig,
)
from homeassistant.util.package import is_installed
from importlib.metadata import version

from .utils import download_model_from_hf, install_llama_cpp_python, format_url, MissingQuantizationException
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_MIN_P,
    CONF_TYPICAL_P,
    CONF_REQUEST_TIMEOUT,
    CONF_BACKEND_TYPE,
    CONF_SELECTED_LANGUAGE,
    CONF_SELECTED_LANGUAGE_OPTIONS,
    CONF_DOWNLOADED_MODEL_FILE,
    CONF_DOWNLOADED_MODEL_QUANTIZATION,
    CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS,
    CONF_PROMPT_TEMPLATE,
    CONF_TOOL_FORMAT,
    CONF_TOOL_MULTI_TURN_CHAT,
    CONF_ENABLE_FLASH_ATTENTION,
    CONF_USE_GBNF_GRAMMAR,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
    CONF_TEXT_GEN_WEBUI_PRESET,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_PROMPT_CACHING_ENABLED,
    CONF_PROMPT_CACHING_INTERVAL,
    CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    CONF_IN_CONTEXT_EXAMPLES_FILE,
    CONF_NUM_IN_CONTEXT_EXAMPLES,
    CONF_OPENAI_API_KEY,
    CONF_TEXT_GEN_WEBUI_ADMIN_KEY,
    CONF_SERVICE_CALL_REGEX,
    CONF_REMOTE_USE_CHAT_ENDPOINT,
    CONF_TEXT_GEN_WEBUI_CHAT_MODE,
    CONF_OLLAMA_KEEP_ALIVE_MIN,
    CONF_OLLAMA_JSON_MODE,
    CONF_GENERIC_OPENAI_PATH,
    CONF_GENERIC_OPENAI_VALIDATE_MODEL,
    CONF_CONTEXT_LENGTH,
    CONF_BATCH_SIZE,
    CONF_THREAD_COUNT,
    CONF_BATCH_THREAD_COUNT,
    DEFAULT_CHAT_MODEL,
    DEFAULT_PORT,
    DEFAULT_SSL,
    DEFAULT_MAX_TOKENS,
    PERSONA_PROMPTS,
    DEFAULT_PROMPT_BASE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_MIN_P,
    DEFAULT_TYPICAL_P,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_BACKEND_TYPE,
    DEFAULT_DOWNLOADED_MODEL_QUANTIZATION,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_TOOL_FORMAT,
    DEFAULT_TOOL_MULTI_TURN_CHAT,
    DEFAULT_ENABLE_FLASH_ATTENTION,
    DEFAULT_USE_GBNF_GRAMMAR,
    DEFAULT_GBNF_GRAMMAR_FILE,
    DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_NUM_INTERACTIONS,
    DEFAULT_PROMPT_CACHING_ENABLED,
    DEFAULT_PROMPT_CACHING_INTERVAL,
    DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    DEFAULT_IN_CONTEXT_EXAMPLES_FILE,
    DEFAULT_NUM_IN_CONTEXT_EXAMPLES,
    DEFAULT_SERVICE_CALL_REGEX,
    DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
    DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE,
    DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
    DEFAULT_OLLAMA_JSON_MODE,
    DEFAULT_GENERIC_OPENAI_PATH,
    DEFAULT_GENERIC_OPENAI_VALIDATE_MODEL,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_THREAD_COUNT,
    DEFAULT_BATCH_THREAD_COUNT,
    BACKEND_TYPE_LLAMA_HF,
    BACKEND_TYPE_LLAMA_EXISTING,
    BACKEND_TYPE_TEXT_GEN_WEBUI,
    BACKEND_TYPE_GENERIC_OPENAI,
    BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER,
    BACKEND_TYPE_OLLAMA,
    PROMPT_TEMPLATE_DESCRIPTIONS,
    TOOL_FORMAT_FULL,
    TOOL_FORMAT_REDUCED,
    TOOL_FORMAT_MINIMAL,
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT,
    TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT,
    TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT,
    DOMAIN,
    HOME_LLM_API_ID,
    DEFAULT_OPTIONS,
    OPTIONS_OVERRIDES,
    RECOMMENDED_CHAT_MODELS,
    EMBEDDED_LLAMA_CPP_PYTHON_VERSION
)

from . import HomeLLMAPI

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

def STEP_LOCAL_SETUP_EXISTING_DATA_SCHEMA(model_file=None, selected_language=None):
    return vol.Schema(
        {
            vol.Required(CONF_DOWNLOADED_MODEL_FILE, default=model_file if model_file else ""): str,
            vol.Required(CONF_SELECTED_LANGUAGE, default=selected_language if selected_language else "en"): SelectSelector(SelectSelectorConfig(
                options=CONF_SELECTED_LANGUAGE_OPTIONS,
                translation_key=CONF_SELECTED_LANGUAGE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
        }
    )

def STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA(*, chat_model=None, downloaded_model_quantization=None, selected_language=None, available_quantizations=None):
    return vol.Schema(
        {
            vol.Required(CONF_CHAT_MODEL, default=chat_model if chat_model else DEFAULT_CHAT_MODEL): SelectSelector(SelectSelectorConfig(
                options=RECOMMENDED_CHAT_MODELS,
                custom_value=True,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            vol.Required(CONF_DOWNLOADED_MODEL_QUANTIZATION, default=downloaded_model_quantization if downloaded_model_quantization else DEFAULT_DOWNLOADED_MODEL_QUANTIZATION): vol.In(available_quantizations if available_quantizations else CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS),
            vol.Required(CONF_SELECTED_LANGUAGE, default=selected_language if selected_language else "en"): SelectSelector(SelectSelectorConfig(
                options=CONF_SELECTED_LANGUAGE_OPTIONS,
                translation_key=CONF_SELECTED_LANGUAGE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
        }
    )

def STEP_REMOTE_SETUP_DATA_SCHEMA(backend_type: str, *, host=None, port=None, ssl=None, chat_model=None, available_chat_models=[], selected_language=None, selected_path=None):

    extra1, extra2 = ({}, {})
    default_port = DEFAULT_PORT

    if backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI: 
        extra2[vol.Optional(CONF_TEXT_GEN_WEBUI_ADMIN_KEY)] = TextSelector(TextSelectorConfig(type="password"))        
    elif backend_type == BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER:
        default_port = "8000"
    elif backend_type == BACKEND_TYPE_OLLAMA:
        default_port = "11434"
    elif backend_type == BACKEND_TYPE_GENERIC_OPENAI:
        default_port = ""
        extra2[vol.Required(
            CONF_GENERIC_OPENAI_PATH,
            default=selected_path if selected_path else DEFAULT_GENERIC_OPENAI_PATH
        )] = TextSelector(TextSelectorConfig(prefix="/"))
        extra1[vol.Required(
            CONF_GENERIC_OPENAI_VALIDATE_MODEL,
            default=DEFAULT_GENERIC_OPENAI_VALIDATE_MODEL
        )] = BooleanSelector(BooleanSelectorConfig())

    return vol.Schema(
        {
            vol.Required(CONF_HOST, default=host if host else ""): str,
            vol.Optional(CONF_PORT, default=port if port else default_port): str,
            vol.Required(CONF_SSL, default=ssl if ssl else DEFAULT_SSL): bool,
            vol.Required(CONF_CHAT_MODEL, default=chat_model if chat_model else DEFAULT_CHAT_MODEL): SelectSelector(SelectSelectorConfig(
                options=available_chat_models,
                custom_value=True,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            **extra1,
            vol.Required(CONF_SELECTED_LANGUAGE, default=selected_language if selected_language else "en"): SelectSelector(SelectSelectorConfig(
                options=CONF_SELECTED_LANGUAGE_OPTIONS,
                translation_key=CONF_SELECTED_LANGUAGE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
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
    async def async_step_model_parameters(
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
    """Handle a config flow for Local LLM Conversation."""

    VERSION = 2
    install_wheel_task = None
    install_wheel_error = None
    download_task = None
    download_error = None
    model_config: dict[str, Any]
    options: dict[str, Any]
    selected_language: str

    @property
    def flow_manager(self) -> config_entries.ConfigEntriesFlowManager:
        """Return the correct flow manager."""
        return self.hass.config_entries.flow

    def async_remove(self) -> None:
        if self.download_task:
            self.download_task.cancel()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        self.model_config = {}
        self.options = {}
        
        # make sure the API is registered
        if not any([x.id == HOME_LLM_API_ID for x in llm.async_get_apis(self.hass)]):
            llm.async_register_api(self.hass, HomeLLMAPI(self.hass))

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
                if is_installed("llama-cpp-python") and version("llama-cpp-python") == EMBEDDED_LLAMA_CPP_PYTHON_VERSION:
                    return await self.async_step_local_model()
                else:
                    return await self.async_step_install_local_wheels()
            else:
                return await self.async_step_remote_model()
        elif self.install_wheel_error:
            errors["base"] = str(self.install_wheel_error)
            self.install_wheel_error = None

        return self.async_show_form(
            step_id="pick_backend", data_schema=schema, errors=errors, last_step=False
        )

    async def async_step_install_local_wheels(
      self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if not self.install_wheel_task:
            _LOGGER.debug("Queuing install task...")
            self.install_wheel_task = self.hass.async_add_executor_job(
                install_llama_cpp_python, self.hass.config.config_dir
            )

            return self.async_show_progress(
                progress_task=self.install_wheel_task,
                step_id="install_local_wheels",
                progress_action="install_local_wheels",
            )
        
        if self.install_wheel_task and not self.install_wheel_task.done():
            return self.async_show_progress(
                progress_task=self.install_wheel_task,
                step_id="install_local_wheels",
                progress_action="install_local_wheels",
            )

        install_exception = self.install_wheel_task.exception()
        if install_exception:
            _LOGGER.warning("Failed to install wheel: %s", repr(install_exception))
            self.install_wheel_error = "pip_wheel_error"
            next_step = "pick_backend"
        else:
            wheel_install_result = self.install_wheel_task.result()
            if not wheel_install_result:
                self.install_wheel_error = "pip_wheel_error"
                next_step = "pick_backend"
            else:
                _LOGGER.debug(f"Finished install: {wheel_install_result}")
                next_step = "local_model"

        self.install_wheel_task = None
        return self.async_show_progress_done(next_step_id=next_step)

    async def async_step_local_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        errors = {}
        description_placeholders = {}

        backend_type = self.model_config[CONF_BACKEND_TYPE]
        if backend_type == BACKEND_TYPE_LLAMA_HF:
            schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA()
        elif backend_type == BACKEND_TYPE_LLAMA_EXISTING:
            schema = STEP_LOCAL_SETUP_EXISTING_DATA_SCHEMA()
        else:
            raise ValueError()

        if self.download_error:
            if isinstance(self.download_error, MissingQuantizationException):
                available_quants = list(set(self.download_error.available_quants).intersection(set(CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS)))

                if len(available_quants) == 0:
                    errors["base"] = "no_supported_ggufs"
                    schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA(
                        chat_model=self.model_config[CONF_CHAT_MODEL],
                        downloaded_model_quantization=self.model_config[CONF_DOWNLOADED_MODEL_QUANTIZATION],
                        selected_language=self.selected_language
                    )
                else:
                    errors["base"] = "missing_quantization"
                    description_placeholders["missing"] = self.download_error.missing_quant
                    description_placeholders["available"] = ", ".join(self.download_error.available_quants)

                    schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA(
                        chat_model=self.model_config[CONF_CHAT_MODEL],
                        downloaded_model_quantization=self.download_error.available_quants[0],
                        selected_language=self.selected_language,
                        available_quantizations=available_quants,
                    )
            else:
                errors["base"] = "download_failed"
                description_placeholders["exception"] = str(self.download_error)
                schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA(
                    chat_model=self.model_config[CONF_CHAT_MODEL],
                    downloaded_model_quantization=self.model_config[CONF_DOWNLOADED_MODEL_QUANTIZATION],
                    selected_language=self.selected_language
                )

        if user_input and "result" not in user_input:
            self.selected_language = user_input.pop(CONF_SELECTED_LANGUAGE, self.hass.config.language)

            self.model_config.update(user_input)

            if backend_type == BACKEND_TYPE_LLAMA_HF:
                return await self.async_step_download()
            else:
                model_file = self.model_config[CONF_DOWNLOADED_MODEL_FILE]
                if os.path.exists(model_file):
                    self.model_config[CONF_CHAT_MODEL] = os.path.basename(model_file)
                    return await self.async_step_model_parameters()
                else:
                    errors["base"] = "missing_model_file"
                    schema = STEP_LOCAL_SETUP_EXISTING_DATA_SCHEMA(model_file, self.selected_language)

        return self.async_show_form(
            step_id="local_model", data_schema=schema, errors=errors, description_placeholders=description_placeholders, last_step=False
        )

    async def async_step_download(
      self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if not self.download_task:
            model_name = self.model_config[CONF_CHAT_MODEL]
            quantization_type = self.model_config[CONF_DOWNLOADED_MODEL_QUANTIZATION]

            storage_folder = os.path.join(self.hass.config.media_dirs.get("local", self.hass.config.path("media")), "models")
            self.download_task = self.hass.async_add_executor_job(
                download_model_from_hf, model_name, quantization_type, storage_folder
            )

            return self.async_show_progress(
                progress_task=self.download_task,
                step_id="download",
                progress_action="download",
            )
        
        if self.download_task and not self.download_task.done():
            return self.async_show_progress(
                progress_task=self.download_task,
                step_id="download",
                progress_action="download",
            )

        download_exception = self.download_task.exception()
        if download_exception:
            _LOGGER.info("Failed to download model: %s", repr(download_exception))
            self.download_error = download_exception
            next_step = "local_model"
        else:
            self.model_config[CONF_DOWNLOADED_MODEL_FILE] = self.download_task.result()
            next_step = "model_parameters"

        self.download_task = None
        return self.async_show_progress_done(next_step_id=next_step)
    
    async def _async_validate_generic_openai(self, user_input: dict) -> tuple:
        """
        Validates a connection to an OpenAI compatible API server and that the model exists on the remote server

        :param user_input: the input dictionary used to build the connection
        :return: a tuple of (error message name, exception detail); both can be None
        """
        try:
            headers = {}
            api_key = user_input.get(CONF_TEXT_GEN_WEBUI_ADMIN_KEY, user_input.get(CONF_OPENAI_API_KEY))
            api_base_path = user_input.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            session = async_get_clientsession(self.hass)
            async with session.get(
                format_url(
                    hostname=self.model_config[CONF_HOST],
                    port=self.model_config[CONF_PORT],
                    ssl=self.model_config[CONF_SSL],
                    path=f"/{api_base_path}/models"
                ),
                timeout=5, # quick timeout
                headers=headers
            ) as response:
                response.raise_for_status()
                models_result = await response.json()

            models = [ model["id"] for model in models_result["data"] ]

            for model in models:
                if model == self.model_config[CONF_CHAT_MODEL]:
                    return None, None, []

            return "missing_model_api", None, models

        except Exception as ex:
            _LOGGER.info("Connection error was: %s", repr(ex))
            return "failed_to_connect", ex, []

    async def _async_validate_text_generation_webui(self, user_input: dict) -> tuple:
        """
        Validates a connection to text-generation-webui and that the model exists on the remote server

        :param user_input: the input dictionary used to build the connection
        :return: a tuple of (error message name, exception detail); both can be None
        """
        try:
            headers = {}
            api_key = user_input.get(CONF_TEXT_GEN_WEBUI_ADMIN_KEY, user_input.get(CONF_OPENAI_API_KEY))
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            session = async_get_clientsession(self.hass)
            async with session.get(
                format_url(
                    hostname=self.model_config[CONF_HOST],
                    port=self.model_config[CONF_PORT],
                    ssl=self.model_config[CONF_SSL],
                    path="/v1/internal/model/list"
                ),
                timeout=5, # quick timeout
                headers=headers
            ) as response:
                response.raise_for_status()
                models = await response.json()

            for model in models["model_names"]:
                if model == self.model_config[CONF_CHAT_MODEL].replace("/", "_"):
                    return None, None, []

            return "missing_model_api", None, models["model_names"]

        except Exception as ex:
            _LOGGER.info("Connection error was: %s", repr(ex))
            return "failed_to_connect", ex, []
        
    async def _async_validate_ollama(self, user_input: dict) -> tuple:
        """
        Validates a connection to ollama and that the model exists on the remote server

        :param user_input: the input dictionary used to build the connection
        :return: a tuple of (error message name, exception detail); both can be None
        """
        try:
            headers = {}
            api_key = user_input.get(CONF_OPENAI_API_KEY)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            session = async_get_clientsession(self.hass)
            async with session.get(
                format_url(
                    hostname=self.model_config[CONF_HOST],
                    port=self.model_config[CONF_PORT],
                    ssl=self.model_config[CONF_SSL],
                    path="/api/tags"
                ),
                timeout=5, # quick timeout
                headers=headers
            ) as response:
                response.raise_for_status()
                models_result = await response.json()

            for model in models_result["models"]:
                model_name = self.model_config[CONF_CHAT_MODEL]
                if model["name"] == model_name:
                    return (None, None, [])
                
            return "missing_model_api", None, [x["name"] for x in models_result["models"]]

        except Exception as ex:
            _LOGGER.info("Connection error was: %s", repr(ex))
            return "failed_to_connect", ex, []

    async def async_step_remote_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        errors = {}
        description_placeholders = {}
        backend_type = self.model_config[CONF_BACKEND_TYPE]
        schema = STEP_REMOTE_SETUP_DATA_SCHEMA(backend_type)

        if user_input:
            try:
                self.selected_language = user_input.pop(CONF_SELECTED_LANGUAGE, self.hass.config.language)

                self.model_config.update(user_input)
                error_message = None

                # validate and load when using text-generation-webui or ollama
                if backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI:
                    error_message, ex, possible_models = await self._async_validate_text_generation_webui(user_input)
                elif backend_type == BACKEND_TYPE_OLLAMA:
                    error_message, ex, possible_models = await self._async_validate_ollama(user_input)
                elif backend_type == BACKEND_TYPE_GENERIC_OPENAI and \
                    user_input.get(CONF_GENERIC_OPENAI_VALIDATE_MODEL, DEFAULT_GENERIC_OPENAI_VALIDATE_MODEL):
                     error_message, ex, possible_models = await self._async_validate_generic_openai(user_input)
                else:
                    possible_models = []

                if error_message:
                    errors["base"] = error_message
                    if ex:
                        description_placeholders["exception"] = str(ex)
                    schema = STEP_REMOTE_SETUP_DATA_SCHEMA(
                        backend_type,
                        host=user_input[CONF_HOST],
                        port=user_input[CONF_PORT],
                        ssl=user_input[CONF_SSL],
                        chat_model=user_input[CONF_CHAT_MODEL],
                        available_chat_models=possible_models,
                        selected_language=self.selected_language,
                        selected_path=user_input.get(CONF_GENERIC_OPENAI_PATH, DEFAULT_GENERIC_OPENAI_PATH),
                    )
                else:
                    return await self.async_step_model_parameters()

            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="remote_model", data_schema=schema, errors=errors, description_placeholders=description_placeholders, last_step=False
        )
    
    async def async_step_model_parameters(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        errors = {}
        description_placeholders = {}
        backend_type = self.model_config[CONF_BACKEND_TYPE]
        model_name = self.model_config[CONF_CHAT_MODEL].lower()

        selected_default_options = { **DEFAULT_OPTIONS }
        for key in OPTIONS_OVERRIDES.keys():
            if key in model_name:
                selected_default_options.update(OPTIONS_OVERRIDES[key])

        persona = PERSONA_PROMPTS.get(self.selected_language, PERSONA_PROMPTS.get("en"))
        selected_default_options[CONF_PROMPT] = selected_default_options[CONF_PROMPT].replace("<persona>", persona)
        
        schema = vol.Schema(local_llama_config_option_schema(self.hass, selected_default_options, backend_type))

        if user_input:
            if not user_input.get(CONF_REFRESH_SYSTEM_PROMPT) and user_input.get(CONF_PROMPT_CACHING_ENABLED):
                errors["base"] = "sys_refresh_caching_enabled"

            if user_input.get(CONF_USE_GBNF_GRAMMAR):
                filename = user_input.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE)
                if not os.path.isfile(os.path.join(os.path.dirname(__file__), filename)):
                    errors["base"] = "missing_gbnf_file"
                    description_placeholders["filename"] = filename
            
            if user_input.get(CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES):
                filename = user_input.get(CONF_IN_CONTEXT_EXAMPLES_FILE, DEFAULT_IN_CONTEXT_EXAMPLES_FILE)
                if not os.path.isfile(os.path.join(os.path.dirname(__file__), filename)):
                    errors["base"] = "missing_icl_file"
                    description_placeholders["filename"] = filename

            if user_input[CONF_LLM_HASS_API] == "none":
                user_input.pop(CONF_LLM_HASS_API)
            
            if len(errors) == 0:
                try:
                    # validate input
                    schema(user_input)

                    self.options = user_input
                    return await self.async_step_finish()
                except Exception as ex:
                    _LOGGER.exception("An unknown error has occurred!")
                    errors["base"] = "unknown"

        return self.async_show_form(
            step_id="model_parameters", data_schema=schema, errors=errors, description_placeholders=description_placeholders,
        )

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:

        model_name = self.model_config.get(CONF_CHAT_MODEL)
        backend = self.model_config[CONF_BACKEND_TYPE]
        if backend == BACKEND_TYPE_LLAMA_EXISTING:
            model_name = os.path.basename(self.model_config.get(CONF_DOWNLOADED_MODEL_FILE))
        location = "llama.cpp" if is_local_backend(backend) else "remote"

        _LOGGER.debug(f"creating model with config: {self.model_config}")
        _LOGGER.debug(f"options: {self.options}")

        return self.async_create_entry(
            title=f"LLM Model '{model_name}' ({location})",
            description="A Large Language Model Chat Agent",
            data=self.model_config,
            options=self.options,
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlow(config_entry)


class OptionsFlow(config_entries.OptionsFlow):
    """Local LLM config flow options handler."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        errors = {}
        description_placeholders = {}

        if user_input is not None:
            if not user_input.get(CONF_REFRESH_SYSTEM_PROMPT) and user_input.get(CONF_PROMPT_CACHING_ENABLED):
                errors["base"] = "sys_refresh_caching_enabled"

            if user_input.get(CONF_USE_GBNF_GRAMMAR):
                filename = user_input.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE)
                if not os.path.isfile(os.path.join(os.path.dirname(__file__), filename)):
                    errors["base"] = "missing_gbnf_file"
                    description_placeholders["filename"] = filename
            
            if user_input.get(CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES):
                filename = user_input.get(CONF_IN_CONTEXT_EXAMPLES_FILE, DEFAULT_IN_CONTEXT_EXAMPLES_FILE)
                if not os.path.isfile(os.path.join(os.path.dirname(__file__), filename)):
                    errors["base"] = "missing_icl_file"
                    description_placeholders["filename"] = filename

            if user_input[CONF_LLM_HASS_API] == "none":
                user_input.pop(CONF_LLM_HASS_API)

            if len(errors) == 0:
                return self.async_create_entry(title="Local LLM Conversation", data=user_input)
            
        schema = local_llama_config_option_schema(
            self.hass,
            self.config_entry.options,
            self.config_entry.data[CONF_BACKEND_TYPE],
        )
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
            errors=errors,
            description_placeholders=description_placeholders,
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

def local_llama_config_option_schema(hass: HomeAssistant, options: MappingProxyType[str, Any], backend_type: str) -> dict:
    """Return a schema for Local LLM completion options."""
    if not options:
        options = DEFAULT_OPTIONS

    apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    result = {
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=apis)),
        vol.Required(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT)},
            default=options[CONF_PROMPT],
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
            CONF_TOOL_FORMAT,
            description={"suggested_value": options.get(CONF_TOOL_FORMAT)},
            default=DEFAULT_TOOL_FORMAT,
        ): SelectSelector(SelectSelectorConfig(
            options=[TOOL_FORMAT_FULL, TOOL_FORMAT_REDUCED, TOOL_FORMAT_MINIMAL],
            translation_key=CONF_TOOL_FORMAT,
            multiple=False,
            mode=SelectSelectorMode.DROPDOWN,
        )),
        vol.Required(
            CONF_TOOL_MULTI_TURN_CHAT,
            description={"suggested_value": options.get(CONF_TOOL_MULTI_TURN_CHAT)},
            default=DEFAULT_TOOL_MULTI_TURN_CHAT,
        ): BooleanSelector(BooleanSelectorConfig()),
        vol.Required(
            CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES,
            description={"suggested_value": options.get(CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES)},
            default=DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES,
        ): BooleanSelector(BooleanSelectorConfig()),
        vol.Required(
            CONF_IN_CONTEXT_EXAMPLES_FILE,
            description={"suggested_value": options.get(CONF_IN_CONTEXT_EXAMPLES_FILE)},
            default=DEFAULT_IN_CONTEXT_EXAMPLES_FILE,
        ): str,
        vol.Required(
            CONF_NUM_IN_CONTEXT_EXAMPLES,
            description={"suggested_value": options.get(CONF_NUM_IN_CONTEXT_EXAMPLES)},
            default=DEFAULT_NUM_IN_CONTEXT_EXAMPLES,
        ): NumberSelector(NumberSelectorConfig(min=1, max=16, step=1)),
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
        ): BooleanSelector(BooleanSelectorConfig()),
        vol.Required(
            CONF_REMEMBER_CONVERSATION,
            description={"suggested_value": options.get(CONF_REMEMBER_CONVERSATION)},
            default=DEFAULT_REMEMBER_CONVERSATION,
        ): BooleanSelector(BooleanSelectorConfig()),
        vol.Optional(
            CONF_REMEMBER_NUM_INTERACTIONS,
            description={"suggested_value": options.get(CONF_REMEMBER_NUM_INTERACTIONS)},
            default=DEFAULT_REMEMBER_NUM_INTERACTIONS,
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
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=DEFAULT_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=3, step=0.05)),
            vol.Required(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_MIN_P,
                description={"suggested_value": options.get(CONF_MIN_P)},
                default=DEFAULT_MIN_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_TYPICAL_P,
                description={"suggested_value": options.get(CONF_TYPICAL_P)},
                default=DEFAULT_TYPICAL_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_PROMPT_CACHING_ENABLED,
                description={"suggested_value": options.get(CONF_PROMPT_CACHING_ENABLED)},
                default=DEFAULT_PROMPT_CACHING_ENABLED,
            ): BooleanSelector(BooleanSelectorConfig()),
            vol.Required(
                CONF_PROMPT_CACHING_INTERVAL,
                description={"suggested_value": options.get(CONF_PROMPT_CACHING_INTERVAL)},
                default=DEFAULT_PROMPT_CACHING_INTERVAL,
            ): NumberSelector(NumberSelectorConfig(min=1, max=60, step=1)),
            # TODO: add rope_scaling_type
            vol.Required(
                CONF_CONTEXT_LENGTH,
                description={"suggested_value": options.get(CONF_CONTEXT_LENGTH)},
                default=DEFAULT_CONTEXT_LENGTH,
            ): NumberSelector(NumberSelectorConfig(min=512, max=32768, step=1)),
            vol.Required(
                CONF_BATCH_SIZE,
                description={"suggested_value": options.get(CONF_BATCH_SIZE)},
                default=DEFAULT_BATCH_SIZE,
            ): NumberSelector(NumberSelectorConfig(min=1, max=8192, step=1)),
            vol.Required(
                CONF_THREAD_COUNT,
                description={"suggested_value": options.get(CONF_THREAD_COUNT)},
                default=DEFAULT_THREAD_COUNT,
            ): NumberSelector(NumberSelectorConfig(min=1, max=(os.cpu_count() * 2), step=1)),
            vol.Required(
                CONF_BATCH_THREAD_COUNT,
                description={"suggested_value": options.get(CONF_BATCH_THREAD_COUNT)},
                default=DEFAULT_BATCH_THREAD_COUNT,
            ): NumberSelector(NumberSelectorConfig(min=1, max=(os.cpu_count() * 2), step=1)),
            vol.Required(
                CONF_ENABLE_FLASH_ATTENTION,
                description={"suggested_value": options.get(CONF_ENABLE_FLASH_ATTENTION)},
                default=DEFAULT_ENABLE_FLASH_ATTENTION,
            ): BooleanSelector(BooleanSelectorConfig()),
            vol.Required(
                CONF_USE_GBNF_GRAMMAR,
                description={"suggested_value": options.get(CONF_USE_GBNF_GRAMMAR)},
                default=DEFAULT_USE_GBNF_GRAMMAR,
            ): BooleanSelector(BooleanSelectorConfig()),
            vol.Required(
                CONF_GBNF_GRAMMAR_FILE,
                description={"suggested_value": options.get(CONF_GBNF_GRAMMAR_FILE)},
                default=DEFAULT_GBNF_GRAMMAR_FILE,
            ): str
        })
    elif backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI:
        result = insert_after_key(result, CONF_MAX_TOKENS, {
            vol.Required(
                CONF_CONTEXT_LENGTH,
                description={"suggested_value": options.get(CONF_CONTEXT_LENGTH)},
                default=DEFAULT_CONTEXT_LENGTH,
            ): NumberSelector(NumberSelectorConfig(min=512, max=32768, step=1)),
            vol.Required(
                CONF_TOP_K,
                description={"suggested_value": options.get(CONF_TOP_K)},
                default=DEFAULT_TOP_K,
            ): NumberSelector(NumberSelectorConfig(min=1, max=256, step=1)),
            vol.Required(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=DEFAULT_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=3, step=0.05)),
            vol.Required(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_MIN_P,
                description={"suggested_value": options.get(CONF_MIN_P)},
                default=DEFAULT_MIN_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_TYPICAL_P,
                description={"suggested_value": options.get(CONF_TYPICAL_P)},
                default=DEFAULT_TYPICAL_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_REQUEST_TIMEOUT,
                description={"suggested_value": options.get(CONF_REQUEST_TIMEOUT)},
                default=DEFAULT_REQUEST_TIMEOUT,
            ): NumberSelector(NumberSelectorConfig(min=5, max=900, step=1, unit_of_measurement=UnitOfTime.SECONDS, mode=NumberSelectorMode.BOX)),
            vol.Required(
                CONF_REMOTE_USE_CHAT_ENDPOINT,
                description={"suggested_value": options.get(CONF_REMOTE_USE_CHAT_ENDPOINT)},
                default=DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
            ): BooleanSelector(BooleanSelectorConfig()),
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
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=DEFAULT_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=3, step=0.05)),
            vol.Required(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_REQUEST_TIMEOUT,
                description={"suggested_value": options.get(CONF_REQUEST_TIMEOUT)},
                default=DEFAULT_REQUEST_TIMEOUT,
            ): NumberSelector(NumberSelectorConfig(min=5, max=900, step=1, unit_of_measurement=UnitOfTime.SECONDS, mode=NumberSelectorMode.BOX)),
            vol.Required(
                CONF_REMOTE_USE_CHAT_ENDPOINT,
                description={"suggested_value": options.get(CONF_REMOTE_USE_CHAT_ENDPOINT)},
                default=DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
            ): BooleanSelector(BooleanSelectorConfig()),
        })
    elif backend_type == BACKEND_TYPE_LLAMA_CPP_PYTHON_SERVER:
        result = insert_after_key(result, CONF_MAX_TOKENS, {
            vol.Required(
                CONF_TOP_K,
                description={"suggested_value": options.get(CONF_TOP_K)},
                default=DEFAULT_TOP_K,
            ): NumberSelector(NumberSelectorConfig(min=1, max=256, step=1)),
            vol.Required(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=DEFAULT_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=3, step=0.05)),
            vol.Required(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_USE_GBNF_GRAMMAR,
                description={"suggested_value": options.get(CONF_USE_GBNF_GRAMMAR)},
                default=DEFAULT_USE_GBNF_GRAMMAR,
            ): BooleanSelector(BooleanSelectorConfig()),
            vol.Required(
                CONF_GBNF_GRAMMAR_FILE,
                description={"suggested_value": options.get(CONF_GBNF_GRAMMAR_FILE)},
                default=DEFAULT_GBNF_GRAMMAR_FILE,
            ): str,
            vol.Required(
                CONF_REQUEST_TIMEOUT,
                description={"suggested_value": options.get(CONF_REQUEST_TIMEOUT)},
                default=DEFAULT_REQUEST_TIMEOUT,
            ): NumberSelector(NumberSelectorConfig(min=5, max=900, step=1, unit_of_measurement=UnitOfTime.SECONDS, mode=NumberSelectorMode.BOX)),
            vol.Required(
                CONF_REMOTE_USE_CHAT_ENDPOINT,
                description={"suggested_value": options.get(CONF_REMOTE_USE_CHAT_ENDPOINT)},
                default=DEFAULT_REMOTE_USE_CHAT_ENDPOINT,
            ): BooleanSelector(BooleanSelectorConfig()),
        })
    elif backend_type == BACKEND_TYPE_OLLAMA:
        result = insert_after_key(result, CONF_MAX_TOKENS, {
            vol.Required(
                CONF_CONTEXT_LENGTH,
                description={"suggested_value": options.get(CONF_CONTEXT_LENGTH)},
                default=DEFAULT_CONTEXT_LENGTH,
            ): NumberSelector(NumberSelectorConfig(min=512, max=32768, step=1)),
            vol.Required(
                CONF_TOP_K,
                description={"suggested_value": options.get(CONF_TOP_K)},
                default=DEFAULT_TOP_K,
            ): NumberSelector(NumberSelectorConfig(min=1, max=256, step=1)),
            vol.Required(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=DEFAULT_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=3, step=0.05)),
            vol.Required(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_TYPICAL_P,
                description={"suggested_value": options.get(CONF_TYPICAL_P)},
                default=DEFAULT_TYPICAL_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_OLLAMA_JSON_MODE,
                description={"suggested_value": options.get(CONF_OLLAMA_JSON_MODE)},
                default=DEFAULT_OLLAMA_JSON_MODE,
            ): BooleanSelector(BooleanSelectorConfig()),
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
            ): BooleanSelector(BooleanSelectorConfig()),
        })

    return result
