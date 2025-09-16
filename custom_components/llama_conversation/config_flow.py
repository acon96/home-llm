"""Config flow for Local LLM Conversation integration."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, cast

import voluptuous as vol

from homeassistant.core import HomeAssistant, callback
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL, CONF_LLM_HASS_API, UnitOfTime
from homeassistant.data_entry_flow import (
    AbortFlow,
    FlowHandler,
    FlowManager,
    FlowResult,
)
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow as BaseConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
    ConfigEntriesFlowManager,
    OptionsFlow as BaseOptionsFlow,
    ConfigEntryState,
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
    TextSelectorType,
    BooleanSelector,
    BooleanSelectorConfig,
)

from .utils import download_model_from_hf, get_llama_cpp_python_version, install_llama_cpp_python, format_url, MissingQuantizationException
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
    CONF_THINKING_PREFIX,
    CONF_THINKING_SUFFIX,
    CONF_TOOL_CALL_PREFIX,
    CONF_TOOL_CALL_SUFFIX,
    CONF_ENABLE_LEGACY_TOOL_CALLING,
    CONF_LLAMACPP_ENABLE_FLASH_ATTENTION,
    CONF_USE_GBNF_GRAMMAR,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
    CONF_TEXT_GEN_WEBUI_PRESET,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_PROMPT_CACHING_ENABLED,
    CONF_PROMPT_CACHING_INTERVAL,
    CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    CONF_IN_CONTEXT_EXAMPLES_FILE,
    CONF_NUM_IN_CONTEXT_EXAMPLES,
    CONF_OPENAI_API_KEY,
    CONF_TEXT_GEN_WEBUI_ADMIN_KEY,
    CONF_TEXT_GEN_WEBUI_CHAT_MODE,
    CONF_OLLAMA_KEEP_ALIVE_MIN,
    CONF_OLLAMA_JSON_MODE,
    CONF_GENERIC_OPENAI_PATH,
    CONF_CONTEXT_LENGTH,
    CONF_LLAMACPP_BATCH_SIZE,
    CONF_LLAMACPP_THREAD_COUNT,
    CONF_LLAMACPP_BATCH_THREAD_COUNT,
    DEFAULT_CHAT_MODEL,
    DEFAULT_PORT,
    DEFAULT_SSL,
    DEFAULT_MAX_TOKENS,
    PERSONA_PROMPTS,
    CURRENT_DATE_PROMPT,
    DEVICES_PROMPT,
    SERVICES_PROMPT,
    TOOLS_PROMPT,
    AREA_PROMPT,
    USER_INSTRUCTION,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_MIN_P,
    DEFAULT_TYPICAL_P,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_BACKEND_TYPE,
    DEFAULT_DOWNLOADED_MODEL_QUANTIZATION,
    DEFAULT_THINKING_PREFIX,
    DEFAULT_THINKING_SUFFIX,
    DEFAULT_TOOL_CALL_PREFIX,
    DEFAULT_TOOL_CALL_SUFFIX,
    DEFAULT_ENABLE_LEGACY_TOOL_CALLING,
    DEFAULT_LLAMACPP_ENABLE_FLASH_ATTENTION,
    DEFAULT_USE_GBNF_GRAMMAR,
    DEFAULT_GBNF_GRAMMAR_FILE,
    DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_NUM_INTERACTIONS,
    DEFAULT_MAX_TOOL_CALL_ITERATIONS,
    DEFAULT_PROMPT_CACHING_ENABLED,
    DEFAULT_PROMPT_CACHING_INTERVAL,
    DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES,
    DEFAULT_IN_CONTEXT_EXAMPLES_FILE,
    DEFAULT_NUM_IN_CONTEXT_EXAMPLES,
    DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE,
    DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
    DEFAULT_OLLAMA_JSON_MODE,
    DEFAULT_GENERIC_OPENAI_PATH,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_LLAMACPP_BATCH_SIZE,
    DEFAULT_LLAMACPP_THREAD_COUNT,
    DEFAULT_LLAMACPP_BATCH_THREAD_COUNT,
    BACKEND_TYPE_LLAMA_HF_SETUP,
    BACKEND_TYPE_LLAMA_EXISTING_SETUP,
    BACKEND_TYPE_LLAMA_CPP,
    BACKEND_TYPE_TEXT_GEN_WEBUI,
    BACKEND_TYPE_GENERIC_OPENAI,
    BACKEND_TYPE_GENERIC_OPENAI_RESPONSES,
    BACKEND_TYPE_LLAMA_CPP_SERVER,
    BACKEND_TYPE_OLLAMA,
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

from . import HomeLLMAPI, LocalLLMConfigEntry, LocalLLMClient
from .backends.generic_openai import GenericOpenAIAPIClient, GenericOpenAIResponsesAPIClient
from .backends.tailored_openai import TextGenerationWebuiClient, LlamaCppServerClient
from .backends.ollama import OllamaAPIClient

_LOGGER = logging.getLogger(__name__)

def is_local_backend(backend):
    return backend in [BACKEND_TYPE_LLAMA_EXISTING_SETUP, BACKEND_TYPE_LLAMA_HF_SETUP, BACKEND_TYPE_LLAMA_CPP]

def pick_backend_schema(backend_type=None, selected_language=None):
    return vol.Schema(
        {
            vol.Required(
                CONF_BACKEND_TYPE,
                default=backend_type if backend_type else DEFAULT_BACKEND_TYPE
            ): SelectSelector(SelectSelectorConfig(
                options=[
                    BACKEND_TYPE_LLAMA_CPP,
                    BACKEND_TYPE_TEXT_GEN_WEBUI,
                    BACKEND_TYPE_GENERIC_OPENAI,
                    BACKEND_TYPE_GENERIC_OPENAI_RESPONSES,
                    BACKEND_TYPE_LLAMA_CPP_SERVER,
                    BACKEND_TYPE_OLLAMA
                ],
                translation_key=CONF_BACKEND_TYPE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            vol.Required(CONF_SELECTED_LANGUAGE, default=selected_language if selected_language else "en"): SelectSelector(SelectSelectorConfig(
                options=CONF_SELECTED_LANGUAGE_OPTIONS,
                translation_key=CONF_SELECTED_LANGUAGE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
        }
    )

def remote_connection_schema(backend_type: str, *, host=None, port=None, ssl=None, selected_path=None):

    extra = {}
    default_port = DEFAULT_PORT

    if backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI:
        extra[vol.Optional(CONF_TEXT_GEN_WEBUI_ADMIN_KEY)] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
    elif backend_type == BACKEND_TYPE_LLAMA_CPP_SERVER:
        default_port = "8000"
    elif backend_type == BACKEND_TYPE_OLLAMA:
        default_port = "11434"
    elif backend_type in [BACKEND_TYPE_GENERIC_OPENAI, BACKEND_TYPE_GENERIC_OPENAI_RESPONSES]:
        default_port = ""

    return vol.Schema(
        {
            vol.Required(CONF_HOST, default=host if host else ""): str,
            vol.Optional(CONF_PORT, default=port if port else default_port): str,
            vol.Required(CONF_SSL, default=ssl if ssl else DEFAULT_SSL): bool,
            vol.Optional(CONF_OPENAI_API_KEY): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
            vol.Required(
                CONF_GENERIC_OPENAI_PATH,
                default=selected_path if selected_path else DEFAULT_GENERIC_OPENAI_PATH
            ): TextSelector(TextSelectorConfig(prefix="/")),
            **extra
        }
    )

class ConfigFlow(BaseConfigFlow, domain=DOMAIN):
    """Handle a config flow for Local LLM Conversation."""

    VERSION = 3

    install_wheel_task = None
    install_wheel_error = None
    installed_version = None
    client_config: dict[str, Any]
    internal_step: str = "init"

    @property
    def flow_manager(self) -> ConfigEntriesFlowManager:
        """Return the correct flow manager."""
        return self.hass.config_entries.flow

    def async_remove(self) -> None:
        if self.install_wheel_task:
            self.install_wheel_task.cancel()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors = {}

        if self.internal_step == "init":
            self.client_config = {}

            # make sure the API is registered
            if not any([x.id == HOME_LLM_API_ID for x in llm.async_get_apis(self.hass)]):
                llm.async_register_api(self.hass, HomeLLMAPI(self.hass))

            self.internal_step = "pick_backend"
            return self.async_show_form(
                step_id="user", data_schema=pick_backend_schema(), last_step=False
            )
        elif self.internal_step == "pick_backend":
            if user_input:
                local_backend = is_local_backend(user_input[CONF_BACKEND_TYPE])
                self.client_config.update(user_input)
                if local_backend:
                    self.installed_version = await self.hass.async_add_executor_job(get_llama_cpp_python_version)
                    _LOGGER.debug(f"installed version: {self.installed_version}")
                    if self.installed_version == EMBEDDED_LLAMA_CPP_PYTHON_VERSION:
                        return await self.async_step_finish()
                    else:
                        self.internal_step = "install_local_wheels"
                        _LOGGER.debug("Queuing install task...")
                        async def install_task():
                            await self.hass.async_add_executor_job(
                                install_llama_cpp_python, self.hass.config.config_dir
                            )

                        self.install_wheel_task = self.hass.async_create_background_task(
                            install_task(), name="llama_cpp_python_installation")

                        return self.async_show_progress(
                            progress_task=self.install_wheel_task,
                            step_id="user",
                            progress_action="install_local_wheels",
                        )
                else:
                    self.internal_step = "configure_connection"
                    return self.async_show_form(
                        step_id="user", data_schema=remote_connection_schema(self.client_config[CONF_BACKEND_TYPE]), last_step=True
                    )
            elif self.install_wheel_error:
                errors["base"] = str(self.install_wheel_error)
                self.install_wheel_error = None
            
            return self.async_show_form(
                step_id="user", data_schema=pick_backend_schema(
                    backend_type=self.client_config.get(CONF_BACKEND_TYPE),
                    selected_language=self.client_config.get(CONF_SELECTED_LANGUAGE)
                ), errors=errors, last_step=False)
        elif self.internal_step == "install_local_wheels":
            if self.install_wheel_task and not self.install_wheel_task.done():
                return self.async_show_progress(
                    progress_task=self.install_wheel_task,
                    step_id="user",
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
                    next_step = "finish"

            self.install_wheel_task = None
            self.internal_step = next_step
            return self.async_show_progress_done(next_step_id="finish")
        elif self.internal_step == "configure_connection":
            if user_input:
                self.client_config.update(user_input)

                # validate remote connections
                is_valid = True
                backend = self.client_config[CONF_BACKEND_TYPE]
                if backend == BACKEND_TYPE_GENERIC_OPENAI:
                    is_valid = await GenericOpenAIAPIClient.async_validate_connection(self.hass, self.client_config)
                elif backend == BACKEND_TYPE_GENERIC_OPENAI_RESPONSES:
                    is_valid = await GenericOpenAIResponsesAPIClient.async_validate_connection(self.hass, self.client_config)
                elif backend == BACKEND_TYPE_TEXT_GEN_WEBUI:
                    is_valid = await TextGenerationWebuiClient.async_validate_connection(self.hass, self.client_config)
                elif backend == BACKEND_TYPE_OLLAMA:
                    is_valid = await OllamaAPIClient.async_validate_connection(self.hass, self.client_config)

                if is_valid:
                    return await self.async_step_finish()
            else:
                return self.async_show_form(
                    step_id="user", data_schema=remote_connection_schema(self.client_config[CONF_BACKEND_TYPE],
                        host=self.client_config.get(CONF_HOST),
                        port=self.client_config.get(CONF_PORT),
                        ssl=self.client_config.get(CONF_SSL),
                        selected_path=self.client_config.get(CONF_GENERIC_OPENAI_PATH)
                    ), last_step=True)
        else:
            raise AbortFlow("Unknown internal step")

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:

        backend = self.client_config[CONF_BACKEND_TYPE]
        title = "Generic AI Provider"
        if is_local_backend(backend):
            title = f"LLama.cpp (llama-cpp-python v{self.installed_version})"
        else:
            host = self.client_config[CONF_HOST]
            port = self.client_config[CONF_PORT]
            ssl = self.client_config[CONF_SSL]
            path = "/" + self.client_config[CONF_GENERIC_OPENAI_PATH]
            if backend == BACKEND_TYPE_GENERIC_OPENAI or backend == BACKEND_TYPE_GENERIC_OPENAI_RESPONSES:
                title = f"Generic OpenAI at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"
            elif backend == BACKEND_TYPE_TEXT_GEN_WEBUI:
                title = f"Text-Gen WebUI at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"
            elif backend == BACKEND_TYPE_OLLAMA:
                title = f"Ollama at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"
            elif backend == BACKEND_TYPE_LLAMA_CPP_SERVER:
                title = f"LLama.cpp Server at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"

        _LOGGER.debug(f"creating provider with config: {self.client_config}")

        return self.async_create_entry(
            title=title,
            description="A Large Language Model Chat Agent",
            data={CONF_BACKEND_TYPE: backend},
            options=self.client_config,
        )
    
    @classmethod
    def async_supports_options_flow(cls, config_entry: ConfigEntry) -> bool:
        return not is_local_backend(config_entry.options.get(CONF_BACKEND_TYPE))

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> BaseOptionsFlow:
        """Create the options flow."""
        return OptionsFlow()
    
    @classmethod
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": LocalLLMSubentryFlowHandler,
        }


class OptionsFlow(BaseOptionsFlow):
    """Local LLM config flow options handler."""

    model_config: dict[str, Any] | None = None

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        errors = {}
        description_placeholders = {}

        if user_input is not None:
            if user_input[CONF_LLM_HASS_API] == "none":
                user_input.pop(CONF_LLM_HASS_API)

            # FIXME: invoke the static function on the appropriate llm client class
            # self.config_entry.runtime_data.validate_connection(user_input) # is this correct?

            if len(errors) == 0:
                return self.async_create_entry(title="Local LLM Conversation", data=user_input)

        schema = remote_connection_schema(
            backend_type=self.config_entry.options[CONF_BACKEND_TYPE],
            host=self.config_entry.options.get(CONF_HOST),
            port=self.config_entry.options.get(CONF_PORT),
            ssl=self.config_entry.options.get(CONF_SSL),
            selected_path=self.config_entry.options.get(CONF_GENERIC_OPENAI_PATH)
        )
        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )
    

def STEP_LOCAL_SETUP_EXISTING_DATA_SCHEMA(model_file=None):
    return vol.Schema(
        {
            vol.Required(CONF_DOWNLOADED_MODEL_FILE, default=model_file if model_file else ""): str,
        }
    )

def STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA(*, chat_model=None, downloaded_model_quantization=None, available_quantizations=None):
    return vol.Schema(
        {
            vol.Required(CONF_CHAT_MODEL, default=chat_model if chat_model else DEFAULT_CHAT_MODEL): SelectSelector(SelectSelectorConfig(
                options=RECOMMENDED_CHAT_MODELS,
                custom_value=True,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            vol.Required(CONF_DOWNLOADED_MODEL_QUANTIZATION, default=downloaded_model_quantization if downloaded_model_quantization else DEFAULT_DOWNLOADED_MODEL_QUANTIZATION): vol.In(available_quantizations if available_quantizations else CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS),
        }
    )

def STEP_REMOTE_MODEL_SELECTION_DATA_SCHEMA(available_models: list[str], chat_model: str | None = None):
    _LOGGER.debug(f"available models: {available_models}")
    return vol.Schema(
        {
            vol.Required(CONF_CHAT_MODEL, default=chat_model if chat_model else available_models[0]): SelectSelector(SelectSelectorConfig(
                options=available_models,
                custom_value=True,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
        }
    )

def build_prompt_template(selected_language: str, prompt_template_template: str):
    persona = PERSONA_PROMPTS.get(selected_language, PERSONA_PROMPTS["en"])
    current_date = CURRENT_DATE_PROMPT.get(selected_language, CURRENT_DATE_PROMPT["en"])
    devices = DEVICES_PROMPT.get(selected_language, DEVICES_PROMPT["en"])
    services = SERVICES_PROMPT.get(selected_language, SERVICES_PROMPT["en"])
    tools = TOOLS_PROMPT.get(selected_language, TOOLS_PROMPT["en"])
    area = AREA_PROMPT.get(selected_language, AREA_PROMPT["en"])
    user_instruction = USER_INSTRUCTION.get(selected_language, USER_INSTRUCTION["en"])

    prompt_template_template = prompt_template_template.replace("<persona>", persona)
    prompt_template_template = prompt_template_template.replace("<current_date>", current_date)
    prompt_template_template = prompt_template_template.replace("<devices>", devices)
    prompt_template_template = prompt_template_template.replace("<services>", services)
    prompt_template_template = prompt_template_template.replace("<tools>", tools)
    prompt_template_template = prompt_template_template.replace("<area>", area)
    prompt_template_template = prompt_template_template.replace("<user_instruction>", user_instruction)

    return prompt_template_template

def local_llama_config_option_schema(
    hass: HomeAssistant,
    parent_options: MappingProxyType[str, Any],
    options: MappingProxyType[str, Any],
    backend_type: str, 
    subentry_type: str,
) -> dict:
    if not options:
        options = DEFAULT_OPTIONS

    default_prompt = build_prompt_template(parent_options[CONF_SELECTED_LANGUAGE], DEFAULT_PROMPT)

    # TODO: we need to make this the "model config" i.e. each subentry defines all of the things to define
    # the model and the parent entry just defines the "connection options" (or llama-cpp-python version)

    # will also need to move the model download steps to the config sub-entry

    result: dict = {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT, default_prompt)},
            default=options.get(CONF_PROMPT, default_prompt),
        ): TemplateSelector(),
        vol.Optional(
            CONF_TEMPERATURE,
            description={"suggested_value": options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)},
            default=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
        ): NumberSelector(NumberSelectorConfig(min=0.0, max=2.0, step=0.05, mode=NumberSelectorMode.BOX)),
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
        ): NumberSelector(NumberSelectorConfig(min=1, max=8192, step=1)),
        vol.Required(
            CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
            description={"suggested_value": options.get(CONF_EXTRA_ATTRIBUTES_TO_EXPOSE)},
            default=DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
        ): TextSelector(TextSelectorConfig(multiple=True)),
        vol.Required(
            CONF_THINKING_PREFIX,
            description={"suggested_value": options.get(CONF_THINKING_PREFIX)},
            default=DEFAULT_THINKING_PREFIX,
        ): str,
        vol.Required(
            CONF_THINKING_SUFFIX,
            description={"suggested_value": options.get(CONF_THINKING_SUFFIX)},
            default=DEFAULT_THINKING_SUFFIX,
        ): str,
        vol.Required(
            CONF_TOOL_CALL_PREFIX,
            description={"suggested_value": options.get(CONF_TOOL_CALL_PREFIX)},
            default=DEFAULT_TOOL_CALL_PREFIX,
        ): str,
        vol.Required(
            CONF_TOOL_CALL_SUFFIX,
            description={"suggested_value": options.get(CONF_TOOL_CALL_SUFFIX)},
            default=DEFAULT_TOOL_CALL_SUFFIX,
        ): str
    }

    if is_local_backend(backend_type):
        result.update({
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
                CONF_LLAMACPP_BATCH_SIZE,
                description={"suggested_value": options.get(CONF_LLAMACPP_BATCH_SIZE)},
                default=DEFAULT_LLAMACPP_BATCH_SIZE,
            ): NumberSelector(NumberSelectorConfig(min=1, max=8192, step=1)),
            vol.Required(
                CONF_LLAMACPP_THREAD_COUNT,
                description={"suggested_value": options.get(CONF_LLAMACPP_THREAD_COUNT)},
                default=DEFAULT_LLAMACPP_THREAD_COUNT,
            ): NumberSelector(NumberSelectorConfig(min=1, max=((os.cpu_count() or 1) * 2), step=1)),
            vol.Required(
                CONF_LLAMACPP_BATCH_THREAD_COUNT,
                description={"suggested_value": options.get(CONF_LLAMACPP_BATCH_THREAD_COUNT)},
                default=DEFAULT_LLAMACPP_BATCH_THREAD_COUNT,
            ): NumberSelector(NumberSelectorConfig(min=1, max=((os.cpu_count() or 1) * 2), step=1)),
            vol.Required(
                CONF_LLAMACPP_ENABLE_FLASH_ATTENTION,
                description={"suggested_value": options.get(CONF_LLAMACPP_ENABLE_FLASH_ATTENTION)},
                default=DEFAULT_LLAMACPP_ENABLE_FLASH_ATTENTION,
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
        result.update({
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
    elif backend_type in BACKEND_TYPE_GENERIC_OPENAI:
        result.update({
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
                CONF_ENABLE_LEGACY_TOOL_CALLING,
                description={"suggested_value": options.get(CONF_ENABLE_LEGACY_TOOL_CALLING)},
                default=DEFAULT_ENABLE_LEGACY_TOOL_CALLING
            ): bool,
        })
    elif backend_type in BACKEND_TYPE_GENERIC_OPENAI_RESPONSES:
        del result[CONF_REMEMBER_NUM_INTERACTIONS]
        result.update({
            vol.Required(
                CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
                description={"suggested_value": options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES)},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=180, step=0.5, unit_of_measurement=UnitOfTime.MINUTES, mode=NumberSelectorMode.BOX)),
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
        })
    elif backend_type == BACKEND_TYPE_LLAMA_CPP_SERVER:
        result.update({
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
                CONF_ENABLE_LEGACY_TOOL_CALLING,
                description={"suggested_value": options.get(CONF_ENABLE_LEGACY_TOOL_CALLING)},
                default=DEFAULT_ENABLE_LEGACY_TOOL_CALLING
            ): bool,
        })
    elif backend_type == BACKEND_TYPE_OLLAMA:
        result.update({
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
        })

    if subentry_type == "conversation":
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
        result.update({
            vol.Optional(
                CONF_LLM_HASS_API,
                description={"suggested_value": options.get(CONF_LLM_HASS_API)},
                default="none",
            ): SelectSelector(SelectSelectorConfig(options=apis)),
            vol.Optional(
                CONF_REFRESH_SYSTEM_PROMPT,
                description={"suggested_value": options.get(CONF_REFRESH_SYSTEM_PROMPT, DEFAULT_REFRESH_SYSTEM_PROMPT)},
                default=options.get(CONF_REFRESH_SYSTEM_PROMPT, DEFAULT_REFRESH_SYSTEM_PROMPT),
            ): BooleanSelector(BooleanSelectorConfig()),
            vol.Optional(
                CONF_REMEMBER_CONVERSATION,
                description={"suggested_value": options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION)},
                default=options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION),
            ): BooleanSelector(BooleanSelectorConfig()),
            vol.Optional(
                CONF_REMEMBER_NUM_INTERACTIONS,
                description={"suggested_value": options.get(CONF_REMEMBER_NUM_INTERACTIONS, DEFAULT_REMEMBER_NUM_INTERACTIONS)},
                default=options.get(CONF_REMEMBER_NUM_INTERACTIONS, DEFAULT_REMEMBER_NUM_INTERACTIONS),
            ): NumberSelector(NumberSelectorConfig(min=0, max=100, mode=NumberSelectorMode.BOX)),
            vol.Optional(
                CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
                description={"suggested_value": options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, DEFAULT_REMEMBER_CONVERSATION)},
                default=options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, DEFAULT_REMEMBER_CONVERSATION),
            ): NumberSelector(NumberSelectorConfig(min=0, max=1440, mode=NumberSelectorMode.BOX)),
            vol.Required(
                CONF_MAX_TOOL_CALL_ITERATIONS,
                description={"suggested_value": options.get(CONF_MAX_TOOL_CALL_ITERATIONS)},
                default=DEFAULT_MAX_TOOL_CALL_ITERATIONS,
            ): int,
        })

    # sort the options
    global_order = [
        # general
        CONF_LLM_HASS_API,
        CONF_PROMPT,
        CONF_CONTEXT_LENGTH,
        CONF_MAX_TOKENS,
        # sampling parameters
        CONF_TEMPERATURE,
        CONF_TOP_P,
        CONF_MIN_P,
        CONF_TYPICAL_P
        CONF_TOP_K,
        # tool calling/reasoning
        CONF_THINKING_PREFIX,
        CONF_THINKING_SUFFIX,
        CONF_TOOL_CALL_PREFIX,
        CONF_TOOL_CALL_SUFFIX,
        CONF_MAX_TOOL_CALL_ITERATIONS,
        CONF_ENABLE_LEGACY_TOOL_CALLING,
        CONF_USE_GBNF_GRAMMAR,
        CONF_GBNF_GRAMMAR_FILE,
        # integration specific options
        CONF_EXTRA_ATTRIBUTES_TO_EXPOSE,
        CONF_REFRESH_SYSTEM_PROMPT,
        CONF_REMEMBER_CONVERSATION,
        CONF_REMEMBER_NUM_INTERACTIONS,
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
        CONF_PROMPT_CACHING_ENABLED,
        CONF_PROMPT_CACHING_INTERVAL,
        CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES,
        CONF_IN_CONTEXT_EXAMPLES_FILE,
        CONF_NUM_IN_CONTEXT_EXAMPLES,
        # backend specific options
        CONF_LLAMACPP_BATCH_SIZE,
        CONF_LLAMACPP_THREAD_COUNT,
        CONF_LLAMACPP_BATCH_THREAD_COUNT,
        CONF_LLAMACPP_ENABLE_FLASH_ATTENTION,
        CONF_TEXT_GEN_WEBUI_ADMIN_KEY,
        CONF_TEXT_GEN_WEBUI_PRESET,
        CONF_TEXT_GEN_WEBUI_CHAT_MODE,
        CONF_OLLAMA_KEEP_ALIVE_MIN,
        CONF_OLLAMA_JSON_MODE,
    ]

    result = { k: v for k, v in sorted(result.items(), key=lambda item: global_order.index(item[0]) if item[0] in global_order else 9999) }

    return result


class LocalLLMSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing Local LLM subentries."""

    def __init__(self) -> None:
        """Initialize the subentry flow."""
        super().__init__()

        # state for subentry flow
        self.model_config: dict[str, Any] = {}
        self.download_task = None
        self.download_error = None
        self.internal_step = "pick_model"

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    @property
    def _client(self) -> LocalLLMClient:
        """Return the Ollama client."""
        entry: LocalLLMConfigEntry = self._get_entry()
        return entry.runtime_data

    async def async_step_pick_model(
        self, user_input: dict[str, Any] | None,
        entry: LocalLLMConfigEntry
    ) -> SubentryFlowResult:
        schema = vol.Schema({})
        errors = {}
        description_placeholders = {}

        if not self.model_config:
            self.model_config = {}

        backend_type = entry.options.get(CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE)
        if backend_type == BACKEND_TYPE_LLAMA_HF_SETUP:
            schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA()
        elif backend_type == BACKEND_TYPE_LLAMA_EXISTING_SETUP:
            schema = STEP_LOCAL_SETUP_EXISTING_DATA_SCHEMA()
        else:
            schema = STEP_REMOTE_MODEL_SELECTION_DATA_SCHEMA(await entry.runtime_data.async_get_available_models())

        if self.download_error:
            if isinstance(self.download_error, MissingQuantizationException):
                available_quants = list(set(self.download_error.available_quants).intersection(set(CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS)))

                if len(available_quants) == 0:
                    errors["base"] = "no_supported_ggufs"
                    schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA(
                        chat_model=self.model_config[CONF_CHAT_MODEL],
                        downloaded_model_quantization=self.model_config[CONF_DOWNLOADED_MODEL_QUANTIZATION],
                    )
                else:
                    errors["base"] = "missing_quantization"
                    description_placeholders["missing"] = self.download_error.missing_quant
                    description_placeholders["available"] = ", ".join(self.download_error.available_quants)

                    schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA(
                        chat_model=self.model_config[CONF_CHAT_MODEL],
                        downloaded_model_quantization=self.download_error.available_quants[0],
                        available_quantizations=available_quants,
                    )
            else:
                errors["base"] = "download_failed"
                description_placeholders["exception"] = str(self.download_error)
                schema = STEP_LOCAL_SETUP_DOWNLOAD_DATA_SCHEMA(
                    chat_model=self.model_config[CONF_CHAT_MODEL],
                    downloaded_model_quantization=self.model_config[CONF_DOWNLOADED_MODEL_QUANTIZATION],
                )

        if user_input and "result" not in user_input:

            self.model_config.update(user_input)

            if backend_type == BACKEND_TYPE_LLAMA_HF_SETUP:
                return await self.async_step_download(entry)
            elif backend_type == BACKEND_TYPE_LLAMA_EXISTING_SETUP:
                model_file = self.model_config[CONF_DOWNLOADED_MODEL_FILE]
                if os.path.exists(model_file):
                    self.model_config[CONF_CHAT_MODEL] = os.path.basename(model_file)
                    return await self.async_step_model_parameters(None, entry)
                else:
                    errors["base"] = "missing_model_file"
                    schema = STEP_LOCAL_SETUP_EXISTING_DATA_SCHEMA(model_file)
            else:
                return await self.async_step_model_parameters(None, entry)

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
            last_step=False,
        )

    async def async_step_download(
        self, entry: LocalLLMConfigEntry
    ) -> SubentryFlowResult:
        if not self.download_task:
            model_name = self.model_config[CONF_CHAT_MODEL]
            quantization_type = self.model_config[CONF_DOWNLOADED_MODEL_QUANTIZATION]

            storage_folder = os.path.join(self.hass.config.media_dirs.get("local", self.hass.config.path("media")), "models")
            self.download_task = self.hass.async_add_executor_job(
                download_model_from_hf, model_name, quantization_type, storage_folder
            )

            return self.async_show_progress(
                progress_task=self.download_task,
                step_id="user",
                progress_action="download",
            )

        if self.download_task and not self.download_task.done():
            return self.async_show_progress(
                progress_task=self.download_task,
                step_id="user",
                progress_action="download",
            )

        download_exception = self.download_task.exception()
        if download_exception:
            _LOGGER.info("Failed to download model: %s", repr(download_exception))
            self.download_error = download_exception
            self.internal_step = "select_local_model"
        else:
            self.model_config[CONF_DOWNLOADED_MODEL_FILE] = self.download_task.result()
            self.internal_step = "model_parameters"

        self.download_task = None
        return self.async_show_progress_done(next_step_id="finish")

    async def async_step_model_parameters(
        self, user_input: dict[str, Any] | None,
        entry: LocalLLMConfigEntry,
    ) -> SubentryFlowResult:
        errors = {}
        description_placeholders = {}
        backend_type = entry.options[CONF_BACKEND_TYPE]

        # determine selected language from model config or parent options
        selected_language = self.model_config.get(
            CONF_SELECTED_LANGUAGE, entry.options.get(CONF_SELECTED_LANGUAGE, "en")
        )
        model_name = self.model_config.get(CONF_CHAT_MODEL, "").lower()

        selected_default_options = {**DEFAULT_OPTIONS}
        for key in OPTIONS_OVERRIDES.keys():
            if key in model_name:
                selected_default_options.update(OPTIONS_OVERRIDES[key])
                break

        # Build prompt template using the selected language
        selected_default_options[CONF_PROMPT] = build_prompt_template(
            selected_language, selected_default_options.get(CONF_PROMPT, DEFAULT_PROMPT)
        )

        schema = vol.Schema(
            local_llama_config_option_schema(
                self.hass,
                entry.options,
                MappingProxyType(selected_default_options),
                backend_type,
                self._subentry_type,
            )
        )

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

                    self.model_config.update(user_input)

                    return await self.async_step_finish()
                except Exception:
                    _LOGGER.exception("An unknown error has occurred!")
                    errors["base"] = "unknown"

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )
    
    async def async_step_failed(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Step after model downloading has failed."""
        return self.async_abort(reason="download_failed")

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Step after model downloading has succeeded."""

        # Model download completed, create/update the entry with stored config
        if self._is_new:
            return self.async_create_entry(
                title=self.model_config.get(CONF_CHAT_MODEL, "Model"),
                data=self.model_config,
            )
        else:
            return self.async_update_and_abort(
                self._get_entry(), self._get_reconfigure_subentry(), data=self.model_config
            )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle model selection and configuration step."""
        entry: LocalLLMConfigEntry = self._get_entry()

        # Ensure the parent entry is loaded before allowing subentry edits
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if self.internal_step == "pick_model":
            return await self.async_step_pick_model(user_input, entry)
        elif self.internal_step == "download":
            return await self.async_step_download(entry)
        elif self.internal_step == "model_parameters":
            return await self.async_step_model_parameters(user_input, entry)
        else:
            return self.async_abort(reason="unknown_internal_step")
        
    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None):
        return await self.async_step_model_parameters(user_input, self._get_entry())

    async_step_init = async_step_user