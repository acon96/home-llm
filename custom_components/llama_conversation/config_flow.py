"""Config flow for Local LLM Conversation integration."""
from __future__ import annotations

from asyncio import Task
import logging
import os
from typing import Any

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL, CONF_LLM_HASS_API, UnitOfTime
from homeassistant.data_entry_flow import (
    AbortFlow,
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

from .utils import download_model_from_hf, get_llama_cpp_python_version, install_llama_cpp_python, \
    is_valid_hostname, get_available_llama_cpp_versions, MissingQuantizationException
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
    CONF_INSTALLED_LLAMACPP_VERSION,
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
    CONF_LLAMACPP_REINSTALL,
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
    option_overrides,
    RECOMMENDED_CHAT_MODELS,
    EMBEDDED_LLAMA_CPP_PYTHON_VERSION
)

from . import HomeLLMAPI, LocalLLMConfigEntry, LocalLLMClient, BACKEND_TO_CLS

_LOGGER = logging.getLogger(__name__)

def _coerce_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

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
    default_path = DEFAULT_GENERIC_OPENAI_PATH

    if backend_type == BACKEND_TYPE_TEXT_GEN_WEBUI:
        extra[vol.Optional(CONF_TEXT_GEN_WEBUI_ADMIN_KEY)] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
    elif backend_type == BACKEND_TYPE_LLAMA_CPP_SERVER:
        default_port = "8000"
    elif backend_type == BACKEND_TYPE_OLLAMA:
        default_port = "11434"
        default_path = ""
    elif backend_type in [BACKEND_TYPE_GENERIC_OPENAI, BACKEND_TYPE_GENERIC_OPENAI_RESPONSES]:
        default_port = ""

    return vol.Schema(
        {
            vol.Required(CONF_HOST, default=host if host else ""): str,
            vol.Optional(CONF_PORT, default=port if port else default_port): str,
            vol.Required(CONF_SSL, default=ssl if ssl else DEFAULT_SSL): bool,
            vol.Optional(CONF_OPENAI_API_KEY): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
            vol.Optional(
                CONF_GENERIC_OPENAI_PATH,
                default=selected_path if selected_path else default_path
            ): TextSelector(TextSelectorConfig(prefix="/")),
            **extra
        }
    )

class ConfigFlow(BaseConfigFlow, domain=DOMAIN):
    """Handle a config flow for Local LLM Conversation."""

    VERSION = 3
    MINOR_VERSION = 1

    install_wheel_task = None
    install_wheel_error = None
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
        description_placeholders = {}

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
                backend = user_input[CONF_BACKEND_TYPE]
                self.client_config.update(user_input)
                if backend == BACKEND_TYPE_LLAMA_CPP:
                    installed_version = await self.hass.async_add_executor_job(get_llama_cpp_python_version)
                    _LOGGER.debug(f"installed version: {installed_version}")
                    if installed_version and installed_version == EMBEDDED_LLAMA_CPP_PYTHON_VERSION:
                        self.client_config[CONF_INSTALLED_LLAMACPP_VERSION] = installed_version
                        return await self.async_step_finish()
                    else:
                        self.internal_step = "install_local_wheels"
                        _LOGGER.debug("Queuing install task...")
                        async def install_task():
                            return await self.hass.async_add_executor_job(
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
            if not self.install_wheel_task:
                return self.async_abort(reason="unknown")
            
            if not self.install_wheel_task.done():
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
                    self.client_config[CONF_INSTALLED_LLAMACPP_VERSION] = await self.hass.async_add_executor_job(get_llama_cpp_python_version)

            self.install_wheel_task = None
            self.internal_step = next_step
            return self.async_show_progress_done(next_step_id="finish")
        elif self.internal_step == "configure_connection":
            if user_input:
                self.client_config.update(user_input)

                hostname = user_input.get(CONF_HOST, "")
                if not is_valid_hostname(hostname):
                    errors["base"] = "invalid_hostname"
                else:
                    # validate remote connections
                    connect_err = await BACKEND_TO_CLS[self.client_config[CONF_BACKEND_TYPE]].async_validate_connection(self.hass, self.client_config)

                    if connect_err:
                        errors["base"] = "failed_to_connect"
                        description_placeholders["exception"] = str(connect_err)
                    else:
                        return await self.async_step_finish()
            
            return self.async_show_form(
                step_id="user", 
                data_schema=remote_connection_schema(
                    self.client_config[CONF_BACKEND_TYPE],
                    host=self.client_config.get(CONF_HOST),
                    port=self.client_config.get(CONF_PORT),
                    ssl=self.client_config.get(CONF_SSL),
                    selected_path=self.client_config.get(CONF_GENERIC_OPENAI_PATH)
                ), 
                errors=errors,
                description_placeholders=description_placeholders,
                last_step=True
            )
        else:
            raise AbortFlow("Unknown internal step")

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:

        backend = self.client_config[CONF_BACKEND_TYPE]
        title = BACKEND_TO_CLS[backend].get_name(self.client_config)
        _LOGGER.debug(f"creating provider with config: {self.client_config}")

        # block duplicate providers
        for entry in self.hass.config_entries.async_entries(DOMAIN):
            if backend == BACKEND_TYPE_LLAMA_CPP and \
               entry.data.get(CONF_BACKEND_TYPE) == BACKEND_TYPE_LLAMA_CPP:
                return self.async_abort(reason="duplicate_client")

        return self.async_create_entry(
            title=title,
            description="A Large Language Model Chat Agent",
            data={CONF_BACKEND_TYPE: backend},
            options=self.client_config,
        )
    
    @classmethod
    def async_supports_options_flow(cls, config_entry: ConfigEntry) -> bool:
        return True

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
            # "ai_task_data": LocalLLMSubentryFlowHandler,
        }


class OptionsFlow(BaseOptionsFlow):
    """Local LLM config flow options handler."""

    model_config: dict[str, Any] | None = None
    reinstall_task: Task[Any] | None = None
    wheel_install_error: str | None = None
    wheel_install_successful: bool = False

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        errors = {}
        description_placeholders = {}

        backend_type = self.config_entry.data.get(CONF_BACKEND_TYPE, DEFAULT_BACKEND_TYPE)
        client_config = dict(self.config_entry.options)

        if self.wheel_install_error:
            _LOGGER.warning("Failed to install wheel: %s", repr(self.wheel_install_error))
            return self.async_abort(reason="pip_wheel_error")

        if self.wheel_install_successful:
            client_config[CONF_INSTALLED_LLAMACPP_VERSION] = await self.hass.async_add_executor_job(get_llama_cpp_python_version)
            _LOGGER.debug(f"new version is: {client_config[CONF_INSTALLED_LLAMACPP_VERSION]}")
            return self.async_create_entry(data=client_config)

        if backend_type == BACKEND_TYPE_LLAMA_CPP:
            potential_versions = await get_available_llama_cpp_versions(self.hass)

            schema = vol.Schema({
                vol.Required(CONF_LLAMACPP_REINSTALL, default=False): BooleanSelector(BooleanSelectorConfig()),
                vol.Required(CONF_INSTALLED_LLAMACPP_VERSION, default=client_config.get(CONF_INSTALLED_LLAMACPP_VERSION, "not installed")): SelectSelector(
                    SelectSelectorConfig(
                        options=[ SelectOptionDict(value=x[0], label=x[0] if not x[1] else f"{x[0]} (local)") for x in potential_versions ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                )
            })

            return self.async_show_form(
                step_id="reinstall",
                data_schema=schema,
            )
        else:

            if user_input is not None:
                client_config.update(user_input)

                # validate remote connections
                connect_err = await BACKEND_TO_CLS[backend_type].async_validate_connection(self.hass, client_config)

                if not connect_err:
                    return self.async_create_entry(data=client_config)
                else:
                    errors["base"] = "failed_to_connect"
                    description_placeholders["exception"] = str(connect_err)

            schema = remote_connection_schema(
                backend_type=backend_type,
                host=client_config.get(CONF_HOST),
                port=client_config.get(CONF_PORT),
                ssl=client_config.get(CONF_SSL),
                selected_path=client_config.get(CONF_GENERIC_OPENAI_PATH)
            )

            return self.async_show_form(
                step_id="init",
                data_schema=schema,
                errors=errors,
                description_placeholders=description_placeholders,
            )
    
    async def async_step_reinstall(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        client_config = dict(self.config_entry.options)

        if user_input is not None:
            if not user_input[CONF_LLAMACPP_REINSTALL]:
                _LOGGER.debug("Reinstall was not selected, finishing")
                return self.async_create_entry(data=client_config)
                
        if not self.reinstall_task:
            if not user_input:
                return self.async_abort(reason="unknown")
            
            desired_version = user_input.get(CONF_INSTALLED_LLAMACPP_VERSION)
            async def install_task():
                return await self.hass.async_add_executor_job(
                    install_llama_cpp_python, self.hass.config.config_dir, True, desired_version
                )

            self.reinstall_task = self.hass.async_create_background_task(
                install_task(), name="llama_cpp_python_installation")

            _LOGGER.debug("Queuing reinstall task...")
            return self.async_show_progress(
                progress_task=self.reinstall_task,
                step_id="reinstall",
                progress_action="install_local_wheels",
            )
        
        if not self.reinstall_task.done():
            return self.async_show_progress(
                progress_task=self.reinstall_task,
                step_id="reinstall",
                progress_action="install_local_wheels",
            )
        
        _LOGGER.debug("done... checking result")
        install_exception = self.reinstall_task.exception()
        if install_exception:
            self.wheel_install_error = repr(install_exception)
            _LOGGER.debug(f"Hit error: {self.wheel_install_error}")
            return self.async_show_progress_done(next_step_id="init")
        else:
            wheel_install_result = self.reinstall_task.result()
            if not wheel_install_result:
                self.wheel_install_error = "Pip returned false"
                _LOGGER.debug(f"Hit error: {self.wheel_install_error} ({wheel_install_result})")
                return self.async_show_progress_done(next_step_id="init")
            else:
                _LOGGER.debug(f"Finished install: {wheel_install_result}")
                self.wheel_install_successful = True
                return self.async_show_progress_done(next_step_id="init")
    

def STEP_LOCAL_MODEL_SELECTION_DATA_SCHEMA(model_file=None, chat_model=None, downloaded_model_quantization=None, available_quantizations=None):
    return vol.Schema(
        {
            vol.Optional(CONF_CHAT_MODEL, default=chat_model if chat_model else DEFAULT_CHAT_MODEL): SelectSelector(SelectSelectorConfig(
                options=RECOMMENDED_CHAT_MODELS,
                custom_value=True,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            vol.Optional(CONF_DOWNLOADED_MODEL_QUANTIZATION, default=downloaded_model_quantization if downloaded_model_quantization else DEFAULT_DOWNLOADED_MODEL_QUANTIZATION): vol.In(available_quantizations if available_quantizations else CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS),
            vol.Optional(CONF_DOWNLOADED_MODEL_FILE, default=model_file if model_file else ""): str,
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
    language: str,
    options: dict[str, Any],
    backend_type: str, 
    subentry_type: str,
) -> dict:

    default_prompt = build_prompt_template(language, DEFAULT_PROMPT)

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
        ): str,
        vol.Required(
            CONF_ENABLE_LEGACY_TOOL_CALLING,
            description={"suggested_value": options.get(CONF_ENABLE_LEGACY_TOOL_CALLING)},
            default=DEFAULT_ENABLE_LEGACY_TOOL_CALLING
        ): bool,
    }

    if backend_type == BACKEND_TYPE_LLAMA_CPP:
        result.update({
                vol.Required(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=DEFAULT_MAX_TOKENS,
            ): NumberSelector(NumberSelectorConfig(min=1, max=8192, step=1)),
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
            ): NumberSelector(NumberSelectorConfig(min=512, max=1_048_576, step=512)),
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
            ): NumberSelector(NumberSelectorConfig(min=512, max=1_048_576, step=512)),
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
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=DEFAULT_MAX_TOKENS,
            ): NumberSelector(NumberSelectorConfig(min=1, max=8192, step=1)),
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
        })
    elif backend_type == BACKEND_TYPE_OLLAMA:
        result.update({
            vol.Required(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=DEFAULT_MAX_TOKENS,
            ): NumberSelector(NumberSelectorConfig(min=1, max=8192, step=1)),
            vol.Required(
                CONF_CONTEXT_LENGTH,
                description={"suggested_value": options.get(CONF_CONTEXT_LENGTH)},
                default=DEFAULT_CONTEXT_LENGTH,
            ): NumberSelector(NumberSelectorConfig(min=512, max=1_048_576, step=512)),
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
    elif subentry_type == "ai_task_data":
        pass # no additional options for ai_task_data for now

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
        CONF_TYPICAL_P,
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
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        schema = vol.Schema({})
        errors = {}
        description_placeholders = {}
        entry = self._get_entry()

        backend_type = entry.data[CONF_BACKEND_TYPE]
        if backend_type == BACKEND_TYPE_LLAMA_CPP:
            schema = STEP_LOCAL_MODEL_SELECTION_DATA_SCHEMA()
        else:
            schema = STEP_REMOTE_MODEL_SELECTION_DATA_SCHEMA(await entry.runtime_data.async_get_available_models())

        if self.download_error:
            if isinstance(self.download_error, MissingQuantizationException):
                available_quants = list(set(self.download_error.available_quants).intersection(set(CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS)))

                if len(available_quants) == 0:
                    errors["base"] = "no_supported_ggufs"
                    schema = STEP_LOCAL_MODEL_SELECTION_DATA_SCHEMA(
                        chat_model=self.model_config[CONF_CHAT_MODEL],
                        downloaded_model_quantization=self.model_config[CONF_DOWNLOADED_MODEL_QUANTIZATION],
                    )
                else:
                    errors["base"] = "missing_quantization"
                    description_placeholders["missing"] = self.download_error.missing_quant
                    description_placeholders["available"] = ", ".join(self.download_error.available_quants)

                    schema = STEP_LOCAL_MODEL_SELECTION_DATA_SCHEMA(
                        chat_model=self.model_config[CONF_CHAT_MODEL],
                        downloaded_model_quantization=self.download_error.available_quants[0],
                        available_quantizations=available_quants,
                    )
            else:
                errors["base"] = "download_failed"
                description_placeholders["exception"] = str(self.download_error)
                schema = STEP_LOCAL_MODEL_SELECTION_DATA_SCHEMA(
                    chat_model=self.model_config[CONF_CHAT_MODEL],
                    downloaded_model_quantization=self.model_config[CONF_DOWNLOADED_MODEL_QUANTIZATION],
                )

        if user_input and "result" not in user_input:

            self.model_config.update(user_input)

            if backend_type == BACKEND_TYPE_LLAMA_CPP:
                model_file = self.model_config.get(CONF_DOWNLOADED_MODEL_FILE, "")
                if not model_file:
                    model_name = self.model_config.get(CONF_CHAT_MODEL)
                    if model_name:
                        return await self.async_step_download()
                    else:
                        errors["base"] = "no_model_name_or_file"
                
                if os.path.exists(model_file):
                    self.model_config[CONF_CHAT_MODEL] = os.path.basename(model_file)
                    return await self.async_step_model_parameters()
                else:
                    errors["base"] = "missing_model_file"
                    schema = STEP_LOCAL_MODEL_SELECTION_DATA_SCHEMA(model_file)
            else:
                return await self.async_step_model_parameters()

        return self.async_show_form(
            step_id="pick_model",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
            last_step=False,
        )

    async def async_step_download(
        self, user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        if not self.download_task:
            model_name = self.model_config[CONF_CHAT_MODEL]
            quantization_type = self.model_config[CONF_DOWNLOADED_MODEL_QUANTIZATION]
            storage_folder = os.path.join(self.hass.config.media_dirs.get("local", self.hass.config.path("media")), "models")

            async def download_task():
                return await self.hass.async_add_executor_job(
                    download_model_from_hf, model_name, quantization_type, storage_folder
                )

            self.download_task = self.hass.async_create_background_task(
                download_task(), name="model_download_task")

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

            self.download_task = None
            return self.async_show_progress_done(next_step_id="pick_model")
        else:
            self.model_config[CONF_DOWNLOADED_MODEL_FILE] = self.download_task.result()

            self.download_task = None
            return self.async_show_progress_done(next_step_id="model_parameters")

    async def async_step_model_parameters(
        self, user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        errors = {}
        description_placeholders = {}
        entry = self._get_entry()
        backend_type = entry.data[CONF_BACKEND_TYPE]

        if CONF_PROMPT not in self.model_config:
            # determine selected language from model config or parent options
            selected_language = self.model_config.get(
                CONF_SELECTED_LANGUAGE, entry.options.get(CONF_SELECTED_LANGUAGE, "en")
            )
            model_name = self.model_config.get(CONF_CHAT_MODEL, "").lower()

            OPTIONS_OVERRIDES = option_overrides(backend_type)

            selected_default_options = {**DEFAULT_OPTIONS}
            for key in OPTIONS_OVERRIDES.keys():
                if key in model_name:
                    selected_default_options.update(OPTIONS_OVERRIDES[key])
                    break

            # Build prompt template using the selected language
            selected_default_options[CONF_PROMPT] = build_prompt_template(
                selected_language, str(selected_default_options.get(CONF_PROMPT, DEFAULT_PROMPT))
            )
            
            self.model_config = {**selected_default_options, **self.model_config}

        schema = vol.Schema(
            local_llama_config_option_schema(
                self.hass,
                entry.options.get(CONF_SELECTED_LANGUAGE, "en"),
                self.model_config,
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

            # --- Normalize numeric fields to ints to avoid slice/type errors later ---
            for key in (
                CONF_REMEMBER_NUM_INTERACTIONS,
                CONF_MAX_TOOL_CALL_ITERATIONS,
                CONF_CONTEXT_LENGTH,
                CONF_MAX_TOKENS,
                CONF_REQUEST_TIMEOUT,
             ):
                if key in user_input:
                    user_input[key] = _coerce_int(user_input[key], user_input.get(key) or 0)
            
            if len(errors) == 0:
                try:
                    # validate input
                    schema(user_input)
                    self.model_config.update(user_input)

                    # clear LLM API if 'none' selected
                    if self.model_config.get(CONF_LLM_HASS_API) == "none":
                        self.model_config.pop(CONF_LLM_HASS_API, None)
                    
                    return await self.async_step_finish()
                except Exception:
                    _LOGGER.exception("An unknown error has occurred!")
                    errors["base"] = "unknown"

        return self.async_show_form(
            step_id="model_parameters",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )

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

        # Ensure the parent entry is loaded before allowing subentry edits
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        
        if not self.model_config:
            self.model_config = {}
        
        return await self.async_step_pick_model(user_input)
    
    async_step_init = async_step_user
        
    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None):
        if not self.model_config:
            self.model_config = dict(self._get_reconfigure_subentry().data)

        return await self.async_step_model_parameters(user_input)
