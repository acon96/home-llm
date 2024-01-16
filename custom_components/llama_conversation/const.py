"""Constants for the LLaMa Conversation integration."""

DOMAIN = "llama_conversation"
CONF_PROMPT = "prompt"
DEFAULT_PROMPT = """You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task ask instructed with the information provided only.
Services: {{ services }}
Devices:
{{ devices }}"""
CONF_CHAT_MODEL = "huggingface_model"
DEFAULT_CHAT_MODEL = "TheBloke/phi-2-GGUF" # "microsoft/phi-2"
CONF_MAX_TOKENS = "max_new_tokens"
DEFAULT_MAX_TOKENS = 128
CONF_TOP_K = "top_k"
DEFAULT_TOP_K = 40
CONF_TOP_P = "top_p"
DEFAULT_TOP_P = 1
CONF_TEMPERATURE = "temperature"
DEFAULT_TEMPERATURE = 0.1
CONF_REQUEST_TIMEOUT = "request_timeout"
DEFAULT_REQUEST_TIMEOUT = 90
CONF_BACKEND_TYPE = "model_backend"
BACKEND_TYPE_LLAMA_HF = "llama_cpp_hf"
BACKEND_TYPE_LLAMA_EXISTING = "llama_cpp_existing"
BACKEND_TYPE_TEXT_GEN_WEBUI = "text-generation-webui_api"
BACKEND_TYPE_GENERIC_OPENAI = "generic_openai"
DEFAULT_BACKEND_TYPE = BACKEND_TYPE_LLAMA_HF
CONF_DOWNLOADED_MODEL_QUANTIZATION = "downloaded_model_quantization"
CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS = ["Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
DEFAULT_DOWNLOADED_MODEL_QUANTIZATION = "Q5_K_M"
CONF_DOWNLOADED_MODEL_FILE = "downloaded_model_file"
DEFAULT_DOWNLOADED_MODEL_FILE = ""
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "5000"
CONF_EXTRA_ATTRIBUTES_TO_EXPOSE = "extra_attributes_to_expose"
DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE = ["rgb_color", "current_temperature", "fan_mode", "media_title", "volume_level" ]
GBNF_GRAMMAR_FILE = "output.gbnf"
CONF_PROMPT_TEMPLATE = "prompt_template"
PROMPT_TEMPLATE_CHATML = "chatml"
PROMPT_TEMPLATE_ALPACA = "alpaca"
PROMPT_TEMPLATE_VICUNA = "vicuna"
PROMPT_TEMPLATE_NONE = "no_prompt_template"
DEFAULT_PROMPT_TEMPLATE = PROMPT_TEMPLATE_CHATML
PROMPT_TEMPLATE_DESCRIPTIONS = {
    PROMPT_TEMPLATE_CHATML: {
        "system": { "prefix": "<|im_start|>system\n", "suffix": "<|im_end|>" },
        "user": { "prefix": "<|im_start|>user\n", "suffix": "<|im_end|>" },
        "assistant": { "prefix": "<|im_start|>assistant\n", "suffix": "" },
        "generation_prompt": "<|im_start|>assistant"
    },
    PROMPT_TEMPLATE_ALPACA: {
        "system": { "prefix": "", "suffix": "\n" },
        "user": { "prefix": "### Instruction:\n", "suffix": "\n" },
        "assistant": { "prefix": "### Response:\n", "suffix": "\n" },
        "generation_prompt": "### Response:"
    },
    PROMPT_TEMPLATE_VICUNA: {
        "system": { "prefix": "", "suffix": "\n" },
        "user": { "prefix": "USER: ", "suffix": "" },
        "assistant": { "prefix": "ASSISTANT: ", "suffix": "</s>" },
        "generation_prompt": "ASSISTANT:"
    },
    PROMPT_TEMPLATE_NONE: {
        "system": { "prefix": "", "suffix": "" },
        "user": { "prefix": "", "suffix": "" },
        "assistant": { "prefix": "", "suffix": "" },
        "generation_prompt": ""
    }
}
CONF_USE_GBNF_GRAMMAR = "gbnf_grammar"
DEFAULT_USE_GBNF_GRAMMAR = False
CONF_TEXT_GEN_WEBUI_PRESET = "text_generation_webui_preset"
CONF_OPENAI_API_KEY = "openai_api_key"
CONF_TEXT_GEN_WEBUI_ADMIN_KEY = "text_generation_webui_admin_key"
CONF_REFRESH_SYSTEM_PROMPT = "refresh_prompt_per_tern"
DEFAULT_REFRESH_SYSTEM_PROMPT = True
CONF_SERVICE_CALL_REGEX = "service_call_regex"
DEFAULT_SERVICE_CALL_REGEX = r"```homeassistant\n([\S \t\n]*?)```"