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
BACKEND_TYPE_LLAMA_HF = "Llama.cpp (HuggingFace)"
BACKEND_TYPE_LLAMA_EXISTING = "Llama.cpp (existing model)"
BACKEND_TYPE_REMOTE = "text-generation-webui API"
DEFAULT_BACKEND_TYPE = BACKEND_TYPE_LLAMA_HF
CONF_BACKEND_TYPE_OPTIONS = [ BACKEND_TYPE_LLAMA_HF, BACKEND_TYPE_LLAMA_EXISTING, BACKEND_TYPE_REMOTE ]
CONF_DOWNLOADED_MODEL_QUANTIZATION = "downloaded_model_quantization"
CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS = ["Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
DEFAULT_DOWNLOADED_MODEL_QUANTIZATION = "Q5_K_M"
CONF_DOWNLOADED_MODEL_FILE = "downloaded_model_file"
DEFAULT_DOWNLOADED_MODEL_FILE = ""
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "5000"
CONF_EXTRA_ATTRIBUTES_TO_EXPOSE = "extra_attributes_to_expose"
DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE = "rgb_color,current_temperature,fan_mode,media_title,volume_level"
GBNF_GRAMMAR_FILE = "output.gbnf"