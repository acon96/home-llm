"""Constants for the LLaMa Conversation integration."""

DOMAIN = "llama_conversation"
CONF_PROMPT = "prompt"
DEFAULT_PROMPT = """You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task ask instructed with the information provided only.
Services: {{ services }}
Devices:
{{ devices }}"""
CONF_CHAT_MODEL = "huggingface_model"
DEFAULT_CHAT_MODEL = "microsoft/phi-2"
CONF_MAX_TOKENS = "max_new_tokens"
DEFAULT_MAX_TOKENS = 128
CONF_TOP_K = "top_k"
DEFAULT_TOP_K = 40
CONF_TOP_P = "top_p"
DEFAULT_TOP_P = 1
CONF_TEMPERATURE = "temperature"
DEFAULT_TEMPERATURE = 0.1
CONF_USE_LOCAL_BACKEND = "use_local_backend"
DEFAULT_USE_LOCAL_BACKEND = True
CONF_DOWNLOADED_MODEL_FILE = "downloaded_model_file"
DEFAULT_DOWNLOADED_MODEL_FILE = ""
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "5000"
