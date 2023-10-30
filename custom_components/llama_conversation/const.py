"""Constants for the LLaMa Conversation integration."""

DOMAIN = "llama_conversation"
CONF_PROMPT = "prompt"
DEFAULT_PROMPT = """You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task ask instructed with the information provided only.
Services: {{ services }}
Devices:
{{ devices }}
Request: {{ user_input }}
Response:"""
CONF_CHAT_MODEL = "huggingface_model"
DEFAULT_CHAT_MODEL = "microsoft/phi-1.5"
CONF_MAX_TOKENS = "max_new_tokens"
DEFAULT_MAX_TOKENS = 128
CONF_TOP_P = "top_p"
DEFAULT_TOP_P = 1
CONF_TEMPERATURE = "temperature"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "5000"
