"""Constants for the Local LLM Conversation integration."""
import types, os
from typing import Any

DOMAIN = "llama_conversation"
HOME_LLM_API_ID = "home-llm-service-api"
SERVICE_TOOL_NAME = "HassCallService"
SERVICE_TOOL_ALLOWED_SERVICES = ["turn_on", "turn_off", "toggle", "press", "increase_speed", "decrease_speed", "open_cover", "close_cover", "stop_cover", "lock", "unlock", "start", "stop", "return_to_base", "pause", "cancel", "add_item", "set_temperature", "set_humidity", "set_fan_mode", "set_hvac_mode", "set_preset_mode"]
SERVICE_TOOL_ALLOWED_DOMAINS = ["light", "switch", "button", "fan", "cover", "lock", "media_player", "climate", "vacuum", "todo", "timer", "script"]
CONF_PROMPT = "prompt"
PERSONA_PROMPTS = {
    "en": "You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.",
    "de": "Du bist \u201eAl\u201c, ein hilfreicher KI-Assistent, der die Ger\u00e4te in einem Haus steuert. F\u00fchren Sie die folgende Aufgabe gem\u00e4\u00df den Anweisungen durch oder beantworten Sie die folgende Frage nur mit den bereitgestellten Informationen.",
    "fr": "Vous \u00eates \u00ab\u00a0Al\u00a0\u00bb, un assistant IA utile qui contr\u00f4le les appareils d'une maison. Effectuez la t\u00e2che suivante comme indiqu\u00e9 ou r\u00e9pondez \u00e0 la question suivante avec les informations fournies uniquement.",
    "es": "Eres 'Al', un \u00fatil asistente de IA que controla los dispositivos de una casa. Complete la siguiente tarea seg\u00fan las instrucciones o responda la siguiente pregunta \u00fanicamente con la informaci\u00f3n proporcionada.",
    "pl": "Jeste\u015b 'Al', pomocnym asystentem AI, kt\u00f3ry kontroluje urz\u0105dzenia w domu. Wykonaj poni\u017csze zadanie zgodnie z instrukcj\u0105 lub odpowiedz na poni\u017csze pytanie, korzystaj\u0105c wy\u0142\u0105cznie z podanych informacji."
}
CURRENT_DATE_PROMPT = {
    "en": """The current time and date is {{ (as_timestamp(now()) | timestamp_custom("%I:%M %p on %A %B %d, %Y", True, "")) }}""",
    "de": """{% set day_name = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"] %}{% set month_name = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"] %}Die aktuelle Uhrzeit und das aktuelle Datum sind {{ (as_timestamp(now()) | timestamp_custom("%H:%M", local=True)) }} {{ day_name[now().weekday()] }}, {{ now().day }} {{ month_name[now().month -1]}} {{ now().year }}.""",
    "fr": """{% set day_name = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"] %}{% set month_name = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"] %} L'heure et la date actuelles sont {{ (as_timestamp(now()) | timestamp_custom("%H:%M", local=True)) }} {{ day_name[now().weekday()] }}, {{ now().day }} {{ month_name[now().month -1]}} {{ now().year }}.""",
    "es": """{% set day_name = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"] %}{% set month_name = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"] %}La hora y fecha actuales son {{ (as_timestamp(now()) | timestamp_custom("%H:%M", local=True)) }} {{ day_name[now().weekday()] }}, {{ now().day }} de {{ month_name[now().month -1]}} de {{ now().year }}.""",
    "pl": """{% set day_name = ["poniedziałek", "wtorek", "środę", "czwartek", "piątek", "sobotę", "niedzielę"] %}{% set month_name = ["styczeń", "luty", "marzec", "kwiecień", "maj", "czerwiec", "lipiec", "sierpień", "wrzesień", "październik", "listopad", "grudzień"] %}Aktualna godzina i data to {{ (as_timestamp(now()) | timestamp_custom("%H:%M", local=True)) }} w {{ day_name[now().weekday()] }}, {{ now().day }} {{ month_name[now().month -1]}} {{ now().year }}."""
}
DEVICES_PROMPT = {
    "en": "Devices",
    "de": "Ger\u00e4te",
    "fr": "Appareils",
    "es": "Dispositivos",
    "pl": "Urządzenia",
}
SERVICES_PROMPT = {
    "en": "Services",
    "de": "Dienste",
    "fr": "Services",
    "es": "Servicios",
    "pl": "Usługi",
}
TOOLS_PROMPT = {
    "en": "Tools",
    "de": "Werkzeuge",
    "fr": "Outils",
    "es": "Herramientas",
    "pl": "Narzędzia",
}
AREA_PROMPT = {
    "en": "Area",
    "de": "Bereich",
    "fr": "Zone",
    "es": "Área",
    "pl": "Obszar",
}
USER_INSTRUCTION = {
    "en": "User instruction",
    "de": "Benutzeranweisung",
    "fr": "Instruction de l'utilisateur ",
    "es": "Instrucción del usuario",
    "pl": "Instrukcja użytkownika"
}
DEFAULT_PROMPT_BASE = """<persona>
<current_date>
<devices>:
{% for device in devices | selectattr('area_id', 'none'): %}
{{ device.entity_id }} '{{ device.name }}' = {{ device.state }}{{ ([""] + device.attributes) | join(";") }}
{% endfor %}
{% for area in devices | rejectattr('area_id', 'none') | groupby('area_name') %}
## <area>: {{ area.grouper }}
{% for device in area.list %}
{{ device.entity_id }} '{{ device.name }}' = {{ device.state }};{{ device.attributes | join(";") }}
{% endfor %}
{% endfor %}"""
DEFAULT_PROMPT_BASE_LEGACY = """<persona>
<current_date>
<devices>:
{{ formatted_devices }}"""
ICL_EXTRAS = """
{% for item in response_examples %}
{{ item.request }}
{{ item.response }}
{{ tool_call_prefix }}{{ item.tool | to_json }}{{ tool_call_suffix }}
{% endfor %}"""
ICL_NO_SYSTEM_PROMPT_EXTRAS = """
{% for item in response_examples %}
{{ item.request }}
{{ item.response }}
{{ tool_call_prefix }}{{ item.tool | to_json }}{{ tool_call_suffix }}
{% endfor %}
<user_instruction>:"""
DEFAULT_PROMPT = DEFAULT_PROMPT_BASE + ICL_EXTRAS
CONF_CHAT_MODEL = "huggingface_model"
DEFAULT_CHAT_MODEL = "acon96/Home-3B-v3-GGUF"
RECOMMENDED_CHAT_MODELS = [ "acon96/Home-3B-v3-GGUF", "acon96/Home-1B-v3-GGUF", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" ]
CONF_MAX_TOKENS = "max_new_tokens"
DEFAULT_MAX_TOKENS = 512
CONF_TOP_K = "top_k"
DEFAULT_TOP_K = 40
CONF_TOP_P = "top_p"
DEFAULT_TOP_P = 1
CONF_TYPICAL_P = "typical_p"
DEFAULT_TYPICAL_P = 1.0
CONF_MIN_P = "min_p"
DEFAULT_MIN_P = 0.0
CONF_TEMPERATURE = "temperature"
DEFAULT_TEMPERATURE = 0.1
CONF_REQUEST_TIMEOUT = "request_timeout"
DEFAULT_REQUEST_TIMEOUT = 90
CONF_BACKEND_TYPE = "model_backend"
BACKEND_TYPE_LLAMA_HF_OLD = "llama_cpp_hf"
BACKEND_TYPE_LLAMA_EXISTING_OLD = "llama_cpp_existing"
BACKEND_TYPE_LLAMA_CPP = "llama_cpp_python"
BACKEND_TYPE_TEXT_GEN_WEBUI = "text-generation-webui_api"
BACKEND_TYPE_GENERIC_OPENAI = "generic_openai"
BACKEND_TYPE_GENERIC_OPENAI_RESPONSES = "generic_openai_responses"
BACKEND_TYPE_LLAMA_CPP_SERVER = "llama_cpp_server"
BACKEND_TYPE_OLLAMA = "ollama"
DEFAULT_BACKEND_TYPE = BACKEND_TYPE_LLAMA_CPP
CONF_INSTALLED_LLAMACPP_VERSION = "installed_llama_cpp_version"
CONF_SELECTED_LANGUAGE = "selected_language"
CONF_SELECTED_LANGUAGE_OPTIONS = [ "en", "de", "fr", "es", "pl"]
CONF_DOWNLOADED_MODEL_QUANTIZATION = "downloaded_model_quantization"
CONF_DOWNLOADED_MODEL_QUANTIZATION_OPTIONS = [
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M",
    "Q2_K", "Q2_K_S", "IQ3_XXS", "IQ3_S", "IQ3_M", "Q3_K", "IQ3_XS", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "IQ4_NL", "IQ4_XS", "Q4_K", "Q4_K_S", "Q4_K_M", "Q5_K", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
    "F16", "BF16", "F32"
]
DEFAULT_DOWNLOADED_MODEL_QUANTIZATION = "Q4_K_M"
CONF_DOWNLOADED_MODEL_FILE = "downloaded_model_file"
DEFAULT_DOWNLOADED_MODEL_FILE = ""
DEFAULT_PORT = "5000"
DEFAULT_SSL = False
CONF_EXTRA_ATTRIBUTES_TO_EXPOSE = "extra_attributes_to_expose"
DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE = ["rgb_color", "brightness", "temperature", "humidity", "fan_mode", "media_title", "volume_level", "item", "wind_speed"]
ALLOWED_SERVICE_CALL_ARGUMENTS = ["rgb_color", "brightness", "temperature", "humidity", "fan_mode", "hvac_mode", "preset_mode", "item", "duration" ]
CONF_THINKING_PREFIX = "thinking_prefix"
DEFAULT_THINKING_PREFIX = "<think>"
CONF_THINKING_SUFFIX = "thinking_suffix"
DEFAULT_THINKING_SUFFIX = "</think>"
CONF_TOOL_CALL_PREFIX = "tool_call_prefix"
DEFAULT_TOOL_CALL_PREFIX = "<tool_call>"
CONF_TOOL_CALL_SUFFIX = "tool_call_suffix"
DEFAULT_TOOL_CALL_SUFFIX = "</tool_call>"
CONF_ENABLE_LEGACY_TOOL_CALLING = "enable_legacy_tool_calling"
DEFAULT_ENABLE_LEGACY_TOOL_CALLING = False
CONF_LLAMACPP_ENABLE_FLASH_ATTENTION = "enable_flash_attention"
DEFAULT_LLAMACPP_ENABLE_FLASH_ATTENTION = False
CONF_USE_GBNF_GRAMMAR = "gbnf_grammar"
DEFAULT_USE_GBNF_GRAMMAR = False
CONF_GBNF_GRAMMAR_FILE = "gbnf_grammar_file"
DEFAULT_GBNF_GRAMMAR_FILE = "output.gbnf"
CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES = "in_context_examples"
DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES = True
CONF_IN_CONTEXT_EXAMPLES_FILE = "in_context_examples_file"
DEFAULT_IN_CONTEXT_EXAMPLES_FILE = "in_context_examples.csv"
CONF_NUM_IN_CONTEXT_EXAMPLES = "num_in_context_examples"
DEFAULT_NUM_IN_CONTEXT_EXAMPLES = 4
CONF_TEXT_GEN_WEBUI_PRESET = "text_generation_webui_preset"
CONF_OPENAI_API_KEY = "openai_api_key"
CONF_TEXT_GEN_WEBUI_ADMIN_KEY = "text_generation_webui_admin_key"
CONF_REFRESH_SYSTEM_PROMPT = "refresh_prompt_per_turn"
DEFAULT_REFRESH_SYSTEM_PROMPT = True
CONF_REMEMBER_CONVERSATION = "remember_conversation"
DEFAULT_REMEMBER_CONVERSATION = True
CONF_REMEMBER_NUM_INTERACTIONS = "remember_num_interactions"
DEFAULT_REMEMBER_NUM_INTERACTIONS = 5
CONF_REMEMBER_CONVERSATION_TIME_MINUTES = "remember_conversation_time_minutes"
DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES = 2
CONF_MAX_TOOL_CALL_ITERATIONS = "max_tool_call_iterations"
DEFAULT_MAX_TOOL_CALL_ITERATIONS = 3
CONF_PROMPT_CACHING_ENABLED = "prompt_caching"
DEFAULT_PROMPT_CACHING_ENABLED = False
CONF_PROMPT_CACHING_INTERVAL = "prompt_caching_interval"
DEFAULT_PROMPT_CACHING_INTERVAL = 30
CONF_TEXT_GEN_WEBUI_CHAT_MODE = "text_generation_webui_chat_mode"
TEXT_GEN_WEBUI_CHAT_MODE_CHAT = "chat"
TEXT_GEN_WEBUI_CHAT_MODE_INSTRUCT = "instruct"
TEXT_GEN_WEBUI_CHAT_MODE_CHAT_INSTRUCT = "chat-instruct"
DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE = TEXT_GEN_WEBUI_CHAT_MODE_CHAT
CONF_OLLAMA_KEEP_ALIVE_MIN = "ollama_keep_alive"
DEFAULT_OLLAMA_KEEP_ALIVE_MIN = 30
CONF_OLLAMA_JSON_MODE = "ollama_json_mode"
DEFAULT_OLLAMA_JSON_MODE = False
CONF_GENERIC_OPENAI_PATH = "openai_path"
DEFAULT_GENERIC_OPENAI_PATH = "v1"
CONF_GENERIC_OPENAI_VALIDATE_MODEL = "openai_validate_model"
DEFAULT_GENERIC_OPENAI_VALIDATE_MODEL = True
CONF_CONTEXT_LENGTH = "context_length"
DEFAULT_CONTEXT_LENGTH = 2048
CONF_LLAMACPP_BATCH_SIZE = "batch_size"
DEFAULT_LLAMACPP_BATCH_SIZE = 512
CONF_LLAMACPP_THREAD_COUNT = "n_threads"
DEFAULT_LLAMACPP_THREAD_COUNT = os.cpu_count()
CONF_LLAMACPP_BATCH_THREAD_COUNT = "n_batch_threads"
DEFAULT_LLAMACPP_BATCH_THREAD_COUNT = os.cpu_count()
CONF_LLAMACPP_REINSTALL = "reinstall_llama_cpp"

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_TOP_K: DEFAULT_TOP_K,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_MIN_P: DEFAULT_MIN_P,
        CONF_TYPICAL_P: DEFAULT_TYPICAL_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_REQUEST_TIMEOUT: DEFAULT_REQUEST_TIMEOUT,
        CONF_LLAMACPP_ENABLE_FLASH_ATTENTION: DEFAULT_LLAMACPP_ENABLE_FLASH_ATTENTION,
        CONF_USE_GBNF_GRAMMAR: DEFAULT_USE_GBNF_GRAMMAR,
        CONF_EXTRA_ATTRIBUTES_TO_EXPOSE: DEFAULT_EXTRA_ATTRIBUTES_TO_EXPOSE,
        CONF_REFRESH_SYSTEM_PROMPT: DEFAULT_REFRESH_SYSTEM_PROMPT,
        CONF_REMEMBER_CONVERSATION: DEFAULT_REMEMBER_CONVERSATION,
        CONF_REMEMBER_NUM_INTERACTIONS: DEFAULT_REMEMBER_NUM_INTERACTIONS,
        CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: DEFAULT_USE_IN_CONTEXT_LEARNING_EXAMPLES,
        CONF_IN_CONTEXT_EXAMPLES_FILE: DEFAULT_IN_CONTEXT_EXAMPLES_FILE,
        CONF_NUM_IN_CONTEXT_EXAMPLES: DEFAULT_NUM_IN_CONTEXT_EXAMPLES,
        CONF_CONTEXT_LENGTH: DEFAULT_CONTEXT_LENGTH,
        CONF_LLAMACPP_BATCH_SIZE: DEFAULT_LLAMACPP_BATCH_SIZE,
        CONF_LLAMACPP_THREAD_COUNT: DEFAULT_LLAMACPP_THREAD_COUNT,
        CONF_LLAMACPP_BATCH_THREAD_COUNT: DEFAULT_LLAMACPP_BATCH_THREAD_COUNT,
        CONF_PROMPT_CACHING_ENABLED: DEFAULT_PROMPT_CACHING_ENABLED,
        CONF_OLLAMA_KEEP_ALIVE_MIN: DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
        CONF_OLLAMA_JSON_MODE: DEFAULT_OLLAMA_JSON_MODE,
        CONF_TEXT_GEN_WEBUI_CHAT_MODE: DEFAULT_TEXT_GEN_WEBUI_CHAT_MODE,
        CONF_TEXT_GEN_WEBUI_PRESET: ""
    }
)

def option_overrides(backend_type: str) -> dict[str, Any]:
    return {
        "home-llama-3.2": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE_LEGACY,
            CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: False,
            CONF_TOOL_CALL_PREFIX: "```homeassistant",
            CONF_TOOL_CALL_SUFFIX: "```",
            CONF_CONTEXT_LENGTH: 131072,
            CONF_MAX_TOOL_CALL_ITERATIONS: 0,
            # llama cpp server doesn't support custom tool calling formats. so just use legacy tool calling
            CONF_ENABLE_LEGACY_TOOL_CALLING: backend_type == BACKEND_TYPE_LLAMA_CPP_SERVER
        },
        "home-3b-v3": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE_LEGACY,
            CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: False,
            CONF_TOOL_CALL_PREFIX: "```homeassistant",
            CONF_TOOL_CALL_SUFFIX: "```",
            CONF_MAX_TOOL_CALL_ITERATIONS: 0,
            # llama cpp server doesn't support custom tool calling formats. so just use legacy tool calling
            CONF_ENABLE_LEGACY_TOOL_CALLING: backend_type == BACKEND_TYPE_LLAMA_CPP_SERVER
        },
        "home-3b-v2": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE_LEGACY,
            CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: False,
            CONF_TOOL_CALL_PREFIX: "```homeassistant",
            CONF_TOOL_CALL_SUFFIX: "```",
            CONF_MAX_TOOL_CALL_ITERATIONS: 0,
            # no prompt formats with tool calling support, so just use legacy tool calling
            CONF_ENABLE_LEGACY_TOOL_CALLING: True
        },
        "home-3b-v1": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE_LEGACY,
            CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: False,
            CONF_TOOL_CALL_PREFIX: "```homeassistant",
            CONF_TOOL_CALL_SUFFIX: "```",
            CONF_MAX_TOOL_CALL_ITERATIONS: 0,
            # no prompt formats with tool calling support, so just use legacy tool calling
            CONF_ENABLE_LEGACY_TOOL_CALLING: True
        },
        "home-1b-v3": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE_LEGACY,
            CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: False,
            CONF_TOOL_CALL_PREFIX: "```homeassistant",
            CONF_TOOL_CALL_SUFFIX: "```",
            CONF_MAX_TOOL_CALL_ITERATIONS: 0,
            # no prompt formats with tool calling support, so just use legacy tool calling
            CONF_ENABLE_LEGACY_TOOL_CALLING: True
        },
        "home-1b-v2": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE_LEGACY,
            CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: False,
            CONF_TOOL_CALL_PREFIX: "```homeassistant",
            CONF_TOOL_CALL_SUFFIX: "```",
            CONF_MAX_TOOL_CALL_ITERATIONS: 0,
            # no prompt formats with tool calling support, so just use legacy tool calling
            CONF_ENABLE_LEGACY_TOOL_CALLING: True
        },
        "home-1b-v1": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE_LEGACY,
            CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: False,
            CONF_TOOL_CALL_PREFIX: "```homeassistant",
            CONF_TOOL_CALL_SUFFIX: "```",
            CONF_MAX_TOOL_CALL_ITERATIONS: 0,
            # no prompt formats with tool calling support, so just use legacy tool calling
            CONF_ENABLE_LEGACY_TOOL_CALLING: True
        },
        "qwen3": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE,
            CONF_TEMPERATURE: 0.6,
            CONF_TOP_K: 20,
            CONF_TOP_P: 0.95
        },
        "mistral": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE + ICL_NO_SYSTEM_PROMPT_EXTRAS,
            CONF_MIN_P: 0.1,
            CONF_TYPICAL_P: 0.9,
            # no prompt formats with tool calling support, so just use legacy tool calling
            CONF_ENABLE_LEGACY_TOOL_CALLING: True,
        },
        "mixtral": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE + ICL_NO_SYSTEM_PROMPT_EXTRAS,
            CONF_MIN_P: 0.1,
            CONF_TYPICAL_P: 0.9,
            # no prompt formats with tool calling support, so just use legacy tool calling
            CONF_ENABLE_LEGACY_TOOL_CALLING: True,
        },
        "llama-3": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE + ICL_EXTRAS,
        },
        "llama3": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE + ICL_EXTRAS,
        },
        "zephyr": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE + ICL_EXTRAS,
            
        },
        "phi-3": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE + ICL_EXTRAS,
        },
        "command-r": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE + ICL_EXTRAS,
        },
        "stablehome": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE_LEGACY,
            CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: False,
        },
        "tinyhome": {
            CONF_PROMPT: DEFAULT_PROMPT_BASE_LEGACY,
            CONF_USE_IN_CONTEXT_LEARNING_EXAMPLES: False,
        },
    }

INTEGRATION_VERSION = "0.4.4"
EMBEDDED_LLAMA_CPP_PYTHON_VERSION = "0.3.16+b6153"
