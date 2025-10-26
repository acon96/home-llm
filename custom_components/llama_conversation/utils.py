import time
import os
import re
import ipaddress
import sys
import platform
import logging
import multiprocessing
import voluptuous as vol
import webcolors
import json
from typing import Any, Dict, List, Sequence, Tuple, cast
from webcolors import CSS3
from importlib.metadata import version

from homeassistant.core import HomeAssistant
from homeassistant.components import conversation
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import intent, llm, aiohttp_client
from homeassistant.requirements import pip_kwargs
from homeassistant.util import color
from homeassistant.util.package import install_package, is_installed

from voluptuous_openapi import convert

from .const import (
    EMBEDDED_LLAMA_CPP_PYTHON_VERSION,
    ALLOWED_SERVICE_CALL_ARGUMENTS,
    SERVICE_TOOL_ALLOWED_SERVICES,
    SERVICE_TOOL_ALLOWED_DOMAINS,
    HOME_LLM_API_ID,
    SERVICE_TOOL_NAME
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llama_cpp.llama_types import ChatCompletionRequestMessage, ChatCompletionTool
else:
    ChatCompletionRequestMessage = Any
    ChatCompletionTool = Any

_LOGGER = logging.getLogger(__name__)

CSS3_NAME_TO_RGB = {
    name: webcolors.name_to_rgb(name, CSS3)
    for name
    in webcolors.names(CSS3)
}

class MissingQuantizationException(Exception):
    def __init__(self, missing_quant: str, available_quants: list[str]):
        super().__init__(missing_quant, available_quants)
        self.missing_quant = missing_quant
        self.available_quants = available_quants

class MalformedToolCallException(Exception):
    def __init__(self, agent_id: str, tool_call_id: str, tool_name: str, tool_args: str, error_msg: str):
        super().__init__(agent_id, tool_call_id, tool_name, tool_args, error_msg)
        self.agent_id = agent_id
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.error_msg = error_msg

    def as_tool_messages(self) -> Sequence[conversation.Content]:
        return [
            conversation.AssistantContent(
                self.agent_id, tool_calls=[llm.ToolInput(self.tool_name, {})]
            ),
            conversation.ToolResultContent(
            self.agent_id, self.tool_call_id, self.tool_name, 
            {"error": f"Error occurred calling tool with args='{self.tool_args}': {self.error_msg}" }
        )]

def closest_color(requested_color):
    min_colors = {}
    
    for name, rgb in CSS3_NAME_TO_RGB.items():
        r_c, g_c, b_c = rgb
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def flatten_vol_schema(schema):
    flattened = []
    def _flatten(current_schema, prefix=''):
        if isinstance(current_schema, vol.Schema):
            if isinstance(current_schema.schema, vol.validators._WithSubValidators):
                for subval in current_schema.schema.validators:
                    _flatten(subval, prefix)
            elif isinstance(current_schema.schema, dict):
                for key, val in current_schema.schema.items():
                    _flatten(val, prefix + str(key) + '/')
        elif isinstance(current_schema, vol.validators._WithSubValidators):
            for subval in current_schema.validators:
                _flatten(subval, prefix)
        elif callable(current_schema):
            flattened.append(prefix[:-1] if prefix else prefix)
    _flatten(schema)
    return flattened

def custom_custom_serializer(value):
    """a vol schema is really not straightforward to convert back into a dictionary"""

    if value is cv.ensure_list:
        return { "type": "list" }
    
    if value is color.color_name_to_rgb:
        return { "type": "string" }
    
    if value is intent.non_empty_string:
        return { "type": "string" }
    
    # media player registers an intent using a lambda...
    # there's literally no way to detect that properly. with that in mind, we have this
    try:
        if value(100) == 1:
            return { "type": "integer" }
    except Exception:
        pass

    # this is throwing exceptions. I thought vol should handle this already
    if isinstance(value, vol.In):
        if isinstance(value.container, dict):
            return { "enum": list(value.container.keys()) }
        else:
            return { "enum": list(value.container) }
    
    if isinstance(value, list):
        result = {}
        for x in value:
            result.update(custom_custom_serializer(x))
        return result
    
    return cv.custom_serializer(value)

def download_model_from_hf(model_name: str, quantization_type: str, storage_folder: str, file_lookup_only: bool = False):
    try:
        from huggingface_hub import hf_hub_download, HfFileSystem
    except Exception as ex:
        raise Exception(f"Failed to import huggingface-hub library. Please re-install the integration.") from ex
    
    fs = HfFileSystem()
    potential_files = [ f for f in fs.glob(f"{model_name}/*.gguf") ]
    wanted_file = [f for f in potential_files if (f"{quantization_type.lower()}.gguf" in f or f"{quantization_type.upper()}.gguf" in f)]

    if len(wanted_file) != 1:
        available_quants = [
            re.split(r"\.|-", file.removesuffix(".gguf"))[-1].upper()
            for file in potential_files
        ]
        raise MissingQuantizationException(quantization_type, available_quants)
    try:
        os.makedirs(storage_folder, exist_ok=True)
    except Exception as ex:
        raise Exception(f"Failed to create the required folder for storing models! You may need to create the path '{storage_folder}' manually.") from ex

    return hf_hub_download(
        repo_id=model_name,
        repo_type="model",
        filename=wanted_file[0].removeprefix(model_name + "/"),
        cache_dir=storage_folder,
        local_files_only=file_lookup_only
    )

def _load_extension():
    """
    Makes sure it is possible to load llama-cpp-python without crashing Home Assistant.
    This needs to be at the root file level because we are using the 'spawn' start method.
    Also ignore ModuleNotFoundError because that just means it's not installed. Not that it will crash HA
    """
    import importlib
    try:
        importlib.import_module("llama_cpp")
    except ModuleNotFoundError:
        pass
    
def validate_llama_cpp_python_installation():
    """
    Spawns another process and tries to import llama.cpp to avoid crashing the main process
    """
    mp_ctx = multiprocessing.get_context('spawn') # required because of aio
    process = mp_ctx.Process(target=_load_extension)
    process.start()
    process.join()

    if process.exitcode != 0:
        raise Exception(f"Failed to properly initialize llama-cpp-python. (Exit code {process.exitcode}.)")

def get_llama_cpp_python_version():
    if not is_installed("llama-cpp-python"):
        return None
    return version("llama-cpp-python")

def get_runtime_and_platform_suffix() -> Tuple[str, str]:
    runtime_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

    platform_suffix = platform.machine()
    # remap other names for architectures to the names we use
    if platform_suffix == "arm64":
        platform_suffix = "aarch64"
    if platform_suffix == "i386" or platform_suffix == "amd64":
        platform_suffix = "x86_64"

    return runtime_version, platform_suffix

async def get_available_llama_cpp_versions(hass: HomeAssistant) -> List[Tuple[str, bool]]:
    github_index_url = "https://acon96.github.io/llama-cpp-python/whl/ha/llama-cpp-python/"
    session = aiohttp_client.async_get_clientsession(hass)
    try:
        async with session.get(github_index_url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch available versions from GitHub (HTTP {resp.status})")
            text = await resp.text()
            # pull version numbers out of h2 tags
            versions = re.findall(r"<h2.*>(.+)</h2>", text)
            remote =  sorted([(v, False) for v in versions], reverse=True)
    except Exception as ex:
        _LOGGER.warning(f"Error fetching available versions from GitHub: {repr(ex)}")
        remote = []

    runtime_version, platform_suffix = get_runtime_and_platform_suffix()
    folder = os.path.dirname(__file__)
    potential_wheels = sorted([ path for path in os.listdir(folder) if path.endswith(f"{platform_suffix}.whl") ], reverse=True)
    local = [ (wheel, True) for wheel in potential_wheels if runtime_version in wheel and "llama_cpp_python" in wheel]
    
    return remote + local

def install_llama_cpp_python(config_dir: str, force_reinstall: bool = False, specific_version: str | None = None) -> bool:

    installed_wrong_version = False
    if is_installed("llama-cpp-python") and not force_reinstall:
        if version("llama-cpp-python") != EMBEDDED_LLAMA_CPP_PYTHON_VERSION:
            installed_wrong_version = True
        else:
            time.sleep(0.5) # I still don't know why this is required
            return True
        
    runtime_version, platform_suffix = get_runtime_and_platform_suffix()

    if not specific_version:
        specific_version = EMBEDDED_LLAMA_CPP_PYTHON_VERSION
    
    if ".whl" in specific_version:
        wheel_location = os.path.join(os.path.dirname(__file__), specific_version)
    else:
        wheel_location = f"https://github.com/acon96/llama-cpp-python/releases/download/{specific_version}/llama_cpp_python-{specific_version}-{runtime_version}-{runtime_version}-linux_{platform_suffix}.whl"

    if install_package(wheel_location, **pip_kwargs(config_dir)):
        _LOGGER.info("llama-cpp-python successfully installed")
        return True
    
    # if it is just the wrong version installed then ignore the installation error
    if not installed_wrong_version:
        _LOGGER.error(
            "Error installing llama-cpp-python. Could not install the binary wheels from GitHub." + \
            "Please manually build or download the wheels and place them in the `/config/custom_components/llama_conversation` directory." + \
            "Make sure that you download the correct .whl file for your platform and python version from the GitHub releases page."
        )
        return False
    else:
        _LOGGER.info(
            "Error installing llama-cpp-python. Could not install the binary wheels from GitHub." + \
            f"You already have a version of llama-cpp-python ({version('llama-cpp-python')}) installed, however it may not be compatible!"
        )
        time.sleep(0.5) # I still don't know why this is required

        return True

def format_url(*, hostname: str, port: str, ssl: bool, path: str):
    return f"{'https' if ssl else 'http'}://{hostname}{ ':' + port if port else ''}{path}"

def get_oai_formatted_tools(llm_api: llm.APIInstance, domains: list[str]) -> List[ChatCompletionTool]:
    if llm_api.api.id == HOME_LLM_API_ID:
        result: List[ChatCompletionTool] = [ {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": f"Call the Home Assistant service '{tool['name']}'",
                "parameters": convert(tool["arguments"], custom_serializer=llm_api.custom_serializer)
            }
        } for tool in get_home_llm_tools(llm_api, domains) ]
    
    else:
        result: List[ChatCompletionTool] = [ {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": convert(tool.parameters, custom_serializer=llm_api.custom_serializer)
            }
        } for tool in llm_api.tools ]

    return result

def get_oai_formatted_messages(conversation: Sequence[conversation.Content], user_content_as_list: bool = False, tool_args_to_str: bool = True) -> List[ChatCompletionRequestMessage]:
        messages: List[ChatCompletionRequestMessage] = []
        for message in conversation:
            if message.role == "system":
                messages.append({
                    "role": "system",
                    "content": message.content
                })
            elif message.role == "user":
                if user_content_as_list:
                    messages.append({
                        "role": "user",
                        "content": [{ "type": "text", "text": message.content }]
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": message.content
                    })
            elif message.role == "assistant":
                if message.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": str(message.content),
                        "tool_calls": [
                            {
                                "type" : "function",
                                "id": t.id,
                                "function": {
                                    "arguments": cast(str, json.dumps(t.tool_args) if tool_args_to_str else t.tool_args),
                                    "name": t.tool_name,
                                }
                            } for t in message.tool_calls
                        ]
                    })
            elif message.role == "tool_result":
                messages.append({
                    "role": "tool",
                    "content": json.dumps(message.tool_result),
                    "tool_call_id": message.tool_call_id
                })

        return messages

def get_home_llm_tools(llm_api: llm.APIInstance, domains: list[str]) -> List[Dict[str, Any]]:
    service_dict = llm_api.api.hass.services.async_services()
    all_services = []
    scripts_added = False
    for domain in domains:
        if domain not in SERVICE_TOOL_ALLOWED_DOMAINS:
            continue

        # scripts show up as individual services
        if domain == "script" and not scripts_added:
            all_services.extend([
                ("script.reload", vol.Schema({vol.Required("target_device"): str})),
                ("script.turn_on", vol.Schema({vol.Required("target_device"): str})),
                ("script.turn_off", vol.Schema({vol.Required("target_device"): str})),
                ("script.toggle", vol.Schema({vol.Required("target_device"): str})),
            ])
            scripts_added = True
            continue

        for name, service in service_dict.get(domain, {}).items():
            if name not in SERVICE_TOOL_ALLOWED_SERVICES:
                continue

            args = flatten_vol_schema(service.schema)
            args_to_expose = set(args).intersection(ALLOWED_SERVICE_CALL_ARGUMENTS)
            service_schema = vol.Schema({
                vol.Required("target_device"): str,
                **{vol.Optional(arg): str for arg in args_to_expose}
            })

            all_services.append((f"{domain}.{name}", service_schema))

    tools: List[Dict[str, Any]] = [
        { "name": service[0], "arguments": service[1] } for service in all_services
    ]

    return tools

def parse_raw_tool_call(raw_block: str | dict, llm_api: llm.APIInstance, user_input: conversation.ConversationInput) -> tuple[llm.ToolInput | None, str | None]:
    if isinstance(raw_block, dict):
        parsed_tool_call = raw_block
    else:
        parsed_tool_call: dict = json.loads(raw_block)

    if llm_api.api.id == HOME_LLM_API_ID:
        schema_to_validate = vol.Schema({
            vol.Required('service'): str,
            vol.Required('target_device'): str,
            vol.Optional('rgb_color'): str,
            vol.Optional('brightness'): vol.Coerce(float),
            vol.Optional('temperature'): vol.Coerce(float),
            vol.Optional('humidity'): vol.Coerce(float),
            vol.Optional('fan_mode'): str,
            vol.Optional('hvac_mode'): str,
            vol.Optional('preset_mode'): str,
            vol.Optional('duration'): str,
            vol.Optional('item'): str,
        })
    else:
        schema_to_validate = vol.Schema({
            vol.Required("name"): str,
            vol.Required("arguments"): vol.Union(str, dict),
        })

    try:
        schema_to_validate(parsed_tool_call)
    except vol.Error as ex:
        _LOGGER.info(f"LLM produced an improperly formatted response: {repr(ex)}")
        raise MalformedToolCallException(user_input.agent_id, "", "unknown", str(raw_block), "Tool call was not properly formatted")

    # try to fix certain arguments
    args_dict = parsed_tool_call if llm_api.api.id == HOME_LLM_API_ID else parsed_tool_call["arguments"]
    tool_name = parsed_tool_call.get("name", parsed_tool_call.get("service", ""))

    if isinstance(args_dict, str):
        if not args_dict.strip():
            args_dict = {} # don't attempt to parse empty arguments
        else:
            try:
                args_dict = json.loads(args_dict)
            except json.JSONDecodeError:
                raise MalformedToolCallException(user_input.agent_id, "", tool_name, str(args_dict), "Tool arguments were not properly formatted JSON")

    # make sure brightness is 0-255 and not a percentage
    if "brightness" in args_dict and 0.0 < args_dict["brightness"] <= 1.0:
        args_dict["brightness"] = int(args_dict["brightness"] * 255)

    # convert string "tuple" to a list for RGB colors
    if "rgb_color" in args_dict and isinstance(args_dict["rgb_color"], str):
        args_dict["rgb_color"] = [ int(x) for x in args_dict["rgb_color"][1:-1].split(",") ]

    to_say = args_dict.pop("to_say", "")
    tool_input = llm.ToolInput(
        tool_name=tool_name,
        tool_args=args_dict,
    )

    return tool_input, to_say

def is_valid_hostname(host: str) -> bool:
    """
    Validates whether a string is a valid hostname or IP address,
    rejecting URLs, paths, ports, query strings, etc.
    """
    if not host or not isinstance(host, str):
        return False

    # Normalize: strip whitespace
    host = host.strip().lower()

    # Special case: localhost
    if host == "localhost":
        return True

    # Try to parse as IPv4
    try:
        ipaddress.IPv4Address(host)
        return True
    except ipaddress.AddressValueError:
        pass

    # Try to parse as IPv6
    try:
        ipaddress.IPv6Address(host)
        return True
    except ipaddress.AddressValueError:
        pass

    # Validate as domain name (RFC 1034/1123)
    # Rules:
    # - Only a-z, 0-9, hyphens
    # - No leading/trailing hyphens
    # - Max 63 chars per label
    # - At least 2 chars in TLD
    # - No consecutive dots

    domain_pattern = re.compile(r"^[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?)*\.[a-z]{2,}$")

    return bool(domain_pattern.match(host))