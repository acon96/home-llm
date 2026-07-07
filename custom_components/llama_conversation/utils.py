from functools import partial
import time
import logging
import os
import re
import sys
import platform
import multiprocessing
import site
import voluptuous as vol
import base64
import fuzzy_json
from subprocess import PIPE, Popen
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast
from importlib.metadata import version

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.components import conversation
from homeassistant.helpers import config_validation as cv, json as json_helper
from homeassistant.helpers import intent, llm, aiohttp_client
from homeassistant.requirements import pip_kwargs
from homeassistant.util import color, package as package_util, json as ha_json
from homeassistant.util.package import is_installed


from voluptuous_openapi import convert as convert_to_openapi

from .const import (
    DOMAIN,
    EMBEDDED_LLAMA_CPP_PYTHON_VERSION,
    ALLOWED_SERVICE_CALL_ARGUMENTS,
    SERVICE_TOOL_ALLOWED_SERVICES,
    SERVICE_TOOL_ALLOWED_DOMAINS,
    SERVICE_TOOL_NAME,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llama_cpp.llama_types import ChatCompletionRequestMessage, ChatCompletionTool
else:
    ChatCompletionRequestMessage = Any
    ChatCompletionTool = Any

_LOGGER = logging.getLogger(__name__)


class MissingQuantizationException(Exception):
    def __init__(self, missing_quant: str, available_quants: list[str]):
        super().__init__(missing_quant, available_quants)
        self.missing_quant = missing_quant
        self.available_quants = available_quants


class LlamaCppPythonInstallError(HomeAssistantError):
    """Raised when llama-cpp-python cannot be installed from the hosted wheels."""

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

def closest_color(requested_color: tuple[int, int, int]) -> str:
    """Find the closest named color to an RGB tuple using HA's built-in color map."""
    r_c, g_c, b_c = requested_color
    min_dist = float("inf")
    closest_name = ""
    for name, rgb in color.COLORS.items():
        rd = (rgb.r - r_c) ** 2
        gd = (rgb.g - g_c) ** 2
        bd = (rgb.b - b_c) ** 2
        dist = rd + gd + bd
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

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

def get_platform_suffix() -> str:
    """Get the platform suffix for wheel files."""
    platform_suffix = platform.machine()
    # remap other names for architectures to the names we use
    if platform_suffix == "arm64":
        platform_suffix = "aarch64"
    if platform_suffix == "i386" or platform_suffix == "amd64":
        platform_suffix = "x86_64"

    return platform_suffix

def get_potential_wheels(folder: str, platform_suffix: str) -> List[str]:
    return sorted([ path for path in os.listdir(folder) if path.endswith(f"{platform_suffix}.whl") ], reverse=True)

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

    platform_suffix = get_platform_suffix()
    folder = os.path.dirname(__file__)
    potential_wheels = await hass.async_add_executor_job(partial(get_potential_wheels, folder, platform_suffix))
    local = [ (wheel, True) for wheel in potential_wheels if "llama_cpp_python" in wheel and ("py3-none" in wheel or f"cp{sys.version_info.major}{sys.version_info.minor}" in wheel)]
    return remote + local


def _install_package_with_stderr(
    package: str,
    *,
    upgrade: bool = True,
    target: str | None = None,
    constraints: str | None = None,
    timeout: int | None = None,
    reinstall: bool = False,
) -> tuple[bool, str | None]:
    env = os.environ.copy()
    args = [
        sys.executable,
        "-m",
        "uv",
        "pip",
        "install",
        "--quiet",
        package,
        "--index-strategy",
        "unsafe-first-match",
    ]

    if timeout:
        env["HTTP_TIMEOUT"] = str(timeout)
    if upgrade:
        args.append("--upgrade")
    if reinstall:
        args.append("--reinstall")
    if constraints is not None:
        args += ["--constraint", constraints]
    if target:
        args += ["--target", os.path.abspath(target)]
    elif (
        not package_util.is_virtual_env()
        and not any(var in env for var in package_util._UV_ENV_PYTHON_VARS)
        and (abs_target := site.getusersitepackages())
    ):
        args += ["--python", sys.executable, "--target", abs_target]

    with Popen(
        args,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        env=env,
        close_fds=False,
    ) as process:
        _, stderr = process.communicate()
        stderr_text = stderr.decode("utf-8").lstrip().strip() or None
        return process.returncode == 0, stderr_text


def install_llama_cpp_python(
    config_dir: str,
    force_reinstall: bool = False,
    specific_version: str | None = None,
    raise_on_error: bool = False,
) -> bool:

    installed_wrong_version = False
    if is_installed("llama-cpp-python") and not force_reinstall:
        if version("llama-cpp-python") != EMBEDDED_LLAMA_CPP_PYTHON_VERSION:
            installed_wrong_version = True
        else:
            time.sleep(0.5) # I still don't know why this is required
            return True
    
    if force_reinstall:
        _LOGGER.info("Force reinstalling llama-cpp-python")
        
    platform_suffix = get_platform_suffix()

    if not specific_version:
        specific_version = EMBEDDED_LLAMA_CPP_PYTHON_VERSION
    
    if ".whl" in specific_version:
        wheel_location = os.path.join(os.path.dirname(__file__), specific_version)
    else:
        wheel_location = f"https://github.com/acon96/llama-cpp-python/releases/download/{specific_version}/llama_cpp_python-{specific_version}-py3-none-linux_{platform_suffix}.whl"

    install_success, install_error = _install_package_with_stderr(
        wheel_location, reinstall=force_reinstall, **pip_kwargs(config_dir)
    )

    if install_success:
        _LOGGER.info("llama-cpp-python successfully installed")
        return True
    
    # if it is just the wrong version installed then ignore the installation error
    if not installed_wrong_version:
        error_message = (
            f"Unable to install package {wheel_location}: {install_error or 'unknown installation error'}. "
            "Please manually build or download the wheels and place them in the `/config/custom_components/llama_conversation` directory. "
            "Make sure that you download the correct .whl file for your platform from the GitHub releases page."
        )
        if raise_on_error:
            raise LlamaCppPythonInstallError(error_message)

        _LOGGER.error("Error installing llama-cpp-python. %s", error_message)
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
    result: List[ChatCompletionTool] = []

    # sort tools by name to improve cache hits
    for tool in sorted(llm_api.tools, key=lambda t: t.name):
        # when combining with home assistant llm APIs, it adds a prefix to differentiate tools; compare against the suffix here
        if tool.name.endswith(SERVICE_TOOL_NAME):
            result.extend([{
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": f"Call the Home Assistant service '{tool['name']}'",
                    "parameters": convert_to_openapi(tool["arguments"], custom_serializer=llm_api.custom_serializer),
                    "strict": True,
                }
            } for tool in get_home_llm_tools(llm_api, domains) ])
        else:
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": convert_to_openapi(tool.parameters, custom_serializer=llm_api.custom_serializer),
                    "strict": True,
                }
            })


    return result

def get_oai_formatted_messages(
        conversation: Sequence[conversation.Content],
        *,
        user_content_as_list: bool = False,
        tool_args_to_str: bool = True,
        tool_result_to_str: bool = True,
    ) -> List[ChatCompletionRequestMessage]:
    messages: List[ChatCompletionRequestMessage] = []
    for message in conversation:
        if message.role == "system":
            messages.append({
                "role": "system",
                "content": message.content
            })
        elif message.role == "user":
            images: list[str] = []
            for attachment in message.attachments or ():
                if not attachment.mime_type.startswith("image/"):
                    raise HomeAssistantError(
                        translation_domain=DOMAIN,
                        translation_key="unsupported_attachment_type",
                    )
                images.append(get_file_contents_base64(attachment.path))

            if user_content_as_list:
                content = [{ "type": "text", "text": message.content }]
                for image in images:
                    content.append({ "type": "image_url", "image_url": {"url": image } })

                messages.append({
                    "role": "user",
                    "content": content
                })
            else:
                message = {
                    "role": "user",
                    "content": message.content
                }
                if images:
                    message["images"] = images
                messages.append(message)
        elif message.role == "assistant":
            if message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": str(message.content) if message.content else "",
                    "tool_calls": [
                        {
                            "type" : "function",
                            "id": t.id,
                            "function": {
                                "arguments": cast(str, json_helper.json_dumps(t.tool_args) if tool_args_to_str else t.tool_args),
                                "name": t.tool_name,
                            }
                        } for t in message.tool_calls
                    ]
                })
        elif message.role == "tool_result":
            if tool_result_to_str:
                content = json_helper.json_dumps(message.tool_result)
            else:
                content = [{
                    "name": message.tool_name,
                    "response": { "result": message.tool_result },
                    }
                ]
            messages.append({
                "role": "tool",
                "name": message.tool_name, # functiongemma compat https://huggingface.co/google/functiongemma-270m-it/blob/main/chat_template.jinja#L232
                "content": content,
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

def parse_raw_tool_call(raw_block: str | dict, agent_id: str) -> tuple[llm.ToolInput | None, str | None]:
    if isinstance(raw_block, dict):
        parsed_tool_call = raw_block
    else:
        try:
            parsed_tool_call: dict = parse_json_with_repair_fallback(raw_block)
        except ha_json.JSON_DECODE_EXCEPTIONS:
            # handle the "gemma" tool calling format
            # call:HassTurnOn{name:<escape>light.living_room_rgbww<escape>}
            gemma_match = re.finditer(r"call:(?P<name>\w+){(?P<args>.+)}", raw_block)
            for match in gemma_match:
                tool_name = match.group("name")
                raw_args = match.group("args")
                args_dict = {}
                for arg_match in re.finditer(r"(?P<key>\w+):<escape>(?P<value>.+?)<escape>", raw_args):
                    args_dict[arg_match.group("key")] = arg_match.group("value")
                
                parsed_tool_call = {
                    "name": tool_name,
                    "arguments": args_dict
                }
                break # TODO: how do we properly handle multiple tool calls in one response?
            else:
                raise MalformedToolCallException(agent_id, "", "unknown", str(raw_block), "Tool call was not properly formatted JSON")

    # try to validate either format
    is_services_tool_call = False
    try:
        base_schema_to_validate = vol.Schema({
            vol.Required("name"): str,
            vol.Required("arguments"): vol.Union(str, dict),
        })
        base_schema_to_validate(parsed_tool_call)
    except vol.Error as ex:
        try:
            home_llm_schema_to_validate = vol.Schema({
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
            home_llm_schema_to_validate(parsed_tool_call)
            is_services_tool_call = True
        except vol.Error as ex:
            _LOGGER.info(f"LLM produced an improperly formatted response: {repr(ex)}")
            raise MalformedToolCallException(agent_id, "", "unknown", str(raw_block), "Tool call was not properly formatted")

    # try to fix certain arguments
    args_dict = parsed_tool_call if is_services_tool_call else parsed_tool_call["arguments"]
    tool_name = SERVICE_TOOL_NAME if is_services_tool_call else parsed_tool_call["name"]

    if isinstance(args_dict, str):
        if not args_dict.strip():
            args_dict = {} # don't attempt to parse empty arguments
        else:
            args_dict = parse_tool_arguments_with_repair_fallback(args_dict, agent_id, tool_name)

    # check if it is actually a service call from a newer LLM
    tool_name_split = tool_name.split(".")
    if len(tool_name_split) == 2 and tool_name_split[1] in SERVICE_TOOL_ALLOWED_SERVICES and tool_name_split[0] in SERVICE_TOOL_ALLOWED_DOMAINS:
        args_dict["service"] = tool_name
        tool_name = SERVICE_TOOL_NAME

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



def get_file_contents_base64(file_path: Path) -> str:
    """Reads a file and returns its contents encoded in base64."""
    with open(file_path, "rb") as f:
        encoded_bytes = base64.b64encode(f.read())
        encoded_str = encoded_bytes.decode('utf-8')
    
    return encoded_str

def parse_json_with_repair_fallback(raw_str: str) -> Any:
    """Tries to parse a string as JSON, and if it fails, attempts to repair common issues and parse again."""
    try:
        return ha_json.json_loads(raw_str)
    except ha_json.JSON_DECODE_EXCEPTIONS as first_ex:
        try:
            return fuzzy_json.loads(raw_str)
        except Exception:
            raise first_ex


def parse_tool_arguments_with_repair_fallback(raw_str: str, agent_id: str, tool_name: str) -> Dict[str, Any]:
    """Parse tool arguments as a JSON object, repairing common syntax issues first."""
    try:
        parsed_args = parse_json_with_repair_fallback(raw_str)
    except ha_json.JSON_DECODE_EXCEPTIONS as first_ex:
        raise MalformedToolCallException(
            agent_id,
            "",
            tool_name,
            str(raw_str),
            "Tool arguments were not properly formatted JSON",
        ) from first_ex

    if not isinstance(parsed_args, dict):
        raise MalformedToolCallException(
            agent_id,
            "",
            tool_name,
            str(raw_str),
            "Tool arguments were not properly formatted JSON",
        )

    return parsed_args


def strip_thinking_blocks(content: str, think_prefix: str, think_suffix: str) -> str:
    """Remove all thinking blocks from a response string.

    If a thinking block starts but never closes, everything after the opening
    prefix is removed to avoid leaking hidden reasoning into speech output.
    """
    if not content or not think_prefix or not think_suffix:
        return content

    cleaned_parts: list[str] = []
    cursor = 0
    content_len = len(content)

    while cursor < content_len:
        block_start = content.find(think_prefix, cursor)
        if block_start == -1:
            cleaned_parts.append(content[cursor:])
            break

        cleaned_parts.append(content[cursor:block_start])

        block_end = content.find(think_suffix, block_start + len(think_prefix))
        if block_end == -1:
            break

        cursor = block_end + len(think_suffix)

    return "".join(cleaned_parts).strip()
    