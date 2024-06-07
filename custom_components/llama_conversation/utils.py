import time
import os
import sys
import platform
import logging
import multiprocessing
import voluptuous as vol
import webcolors
from importlib.metadata import version

from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import intent
from homeassistant.requirements import pip_kwargs
from homeassistant.util import color
from homeassistant.util.package import install_package, is_installed

from .const import (
    INTEGRATION_VERSION,
    EMBEDDED_LLAMA_CPP_PYTHON_VERSION,
)

_LOGGER = logging.getLogger(__name__)

class MissingQuantizationException(Exception):
    def __init__(self, missing_quant: str, available_quants: list[str]):
        self.missing_quant = missing_quant
        self.available_quants = available_quants

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
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
    
    if isinstance(value, list):
        result = {}
        for x in value:
            result.update(custom_custom_serializer(x))
        return result
    
    return cv.custom_serializer(value)

def download_model_from_hf(model_name: str, quantization_type: str, storage_folder: str):
    try:
        from huggingface_hub import hf_hub_download, HfFileSystem
    except Exception as ex:
        raise Exception(f"Failed to import huggingface-hub library. Please re-install the integration.") from ex
    
    fs = HfFileSystem()
    potential_files = [ f for f in fs.glob(f"{model_name}/*.gguf") ]
    wanted_file = [f for f in potential_files if (f".{quantization_type.lower()}." in f or f".{quantization_type.upper()}." in f)]

    if len(wanted_file) != 1:
        available_quants = [file.split(".")[-2].upper() for file in potential_files]
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

def install_llama_cpp_python(config_dir: str):

    installed_wrong_version = False
    if is_installed("llama-cpp-python"):
        if version("llama-cpp-python") != EMBEDDED_LLAMA_CPP_PYTHON_VERSION:
            installed_wrong_version = True
        else:
            time.sleep(0.5) # I still don't know why this is required
            return True
    
    platform_suffix = platform.machine()
    if platform_suffix == "arm64":
        platform_suffix = "aarch64"

    runtime_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

    instruction_extensions_suffix = ""
    if platform_suffix == "amd64" or platform_suffix == "i386":
        instruction_extensions_suffix = "-noavx"

        try:
            with open("/proc/cpuinfo") as f:
                cpu_features = [ line for line in f.readlines() if line.startswith("Features") or line.startswith("flags")][0]

            _LOGGER.debug(cpu_features)
            if " avx512f " in cpu_features and " avx512bw " in cpu_features:
                instruction_extensions_suffix = "-avx512"
            elif " avx2 " in cpu_features and \
                 " avx " in cpu_features and \
                 " f16c " in cpu_features and \
                 " fma " in cpu_features and \
                 (" sse3 " in cpu_features or " ssse3 " in cpu_features):
                instruction_extensions_suffix = ""
        except Exception as ex:
            _LOGGER.debug(f"Couldn't detect CPU features: {ex}")
            # default to the noavx build to avoid crashing home assistant
            instruction_extensions_suffix = "-noavx"
    
    folder = os.path.dirname(__file__)
    potential_wheels = sorted([ path for path in os.listdir(folder) if path.endswith(f"{platform_suffix}{instruction_extensions_suffix}.whl") ], reverse=True)
    potential_wheels = [ wheel for wheel in potential_wheels if runtime_version in wheel ]
    if len(potential_wheels) > 0:

        latest_wheel = potential_wheels[0]

        _LOGGER.info("Installing llama-cpp-python from local wheel")
        _LOGGER.debug(f"Wheel location: {latest_wheel}")
        return install_package(os.path.join(folder, latest_wheel), pip_kwargs(config_dir))
        
    github_release_url = f"https://github.com/acon96/home-llm/releases/download/v{INTEGRATION_VERSION}/llama_cpp_python-{EMBEDDED_LLAMA_CPP_PYTHON_VERSION}-{runtime_version}-{runtime_version}-musllinux_1_2_{platform_suffix}{instruction_extensions_suffix}.whl"
    if install_package(github_release_url, pip_kwargs(config_dir)):
        _LOGGER.info("llama-cpp-python successfully installed from GitHub release")
        return True
    
    # if it is just the wrong version installed then ignore the installation error
    if not installed_wrong_version:
        _LOGGER.error(
            "Error installing llama-cpp-python. Could not install the binary wheels from GitHub for " + \
            f"platform: {platform_suffix}{instruction_extensions_suffix}, python version: {sys.version_info.major}.{sys.version_info.minor}. " + \
            "Please manually build or download the wheels and place them in the `/config/custom_components/llama_conversation` directory." + \
            "Make sure that you download the correct .whl file for your platform and python version from the GitHub releases page."
        )
        return False
    else:
        _LOGGER.info(
            "Error installing llama-cpp-python. Could not install the binary wheels from GitHub for " + \
            f"platform: {platform_suffix}{instruction_extensions_suffix}, python version: {sys.version_info.major}.{sys.version_info.minor}. " + \
            f"You already have a version of llama-cpp-python ({version('llama-cpp-python')}) installed, however it may not be compatible!"
        )
        time.sleep(0.5) # I still don't know why this is required

        return True

def format_url(*, hostname: str, port: str, ssl: bool, path: str):
    return f"{'https' if ssl else 'http'}://{hostname}{ ':' + port if port else ''}{path}"