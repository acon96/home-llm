# Home LLM
This project provides the required "glue" components to control your Home Assistant installation with a **completely local** Large Language Model acting as a personal assistant. The goal is to provide a drop in solution to be used as a "conversation agent" component by Home Assistant.  The 2 main pieces of this solution are the Home LLM model and Local LLM Conversation integration.

## Quick Start
Please see the [Setup Guide](./docs/Setup.md) for more information on installation.

## Local LLM Integration
**The latest version of this integration requires Home Assistant 2025.7.0 or newer**

In order to integrate with Home Assistant, we provide a custom component that exposes the locally running LLM as a "conversation agent" or as an "ai task handler".

This component can be interacted with in a few ways:  
- using a chat interface so you can chat with it.
- integrating with Speech-to-Text and Text-to-Speech addons so you can just speak to it.
- using automations or scripts to trigger "ai tasks"; these process input data with a prompt, and return structured data that can be used in further automations.

The integration can either run the model in a few ways:
1. Directly as part of the Home Assistant software using llama-cpp-python
2. On a separate machine using one of the following backends:
    - [Ollama](https://ollama.com/) (easier)
    - [LocalAI](https://localai.io/) via the Generic OpenAI backend (easier)
    - [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) project (advanced)
    - [llama.cpp example server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#readme) (advanced)

## Home LLM Model
The "Home" models are a fine tuning of various Large Languages Models that are under 5B parameters.  The models are able to control devices in the user's house as well as perform basic question and answering.  The fine tuning dataset is a [custom synthetic dataset](./data) designed to teach the model function calling based on the device information in the context.

The latest models can be found on HuggingFace:

**Llama 3.2**:  
3B: https://huggingface.co/acon96/Home-Llama-3.2-3B  
1B: TBD  

**Qwen 3**:  
0.6B: TBD  
1.5B: TBD  

**Gemma3**:  
1B: TBD
270M: https://huggingface.co/acon96/Home-FunctionGemma-270m

<details>

<summary>Old Models</summary>  

3B v3 (Based on StableLM-Zephyr-3B): https://huggingface.co/acon96/Home-3B-v3-GGUF  (Zephyr prompt format)  
1B v3 (Based on TinyLlama-1.1B): https://huggingface.co/acon96/Home-1B-v3-GGUF  (Zephyr prompt format)  
3B v2 (Based on Phi-2): https://huggingface.co/acon96/Home-3B-v2-GGUF  (ChatML prompt format)  
1B v2 (Based on Phi-1.5): https://huggingface.co/acon96/Home-1B-v2-GGUF  (ChatML prompt format)  
1B v1 (Based on Phi-1.5): https://huggingface.co/acon96/Home-1B-v1-GGUF  (ChatML prompt format)  

Non English experiments:  
German, French, & Spanish (3B): https://huggingface.co/acon96/stablehome-multilingual-experimental  
Polish (1B): https://huggingface.co/acon96/tinyhome-polish-experimental  

NOTE: The models below are only compatible with version 0.2.17 and older!  
3B v1 (Based on Phi-2): https://huggingface.co/acon96/Home-3B-v1-GGUF  (ChatML prompt format)  


</details>

The model is quantized using Llama.cpp in order to enable running the model in super low resource environments that are common with Home Assistant installations such as Raspberry Pis.

### Synthetic Dataset
The synthetic dataset is aimed at covering basic day to day operations in home assistant such as turning devices on and off.
The supported entity types are: light, fan, cover, lock, media_player, climate, switch

The dataset is available on HuggingFace: https://huggingface.co/datasets/acon96/Home-Assistant-Requests-V2  
The source for the dataset is in the [data](/data) of this repository.

### Training

If you want to fine-tune a model yourself, see the details on how to do it in the [Training README](./train/README.md).

## Version History
| Version | Description                                                                                                                                                                                                                                           |
|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v0.4.6  | New dataset supporting proper tool calling, Add Anthropic "messages" style API support                                                                                                                                                                |
| v0.4.5  | Add support for AI Task entities, Replace custom Ollama API implementation with the official `ollama-python` package to avoid future compatibility issues, Support multiple LLM APIs at once, Fix issues in tool call handling for various backends   | 
| v0.4.4  | Fix issue with OpenAI backends appending `/v1` to all URLs, and fix an issue with tools being serialized into the system prompt.                                                                                                                      |
| v0.4.3  | Fix an issue with the integration not creating model configs properly during setup                                                                                                                                                                    |
| v0.4.2  | Fix the following issues: not correctly setting default model settings during initial setup, non-integers being allowed in numeric config fields, being too strict with finish_reason requirements, and not letting the user clear the active LLM API |
| v0.4.1  | Fix an issue with using Llama.cpp models downloaded from HuggingFace                                                                                                                                                                                  |
| v0.4    | Rewrite integration to support tool calling models/agentic tool use loop, voice streaming, multiple config sub-entries per backend, and dynamic llama.cpp processor selection                                                                         |
| v0.3.11 | Bug-fixes and llama.cpp version update                                                                                                                                                                                                                |
| v0.3.10 | Add support for the OpenAI "Responses" API endpoint, Update llama.cpp version, Fix for breaking change in HA version 2025.7.0                                                                                                                         |
| v0.3.9  | Update llama.cpp version, fix installation bugs, fix conversation history not working                                                                                                                                                                 |
| v0.3.8  | Update llama.cpp, remove think blocks from "thinking" models, fix wheel detection for some Intel CPUs, Fixes for compatibility with latest Home Assistant version (2025.4), other small bug fixes                                                     |
| v0.3.7  | Update llama.cpp version to support newer models, Update minimum Home Assistant version to 2024.12.3, Add German In-Context Learning examples, Fix multi-turn use, Fix an issue with webcolors                                                        | 
| v0.3.6  | Small llama.cpp backend fixes                                                                                                                                                                                                                         |
| v0.3.5  | Fix for llama.cpp backend installation, Fix for Home LLM v1-3 API parameters, add Polish ICL examples                                                                                                                                                 |
| v0.3.4  | Significantly improved language support including full Polish translation, Update bundled llama-cpp-python to support new models, various bug fixes                                                                                                   |
| v0.3.3  | Improvements to the Generic OpenAI Backend, improved area handling, fix issue using RGB colors, remove EOS token from responses, replace requests dependency with aiohttp included with Home Assistant                                                |
| v0.3.2  | Fix for exposed script entities causing errors, fix missing GBNF error, trim whitespace from model output                                                                                                                                             |
| v0.3.1  | Adds basic area support in prompting, Fix for broken requirements, fix for issue with formatted tools, fix custom API not registering on startup properly                                                                                             |
| v0.3    | Adds support for Home Assistant LLM APIs, improved model prompting and tool formatting options, and automatic detection of GGUF quantization levels on HuggingFace                                                                                    |
| v0.2.17 | Disable native llama.cpp wheel optimizations, add Command R prompt format                                                                                                                                                                             |
| v0.2.16 | Fix for missing huggingface_hub package preventing startup                                                                                                                                                                                            |
| v0.2.15 | Fix startup error when using llama.cpp backend and add flash attention to llama.cpp backend                                                                                                                                                           |
| v0.2.14 | Fix llama.cpp wheels + AVX detection                                                                                                                                                                                                                  |
| v0.2.13 | Add support for Llama 3, build llama.cpp wheels that are compatible with non-AVX systems, fix an error with exposing script entities, fix multiple small Ollama backend issues, and add basic multi-language support                                  |
| v0.2.12 | Fix cover ICL examples, allow setting number of ICL examples, add min P and typical P sampler options, recommend models during setup, add JSON mode for Ollama backend, fix missing default options                                                   |
| v0.2.11 | Add prompt caching, expose llama.cpp runtime settings, build llama-cpp-python wheels using GitHub actions, and install wheels directly from GitHub                                                                                                    |
| v0.2.10 | Allow configuring the model parameters during initial setup, attempt to auto-detect defaults for recommended models, Fix to allow lights to be set to max brightness                                                                                  |
| v0.2.9  | Fix HuggingFace Download, Fix llama.cpp wheel installation, Fix light color changing, Add in-context-learning support                                                                                                                                 |
| v0.2.8  | Fix ollama model names with colons                                                                                                                                                                                                                    |
| v0.2.7  | Publish model v3, Multiple Ollama backend improvements, Updates for HA 2024.02, support for voice assistant aliases                                                                                                                                   |
| v0.2.6  | Bug fixes, add options for limiting chat history, HTTPS endpoint support, added zephyr prompt format.                                                                                                                                                 |
| v0.2.5  | Fix Ollama max tokens parameter, fix GGUF download from Hugging Face, update included llama-cpp-python to 0.2.32, and add parameters to function calling for dataset + component, & model update                                                      |
| v0.2.4  | Fix API key auth on model load for text-generation-webui, and add support for Ollama API backend                                                                                                                                                      |
| v0.2.3  | Fix API key auth, Support chat completion endpoint, and refactor to make it easier to add more remote backends                                                                                                                                        |
| v0.2.2  | Fix options window after upgrade, fix training script for new Phi model format, and release new models                                                                                                                                                |
| v0.2.1  | Properly expose generation parameters for each backend, handle config entry updates without reloading, support remote backends with an API key                                                                                                        |
| v0.2    | Bug fixes, support more backends, support for climate + switch devices, JSON style function calling with parameters, GBNF grammars                                                                                                                    |
| v0.1    | Initial Release                                                                                                                                                                                                                                       |
