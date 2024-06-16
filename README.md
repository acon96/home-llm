# Home LLM
This project provides the required "glue" components to control your Home Assistant installation with a **completely local** Large Language Model acting as a personal assistant. The goal is to provide a drop in solution to be used as a "conversation agent" component by Home Assistant.  The 2 main pieces of this solution are the Home LLM model and Local LLM Conversation integration.

## Quick Start
Please see the [Setup Guide](./docs/Setup.md) for more information on installation.

## Local LLM Conversation Integration
In order to integrate with Home Assistant, we provide a custom component that exposes the locally running LLM as a "conversation agent".

This component can be interacted with in a few ways:  
- using a chat interface so you can chat with it.
- integrating with Speech-to-Text and Text-to-Speech addons so you can just speak to it.

The integration can either run the model in 2 different ways:
1. Directly as part of the Home Assistant software using llama-cpp-python
2. On a separate machine using one of the following backends:
    - [Ollama](https://ollama.com/) (easier)
    - [LocalAI](https://localai.io/) via the Generic OpenAI backend (easier)
    - [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) project (advanced)
    - [llama.cpp example server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md) (advanced)

## Home LLM Model
The "Home" models are a fine tuning of various Large Languages Models that are under 5B parameters.  The models are able to control devices in the user's house as well as perform basic question and answering.  The fine tuning dataset is a [custom synthetic dataset](./data) designed to teach the model function calling based on the device information in the context.

The latest models can be found on HuggingFace:  
3B v3 (Based on StableLM-Zephyr-3B): https://huggingface.co/acon96/Home-3B-v3-GGUF  (Zephyr prompt format)  
1B v3 (Based on TinyLlama-1.1B): https://huggingface.co/acon96/Home-1B-v3-GGUF  (Zephyr prompt format)  

Non English experiments:  
German, French, & Spanish (3B): https://huggingface.co/acon96/stablehome-multilingual-experimental  
Polish (1B): https://huggingface.co/acon96/tinyhome-polish-experimental  

<details>

<summary>Old Models</summary>  

3B v2 (Based on Phi-2): https://huggingface.co/acon96/Home-3B-v2-GGUF  (ChatML prompt format)  
1B v2 (Based on Phi-1.5): https://huggingface.co/acon96/Home-1B-v2-GGUF  (ChatML prompt format)  
1B v1 (Based on Phi-1.5): https://huggingface.co/acon96/Home-1B-v1-GGUF  (ChatML prompt format)  

NOTE: The models below are only compatible with version 0.2.17 and older!
3B v1 (Based on Phi-2): https://huggingface.co/acon96/Home-3B-v1-GGUF  (ChatML prompt format)  

</details>

The model is quantized using Llama.cpp in order to enable running the model in super low resource environments that are common with Home Assistant installations such as Raspberry Pis.

The model can be used as an "instruct" type model using the [ChatML](https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/ai-services/openai/includes/chat-markup-language.md) or [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/discussions/3) prompt format (depends on the model). The system prompt is used to provide information about the state of the Home Assistant installation including available devices and callable services.

Example "system" prompt: 
```
You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.
The current time and date is 08:12 AM on Thursday March 14, 2024
Services: light.turn_off(), light.turn_on(brightness,rgb_color), fan.turn_on(), fan.turn_off()
Devices:
light.office 'Office Light' = on;80%
fan.office 'Office fan' = off
light.kitchen 'Kitchen Light' = on;80%;red
light.bedroom 'Bedroom Light' = off
```

For more about how the model is prompted see [Model Prompting](/docs/Model%20Prompting.md)

Output from the model will consist of a response that should be relayed back to the user, along with an optional code block that will invoke different Home Assistant "services". The output format from the model for function calling is as follows:

`````
turning on the kitchen lights for you now
```homeassistant
{ "service": "light.turn_on", "target_device": "light.kitchen" }
```
`````

Due to the mix of data used during fine tuning, the 3B model is also capable of basic instruct and QA tasks. For example, the model is able to perform basic logic tasks such as the following (Zephyr prompt format shown):

```
<|system|>You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.
*snip*
<|endoftext|>
<|user|>
if mary is 7 years old, and I am 3 years older than her. how old am I?<|endoftext|>
<|assistant|>
If Mary is 7 years old, then you are 10 years old (7+3=10).<|endoftext|>
```

### Synthetic Dataset
The synthetic dataset is aimed at covering basic day to day operations in home assistant such as turning devices on and off.
The supported entity types are: light, fan, cover, lock, media_player, climate, switch

The dataset is available on HuggingFace: https://huggingface.co/datasets/acon96/Home-Assistant-Requests  
The source for the dataset is in the [data](/data) of this repository.

### Training
The 3B model was trained as a full fine-tuning on 2x RTX 4090 (48GB). Training time took approximately 28 hours. It was trained on the `--large` dataset variant.

<details>
<summary>Training Arguments</summary>

```console
accelerate launch --config_file fsdp_config.yaml train.py \
    --run_name home-3b \
    --base_model stabilityai/stablelm-zephyr-3b \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --learning_rate 1e-5 --batch_size 64 --epochs 1 \
    --micro_batch_size 2 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 \
    --save_steps 50 --save_total_limit 10 --eval_steps 100 --logging_steps 2
```

</details>

The 1B model was trained as a full fine-tuning on an RTX 3090 (24GB). Training took approximately 2 hours. It was trained on the `--medium` dataset variant.

<details>
<summary>Training Arguments</summary>

```console
python3 train.py \
    --run_name home-1b \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 2e-5 --batch_size 32 \
    --micro_batch_size 8 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 --save_steps 100 --save_total_limit 10
```

</details>
<br/>

## Home Assistant Addon
In order to facilitate running the project entirely on the system where Home Assistant is installed, there is an experimental Home Assistant Add-on that runs the oobabooga/text-generation-webui to connect to using the "remote" backend options.  The addon can be found in the [addon/](./addon/README.md) directory.


## Version History
| Version | Description                                                                                                                                                                                                          |
|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v0.3.3  | Improvements to the Generic OpenAI Backend, improved area handling, fix issue using RGB colors, remove EOS token from responses, replace requests dependency with aiohttp included with Home Assistant               |
| v0.3.2  | Fix for exposed script entities causing errors, fix missing GBNF error, trim whitespace from model output                                                                                                            |
| v0.3.1  | Adds basic area support in prompting, Fix for broken requirements, fix for issue with formatted tools, fix custom API not registering on startup properly                                                            |
| v0.3    | Adds support for Home Assistant LLM APIs, improved model prompting and tool formatting options, and automatic detection of GGUF quantization levels on HuggingFace                                                   |
| v0.2.17 | Disable native llama.cpp wheel optimizations, add Command R prompt format                                                                                                                                            |
| v0.2.16 | Fix for missing huggingface_hub package preventing startup                                                                                                                                                           |
| v0.2.15 | Fix startup error when using llama.cpp backend and add flash attention to llama.cpp backend                                                                                                                          |
| v0.2.14 | Fix llama.cpp wheels + AVX detection                                                                                                                                                                                 |
| v0.2.13 | Add support for Llama 3, build llama.cpp wheels that are compatible with non-AVX systems, fix an error with exposing script entities, fix multiple small Ollama backend issues, and add basic multi-language support |
| v0.2.12 | Fix cover ICL examples, allow setting number of ICL examples, add min P and typical P sampler options, recommend models during setup, add JSON mode for Ollama backend, fix missing default options                  |
| v0.2.11 | Add prompt caching, expose llama.cpp runtime settings, build llama-cpp-python wheels using GitHub actions, and install wheels directly from GitHub                                                                   |
| v0.2.10 | Allow configuring the model parameters during initial setup, attempt to auto-detect defaults for recommended models, Fix to allow lights to be set to max brightness                                                 |
| v0.2.9  | Fix HuggingFace Download, Fix llama.cpp wheel installation, Fix light color changing, Add in-context-learning support                                                                                                |
| v0.2.8  | Fix ollama model names with colons                                                                                                                                                                                   |
| v0.2.7  | Publish model v3, Multiple Ollama backend improvements, Updates for HA 2024.02, support for voice assistant aliases                                                                                                  |
| v0.2.6  | Bug fixes, add options for limiting chat history, HTTPS endpoint support, added zephyr prompt format.                                                                                                                |
| v0.2.5  | Fix Ollama max tokens parameter, fix GGUF download from Hugging Face, update included llama-cpp-python to 0.2.32, and add parameters to function calling for dataset + component, & model update                     |
| v0.2.4  | Fix API key auth on model load for text-generation-webui, and add support for Ollama API backend                                                                                                                     |
| v0.2.3  | Fix API key auth, Support chat completion endpoint, and refactor to make it easier to add more remote backends                                                                                                       |
| v0.2.2  | Fix options window after upgrade, fix training script for new Phi model format, and release new models                                                                                                               |
| v0.2.1  | Properly expose generation parameters for each backend, handle config entry updates without reloading, support remote backends with an API key                                                                       |
| v0.2    | Bug fixes, support more backends, support for climate + switch devices, JSON style function calling with parameters, GBNF grammars                                                                                   |
| v0.1    | Initial Release                                                                                                                                                                                                      |
