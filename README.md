# Home LLM
This project provides the required "glue" components to control your Home Assistant installation with a completely local Large Language Model acting as a personal assistant. The goal is to provide a drop in solution to be used as a "conversation agent" component by Home Assistant.

### Home Assistant Component

In order to integrate with Home Assistant, we provide a `custom_component` that exposes the locally running LLM as a "conversation agent".

This component can be interacted with in a few ways:

- using a chat interface so you can chat with it.
- integrating with Speech-to-Text and Text-to-Speech addons so you can just speak to it.

The component can either run the model directly as part of the Home Assistant software using llama-cpp-python, or you can run the [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) project to provide access to the LLM via an API interface.

When doing this, you can host the model yourself and point the add-on at machine where the model is hosted, or you can run the model using text-generation-webui using the provided [custom Home Assistant add-on](./addon).


## Requirements

- supported version of HomeAssistant
- `llama-cpp-python>=0.2.29`
- [HACs](https://hacs.xyz/docs/setup/download/) (if you want to install it that way)
- SSH or Web Terminal access to your HomeAssistant instance.


## 🏃 Getting Started

Installing and configuration HomeLLM will involve several steps: 

1. ⬇️ Install `llama-cpp-python`
2. 💾 Install the HomeLLM component
3. ⚙️ Choose and Configure a Backend
3. 🗣️ Configure the Voice Assistant

### ⬇️ Install `llama-cpp-python`

Obtain terminal access to the HomeAssistant instance and create some prerequisite folders. We'll download a set of prebundled python wheel files.

```console
mkdir -p /config/custom_components/llama_conversation
cd /config/custom_components/llama_conversation

wget https://github.com/acon96/home-llm/raw/develop/dist/llama_cpp_python-0.2.38-cp311-cp311-musllinux_1_2_aarch64.whl
wget https://github.com/acon96/home-llm/raw/develop/dist/llama_cpp_python-0.2.38-cp311-cp311-musllinux_1_2_x86_64.whl
```

> ❔ 🤔 How to get Terminal Access?
> 
> There'll be many ways, but for the sake of simplicity you can try out these 
> addons: 
> 
> - https://github.com/hassio-addons/repository?tab=readme-ov-file#-studio-code-server
> - https://github.com/hassio-addons/repository?tab=readme-ov-file#-advanced-ssh--web-terminal


### 💾 🚕 Install HomeLMM with HACs

> 🛑 ✋🏻 Requires HACs
> 
> First make sure you have [HACs installed](https://hacs.xyz/docs/setup/download/)

Once you have HACs installed, this button will help you add the repository to HACS and open the download page

1. [![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?category=Integration&repository=home-llm&owner=acon96)
2. Restart Home Assistant

The "LLaMA Conversation" integration should show up in the `Settings > Devices and Services > [Devices]` tab now:
![image](https://github.com/acon96/home-llm/assets/61225/4427e362-e443-4796-bee8-5bdda18305d0)


### 💾 🔨 Install HomeLMM Manually

1. Ensure you have either the Samba, SSH, FTP, or another add-on installed that gives you access to the `config` folder
2. If there is not already a `custom_components` folder, create one now.
3. Copy the `custom_components/llama_conversation` folder from this repo to `config/custom_components/llama_conversation` on your Home Assistant machine.
4. Restart Home Assistant: `Developer Tools -> Services -> Run` : `homeassistant.restart`

The "LLaMA Conversation" integration should show up in the `Settings > Devices and Services > [Devices]` tab now:
![image](https://github.com/acon96/home-llm/assets/61225/4427e362-e443-4796-bee8-5bdda18305d0)


### ⚙️ Configuration and Setup

When setting up the component, there are 4 different "backend" options to choose from:

a. Llama.cpp with a model from HuggingFace
b. Llama.cpp with a locally provided model
c. A remote instance of text-generation-webui
d. A generic OpenAI API compatible interface; *should* be compatible with LocalAI, LM Studio, and all other OpenAI compatible backends

See [docs/Backend Configuration.md](/docs/Backend%20Configuration.md) for more info.

#### a. Installing llama-cpp-python for local model usage

In order to run a model directly as part of your Home Assistant installation, you will need to install one of the pre-build wheels because there are no existing musllinux wheels for the package. Compatible wheels for x86_x64 and arm64 are provided in the [dist](./dist) folder. Copy the `*.whl` files to the `custom_components/llama_conversation/` folder. They will be installed while setting up the component.

#### b. Setting up the Llama.cpp backend with a model from HuggingFace

You need the following settings to configure the local backend from HuggingFace:
1. Model Name: the name of the model in the form `repo/model-name`. The repo MUST contain a GGUF quantized model.
2. Model Quantization: The quantization level to download. Pick from the list. Higher quantizations use more RAM but have higher quality responses.

#### c. Setting up the Llama.cpp backend with a locally downloaded model

You need the following settings to configure the local backend from HuggingFace:
1. Model File Name: the file name where Home Assistant can access the model to load. Most likely a sub-path of `/config` or `/media` or wherever you copied the model file to.

#### d. Setting up the "remote" backends

You need the following settings in order to configure the "remote" backend:
1. Hostname: the host of the machine where text-generation-webui API is hosted. If you are using the provided add-on then the hostname is `local-text-generation-webui` or `f459db47-text-generation-webui` depending on how the addon was installed.
2. Port: the port for accessing the text-generation-webui API. NOTE: this is not the same as the UI port. (Usually 5000)
3. Name of the Model: This name must EXACTLY match the name as it appears in `text-generation-webui`

With the remote text-generation-webui backend, the component will validate that the selected model is available for use and will ensure it is loaded remotely. The Generic OpenAI compatible version does NOT do any validation or model loading.

**Setting up with LocalAI**:  
If you are an existing LocalAI user or would like to use LocalAI as your backend, please refer to [this](https://io.midori-ai.xyz/howtos/setup-with-ha/) website which has instructions on how to setup LocalAI to work with Home-LLM including automatic installation of the latest version of the the Home-LLM model. The auto-installer (LocalAI Manager) will automatically download and setup LocalAI and/or the model of your choice and automatically create the necessary template files for the model to work with this integration.


### 🗣️ Configuring the component as a Conversation Agent

1. Navigate to `Settings` -> `Voice Assistants`
2. Select `+ Add Assistant`
3. Name the assistant whatever you want.
4. Select the [conversation agent](#link-to-the-title-id-where-you-guide-the-user-in-doing-this) that we created previously.
5. If using STT or TTS configure these now
6. Return to the "Overview" dashboard and select chat icon in the top left.
7. From here you can submit queries to the AI agent.

In order for any entities be available to the agent, you must "expose" them first.

1. Navigate to "Settings" -> "Voice Assistants" -> "Expose" Tab
2. Select "+ Expose Entities" in the bottom right
3. Check any entities you would like to be exposed to the conversation agent.

> 🛑 ✋🏻 Security Warning 
> 
> Any devices that you select to be exposed to the model will be added as 
> context and potentially have their state changed by the model.
> 
> Only expose devices that you want the model modifying the state of.
>
> The model may occasionally hallucinate and issue commands to the wrong device!
> 
> Use.At.Your.Own.Risk 💣 


## Technical Details


### Constrained Grammar

When running the model locally with [Llama.cpp], the component also constrains the model output using a GBNF grammar.
This forces the model to provide valid output no matter what since its outputs are constrained to valid JSON every time.
This helps the model perform significantly better at lower quantization levels where it would previously generate syntax errors.

For more information See [output.gbnf](./custom_components/llama_conversation/output.gbnf) for the existing grammar.


### Model
The "Home" models are a fine tuning of the Phi model series from Microsoft.  The model is able to control devices in the user's house as well as perform basic question and answering.  The fine tuning dataset is a combination of the [Cleaned Stanford Alpaca Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned) as well as a [custom synthetic dataset](./data) designed to teach the model function calling based on the device information in the context.

The latest models can be found on HuggingFace:  
3B v2 (Based on Phi-2): https://huggingface.co/acon96/Home-3B-v2-GGUF  
1B v2 (Based on Phi-1.5): https://huggingface.co/acon96/Home-1B-v2-GGUF  

<details>

<summary>Old Models</summary>  

3B v1 (Based on Phi-2): https://huggingface.co/acon96/Home-3B-v1-GGUF  
1B v1 (Based on Phi-1.5): https://huggingface.co/acon96/Home-1B-v1-GGUF  

</details>

Make sure you have `llama-cpp-python>=0.2.29` in order to run these models.

The main difference between the 2 models (besides parameter count) is the training data. The 1B model is ONLY trained on the synthetic dataset provided in this project, while the 3B model is trained on a mixture of this synthetic dataset, and the cleaned Stanford Alpaca dataset.

The model is quantized using Llama.cpp in order to enable running the model in super low resource environments that are common with Home Assistant installations such as Raspberry Pis.

The model can be used as an "instruct" type model using the ChatML prompt format. The system prompt is used to provide information about the state of the Home Assistant installation including available devices and callable services.

Example "system" prompt: 
```
<|im_start|>system
You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.
Services: light.turn_off(), light.turn_on(brightness,rgb_color), fan.turn_on(), fan.turn_off()
Devices:
light.office 'Office Light' = on;80%
fan.office 'Office fan' = off
light.kitchen 'Kitchen Light' = on;80%;red
light.bedroom 'Bedroom Light' = off<|im_end|>
```

For more about how the model is prompted see [Model Prompting](/docs/Model%20Prompting.md)

Output from the model will consist of a response that should be relayed back to the user, along with an optional code block that will invoke different Home Assistant "services". The output format from the model for function calling is as follows:

`````
<|im_start|>assistant
turning on the kitchen lights for you now
```homeassistant
{ "service": "light.turn_on", "target_device": "light.kitchen" }
```<|im_end|>
`````

Due to the mix of data used during fine tuning, the 3B model is also capable of basic instruct and QA tasks. For example, the model is able to perform basic logic tasks such as the following:

```
<|im_start|>system You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.
*snip*
<|im_end|>
<|im_start|>user
if mary is 7 years old, and I am 3 years older than her. how old am I?<|im_end|>
<|im_start|>assistant
If Mary is 7 years old, then you are 10 years old (7+3=10).<|im_end|>
```

### Synthetic Dataset
The synthetic dataset is aimed at covering basic day to day operations in home assistant such as turning devices on and off.
The supported entity types are: light, fan, cover, lock, media_player, climate, switch

The dataset is available on HuggingFace: https://huggingface.co/datasets/acon96/Home-Assistant-Requests  
The source for the dataset is in the [data](/data) of this repository.

### Training
The 3B model was trained as a LoRA on an RTX 3090 (24GB) using the following settings for the custom training script. The embedding weights were "saved" and trained normally along with the rank matricies in order to train the newly added tokens to the embeddings. The full model is merged together at the end. Training took approximately 10 hours.

<details>
<summary>Training Arguments</summary>

```console
python3 train.py \
    --run_name home-3b \
    --base_model microsoft/phi-2 \
    --add_pad_token \
    --add_chatml_tokens \
    --bf16 \
    --train_dataset data/home_assistant_alpaca_merged_train.json \
    --learning_rate 1e-5 \
    --save_steps 1000 \
    --micro_batch_size 2 --gradient_checkpointing \
    --ctx_size 2048 \
    --group_by_length \
    --use_lora --lora_rank 32 --lora_alpha 64 --lora_modules fc1,fc2,q_proj,v_proj,dense --lora_modules_to_save embed_tokens,lm_head --lora_merge
```

</details>

The 1B model was trained as a full fine-tuning on on an RTX 3090 (24GB). Training took approximately 2.5 hours.

<details>
<summary>Training Arguments</summary>

```console
python3 train.py \
    --run_name home-1b \
    --base_model microsoft/phi-1_5 \
    --add_pad_token \
    --add_chatml_tokens \
    --bf16 \
    --train_dataset data/home_assistant_train.json \
    --learning_rate 1e-5 \
    --micro_batch_size 4 --gradient_checkpointing \
    --ctx_size 2048
```

</details>
<br/>



### Running the text-generation-webui add-on
In order to facilitate running the project entirely on the system where Home Assistant is installed, there is an experimental Home Assistant Add-on that runs the oobabooga/text-generation-webui to connect to using the "remote" backend option.

You can use this button to automatically download and build the addon for `oobabooga/text-generation-webui`

[![Open your Home Assistant instance and show the dashboard of an add-on.](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=f459db47_text-generation-webui&repository_url=https%3A%2F%2Fgithub.com%2Facon96%2Fhome-llm)

If the automatic installation fails then you can install the addon manually using the following steps:

1. Ensure you have either the Samba, SSH, FTP, or another add-on installed that gives you access to the `addons` folder
2. Copy the `addon` folder from this repo to `addons/text-generation-webui` on your Home Assistant machine.
3. Go to the "Add-ons" section in settings and then pick the "Add-on Store" from the bottom right corner.
4. Select the 3 dots in the top right and click "Check for Updates" and Refresh the webpage.
5. There should now be a "Local Add-ons" section at the top of the "Add-on Store"
6. Install the `oobabooga-text-generation-webui` add-on. It will take ~15-20 minutes to build the image on a Raspberry Pi.
7. Copy any models you want to use to the `addon_configs/local_text-generation-webui/models` folder or download them using the UI.
8. Load up a model to use. NOTE: The timeout for ingress pages is only 60 seconds so if the model takes longer than 60 seconds to load (very likely) then the UI will appear to time out and you will need to navigate to the add-on's logs to see when the model is fully loaded.

### Performance of running the model on a Raspberry Pi
The RPI4 4GB that I have was sitting right at 1.5 tokens/sec for prompt eval and 1.6 tokens/sec for token generation when running the `Q4_K_M` quant. I was reliably getting responses in 30-60 seconds after the initial prompt processing which took almost 5 minutes. It depends significantly on the number of devices that have been exposed as well as how many states have changed since the last invocation because llama.cpp caches KV values for identical prompt prefixes.

It is highly recommend to set up text-generation-webui on a separate machine that can take advantage of a GPU.

## Version History
| Version | Description                                                                                                                                                                                      |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v0.2.5  | Fix Ollama max tokens parameter, fix GGUF download from Hugging Face, update included llama-cpp-python to 0.2.32, and add parameters to function calling for dataset + component, & model update |
| v0.2.4  | Fix API key auth on model load for text-generation-webui, and add support for Ollama API backend                                                                                                 |
| v0.2.3  | Fix API key auth, Support chat completion endpoint, and refactor to make it easier to add more remote backends                                                                                   |
| v0.2.2  | Fix options window after upgrade, fix training script for new Phi model format, and release new models                                                                                           |
| v0.2.1  | Properly expose generation parameters for each backend, handle config entry updates without reloading, support remote backends with an API key                                                   |
| v0.2    | Bug fixes, support more backends, support for climate + switch devices, JSON style function calling with parameters, GBNF grammars                                                               |
| v0.1    | Initial Release                                                                                                                                                                                  |
