# Setup Instructions

1. [Home Assistant Component](#home-assistant-component)
2. [Configuring the LLM as a Conversation Agent](#configuring-as-a-conversation-agent)
3. [Setting up the text-generation-webui Addon](#text-generation-webui-add-on)

## Home Assistant Component
### Requirements

- A supported version of Home Assistant; `2023.10.0` or newer
- SSH or Samba access to your Home Assistant instance

**Optional:**
- [HACs](https://hacs.xyz/docs/setup/download/) (if you want to install it that way)

### 💾 🚕 Install the Home Assistant Component with HACs

> 🛑 ✋🏻 Requires HACs
> 
> First make sure you have [HACs installed](https://hacs.xyz/docs/setup/download/)

Once you have HACs installed, this button will help you add the repository to HACS and open the download page

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?category=Integration&repository=home-llm&owner=acon96)
 
**Remember to restart Home Assistant after installing the component!**

A "LLaMA Conversation" device should show up in the `Settings > Devices and Services > [Devices]` tab now:
![image](https://github.com/acon96/home-llm/assets/61225/4427e362-e443-4796-bee8-5bdda18305d0)


### 💾 🔨 Install the Home Assistant Component Manually

1. Ensure you have either the Samba, SSH, FTP, or another add-on installed that gives you access to the `config` folder
2. If there is not already a `custom_components` folder, create one now.
3. Copy the `custom_components/llama_conversation` folder from this repo to `config/custom_components/llama_conversation` on your Home Assistant machine.
4. Restart Home Assistant: `Developer Tools -> Services -> Run` : `HomeAssistant.restart`

A "LLaMA Conversation" device should show up in the `Settings > Devices and Services > [Devices]` tab now:
![image](https://github.com/acon96/home-llm/assets/61225/4427e362-e443-4796-bee8-5bdda18305d0)


### ⚙️ Configuration and Setup
You must configure at least one model by configuring the integration.

1. `Settings > Devices and Services`.
2. Click the `Add Integration` button in the bottom right of the screen.
3. Filter the list of "brand names" for llama, and "LLaMa Conversation" should remain.
4. Choose the backend you will be using to host the model:
    1. Using builtin llama.cpp with hugging face
    2. Using builtin llama.cpp with existing model file
    3. using text-generation-webui api
    4. using generic openapi compatiable api
    5. using ollama api

### llama-cpp-python Wheel Installation

If you plan on running the model locally on the same hardware as your Home Assistant server, then the recommended way to run the model is to use Llama.cpp. Unfortunately there are not pre-build wheels for this package for the musllinux runtime that Home Assistant Docker images use. To get around this, we provide compatible wheels for x86_x64 and arm64 in the [dist](./dist) folder. 

Download the `*.whl` file that matches your hardware and then copy the `*.whl` file to the `custom_components/llama_conversation/` folder. It will be installed as a configuration step while setting up the Home Assistant component.

| wheel | platform | home assistant version |
| --- | --- | --- |
| llama_cpp_python-{version}-cp311-cp311-musllinux_1_2_aarch64.whl | aarch64 (RPi 4 and 5) | `2024.1.4` and older |
| llama_cpp_python-{version}-cp311-cp311-musllinux_1_2_x86_64.whl | x86_64 (Intel + AMD) | `2024.1.4` and older |
| llama_cpp_python-{version}-cp312-cp312-musllinux_1_2_aarch64.whl | aarch64 (RPi 4 and 5) | `2024.2.0` and newer |
| llama_cpp_python-{version}-cp312-cp312-musllinux_1_2_x86_64.whl | x86_64 (Intel + AMD) | `2024.2.0` and newer |

### Constrained Grammar

When running the model locally with [Llama.cpp], the component also constrains the model output using a GBNF grammar.
This forces the model to provide valid output no matter what since its outputs are constrained to valid JSON every time.
This helps the model perform significantly better at lower quantization levels where it would previously generate syntax errors. It is recommended to turn this on when using the component as it will reduce the incorrect output from the model.

For more information See [output.gbnf](./custom_components/llama_conversation/output.gbnf) for the existing grammar.


### Backend Configuration

![image](https://github.com/airtonix/home-llm/assets/61225/6f5d9748-5bfc-47ce-8abc-4f07d389a73f)

When setting up the component, there are 5 different "backend" options to choose from:

a. Llama.cpp with a model from HuggingFace  <--- recommended if you are lost  
b. Llama.cpp with a locally provided model  
c. A remote instance of text-generation-webui  
d. A generic OpenAI API compatible interface; *should* be compatible with LocalAI, LM Studio, and all other OpenAI compatible backends  
e. Ollama api  

See [docs/Backend Configuration.md](/docs/Backend%20Configuration.md) for more info.

#### Llama.cpp Backend with a model from HuggingFace

This is option A

It is recommended to use either `acon96/Home-3B-v3-GGUF` or `acon96/Home-1B-v2-GGUF` as the model for this integration.
NOTE: if you are using `acon96/Home-3B-v3-GGUF`, you need to set the prompt template to `Zephyr` after setting up the component by configuring the model after creation.

You need the following settings to configure the local backend from HuggingFace:
1. **Model Name**: the name of the model in the form `repo/model-name`. The repo MUST contain a GGUF quantized model.
2. **Model Quantization**: The quantization level to download. Pick from the list. Higher quantizations use more RAM but have higher quality responses.

#### Llama.cpp Backend with a locally downloaded model

This is option B

Please download the model file from HuggingFace and copy it to your Home Assistant device. Recommended models are [acon96/Home-3B-v3-GGUF](https://huggingface.co/acon96/Home-3B-v3-GGUF) or [acon96/Home-1B-v2-GGUF](https://huggingface.co/acon96/Home-1B-v2-GGUF).

NOTE: if you are using `acon96/Home-3B-v3-GGUF`, you need to set the prompt template to `Zephyr` after setting up the component by configuring the model after creation.

You need the following settings to configure the local backend from HuggingFace:
1. **Model File Name**: the file name where Home Assistant can access the model to load. Most likely a sub-path of `/config` or `/media` or wherever you copied the model file to.

#### Remote Backends

This is options C, D and E

You need the following settings in order to configure the "remote" backend:
1. **Hostname**: the host of the machine where text-generation-webui API is hosted. If you are using the provided add-on then the hostname is `local-text-generation-webui` or `f459db47-text-generation-webui` depending on how the addon was installed.
2. **Port**: the port for accessing the text-generation-webui API. NOTE: this is not the same as the UI port. (Usually 5000)
3. **Name of the Model**: This name must EXACTLY match the name as it appears in `text-generation-webui`

With the remote text-generation-webui backend, the component will validate that the selected model is available for use and will ensure it is loaded remotely. The Generic OpenAI compatible version does NOT do any validation or model loading.

**Setting up with LocalAI**:  
If you are an existing LocalAI user or would like to use LocalAI as your backend, please refer to [this](https://io.midori-ai.xyz/howtos/setup-with-ha/) website which has instructions on how to setup LocalAI to work with Home-LLM including automatic installation of the latest version of the the Home-LLM model. The auto-installer (LocalAI Manager) will automatically download and setup LocalAI and/or the model of your choice and automatically create the necessary template files for the model to work with this integration.

## Configuring as a Conversation Agent

> 🛑 ✋🏻 Security Warning 
> 
> Any devices that you select to be exposed to the model will be added as 
> context and potentially have their state changed by the model.
> 
> Only expose devices that you want the model modifying the state of.
>
> The model may occasionally hallucinate and issue commands to the wrong device!
> 
> Use At Your Own Risk

1. Navigate to `Settings` -> `Voice Assistants`
2. Select `+ Add Assistant`
3. Name the assistant whatever you want.
4. Select the conversation agent that we created previously.
5. If using STT or TTS configure these now
6. Return to the "Overview" dashboard and select chat icon in the top left.
7. From here you can submit queries to the AI agent.

In order for any entities be available to the agent, you must "expose" them first.

1. Navigate to "Settings" -> "Voice Assistants" -> "Expose" Tab
2. Select "+ Expose Entities" in the bottom right
3. Check any entities you would like to be exposed to the conversation agent.

> Note:
> When exposing entities to the model, you are adding tokens to the model's context. If you exceed the context length of the model, then your interactions with the model will fail due to the instructions being dropped out of the context's sliding window.  
> It is recommended to only expose a maximum of 32 entities to this conversation agent at this time.

## text-generation-webui add-on
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