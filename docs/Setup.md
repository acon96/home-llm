# Quickstart Guide

## TOC
* [Intro](#intro)
    * [Requirements](#requirements)
* [Install the Home Assistant Component with HACs](#install-the-home-assistant-component-with-hacs)
* [Path 1: Using the Home Model with the Llama.cpp Backend](#path-1-using-the-home-model-with-llamacpp-backend)
    * [Overview](#overview)
    * [Step 1: Wheel Installation for llama-cpp-python](#step-1-wheel-installation-for-llama-cpp-python)
    * [Step 2: Model Selection](#step-2-model-selection)
    * [Step 3: Model Configuration](#step-3-model-configuration)
* [Path 2: Using Mistral-Instruct-7B with Ollama Backend](#path-2-using-mistral-instruct-7b-with-ollama-backend)
    * [Overview](#overview-1)
    * [Step 1: Downloading and serving the Model](#step-1-downloading-and-serving-the-model)
    * [Step 2: Connect to the Ollama API](#step-2-connect-to-the-ollama-api)
    * [Step 3: Model Configuration](#step-3-model-configuration-1)
* [Configuring the Integration as a Conversation Agent](#configuring-the-integration-as-a-conversation-agent)
* [Finished!](#finished)


## Intro
Welcome to the Quickstart guide for setting up Home-LLM. The component has MANY configuration options and is designed for experimentation with Home Assistant and LLMs. This guide outlines two main paths to get you started on your journey:
1. using the Llama.cpp backend with our provided fine-tuned model downloaded from HuggingFace  
2. using the Ollama backend with the Mistral-Instruct-7B model using in-context learning

### Requirements
- Knowledge of how to use a command line on Linux, MacOS, or Windows Subsystem for Linux
- A supported version of Home Assistant; `2023.10.0` or newer
- SSH or Samba access to your Home Assistant instance
- [HACs](https://hacs.xyz/docs/setup/download/) is installed

## Install the Home Assistant Component with HACs
The following link will open your Home Assistant installation and download the integration:  
**Remember to restart Home Assistant after installing the component!**

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?category=Integration&repository=home-llm&owner=acon96)

After installation, A "LLaMA Conversation" device should show up in the `Settings > Devices and Services > [Devices]` tab now.

## Path 1: Using the Home Model with the Llama.cpp Backend
### Overview
This setup path involves downloading a fine-tuned model from HuggingFace and integrating it with Home Assistant using the Llama.cpp backend. This option is for Home Assistant setups without a dedicated GPU, and the model is capable of running on most devices, and can even run on a Raspberry Pi (although slowly).

### Step 1: Wheel Installation for llama-cpp-python
In order to run the Llama.cpp backend as part of Home Assistant, we need to install the binary "wheel" distribution that is pre-built for compatibility with Home Assistant.

The `*.whl` files are located in the [/dist](/dist) folder of this repository.

To ensure compatibility with your Home Assistant and Python versions, select the correct `.whl` file for your hardware's architecture:
- For Home Assistant `2024.1.4` and older, use the Python 3.11 wheels (`cp311`)
- For Home Assistant `2024.2.0` and newer, use the Python 3.12 wheels (`cp312`)
- **ARM devices** (e.g., Raspberry Pi 4/5):
    - Example filenames:
        - `llama_cpp_python-{version}-cp311-cp311-musllinux_1_2_aarch64.whl`
        - `llama_cpp_python-{version}-cp312-cp312-musllinux_1_2_aarch64.whl`
- **x86_64 devices** (e.g., Intel/AMD desktops):
    - Example filenames:
        - `llama_cpp_python-{version}-cp311-cp311-musllinux_1_2_x86_64.whl`
        - `llama_cpp_python-{version}-cp312-cp312-musllinux_1_2_x86_64.whl`

Download the appropriate wheel and copy it to the `custom_components/llama_conversation/` directory.

After the wheel file has been copied to the correct folder.
1. In Home Assistant: navigate to `Settings > Devices and Services`
2. Select the `+ Add Integration` button in the bottom right corner
3. Search for, and select `LLaMA Conversation`
4. With the `Llama.cpp (HuggingFace)` backend selected, click `Submit`

This will trigger the installation of the wheel. If you ever need to update the version of Llama.cpp, you can copy a newer wheel file to the same folder, and re-create the integration; this will re-trigger the install process.

Once `llama-cpp-python` is installed, continue to the model selection.

### Step 2: Model Selection
The next step is to specify which model will be used by the integration. You may select any repository on HuggingFace that has a model in GGUF format in it.  We will use `acon96/Home-3B-v3-GGUF` for this example.  If you have less than 4GB of RAM then use `acon96/Home-1B-v2-GGUF`.

**Model Name**: Use either `acon96/Home-3B-v3-GGUF` or `acon96/Home-1B-v2-GGUF`  
**Quantization Level**: The model will be downloaded in the selected quantization level from the HuggingFace repository. If unsure which level to choose, select `Q4_K_M`.  

Pressing `Submit` will download the model from HuggingFace.

### Step 3: Model Configuration
This step allows you to configure how the model is "prompted". See [here](./Model%20Prompting.md) for more information on how that works.

For now, defaults for the model should have been populated and you can just scroll to the bottom and click `Submit`.

The model will be loaded into memory and should now be available to select as a conversation agent!

## Path 2: Using Mistral-Instruct-7B with Ollama Backend
### Overview
For those who have access to a GPU, you can also use the Mistral-Instruct-7B model to power your conversation agent. This path requires a separate machine that has a GPU and has [Ollama](https://ollama.com/) already installed on it.  This path utilizes in-context learning examples, to prompt the model to produce the output that we expect.

### Step 1: Downloading and serving the Model
Mistral can be easily set up and downloaded on the serving machine using the `ollama pull mistral` command.

In order to access the model from another machine, we need to run the Ollama API server open to the local network. This can be achieved using the `OLLAMA_HOST=0.0.0.0:11434 ollama serve` command. **DO NOT RUN THIS COMMAND ON ANY PUBLICLY
 ACCESSIBLE SERVERS AS IT LISTENS ON ALL NETWORK INTERFACES**

### Step 2: Connect to the Ollama API

1. In Home Assistant: navigate to `Settings > Devices and Services`
2. Select the `+ Add Integration` button in the bottom right corner
3. Search for, and select `LLaMA Conversation`
4. Select `Ollama API` from the dropdown and click `Submit`
5. Set up the connection to the API:
    - **IP Address**: Fill out IP Address for the machine hosting Ollama
    - **Port**: leave on `11434`
    - **Use HTTPS**: unchecked
    - **Model Name**: `mistral:latest`
    - **API Key**: leave blank
6. Click `Submit`

### Step 3: Model Configuration
This step allows you to configure how the model is "prompted". See [here](./Model%20Prompting.md) for more information on how that works.

For now, defaults for the model should have been populated and you can just scroll to the bottom and click `Submit`.

> NOTE: The key settings in this case are that our prompt references the `{{ response_examples }}` variable and the `Enable in context learning (ICL) examples` option is turned on.

## Configuring the Integration as a Conversation Agent
Now that the integration is configured and providing the conversation agent, we need to configure Home Assistant to use our conversation agent instead of the built in intent recognition system.

> 🛑 Warning 🛑
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
5. If you wish to use Speech to Text or Text to Speech, set those up now (left as an exercise to the reader)

In order for any entities be available to the agent, you must "expose" them first.  An exposed entity is added to the model's context and the model is able to call services on your behalf against those entities.

1. Navigate to "Settings" -> "Voice Assistants" -> "Expose" Tab
2. Select "+ Expose Entities" in the bottom right
3. Check any entities you would like to be exposed to the conversation agent.

> Note:
> When exposing entities to the model, you are adding tokens to the model's context. If you exceed the context length of the model, then your interactions with the model will fail due to the instructions being dropped out of the context's sliding window.  
> It is recommended to only expose a maximum of 32 entities to this conversation agent at this time.

## Finished!
Return to the "Overview" dashboard and select chat icon in the top left.  
From here you can chat with the AI model and request it to control your house.