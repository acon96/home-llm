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
* [Path 3: Using Llama-3-8B-Instruct with LM Studio](#path-3-using-llama-3-8b-instruct-with-lm-studio)
    * [Overview](#overview-2)
    * [Step 1: Downloading and serving the Model](#step-1-downloading-and-serving-the-model-1)
    * [Step 2: Connect to the LM Studio API](#step-2-connect-to-the-lm-studio-api)
    * [Step 3: Model Configuration](#step-3-model-configuration-2)
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

After installation, A "Local LLM Conversation" device should show up in the `Settings > Devices and Services > [Devices]` tab now.

## Path 1: Using the Home Model with the Llama.cpp Backend
### Overview
This setup path involves downloading a fine-tuned model from HuggingFace and integrating it with Home Assistant using the Llama.cpp backend. This option is for Home Assistant setups without a dedicated GPU, and the model is capable of running on most devices, and can even run on a Raspberry Pi (although slowly).

### Step 1: Wheel Installation for llama-cpp-python
1. In Home Assistant: navigate to `Settings > Devices and Services`
2. Select the `+ Add Integration` button in the bottom right corner
3. Search for, and select `Local LLM Conversation`
4. With the `Llama.cpp (HuggingFace)` backend selected, click `Submit`

This should download and install `llama-cpp-python` from GitHub. If the installation fails for any reason, follow the manual installation instructions [here](./Backend%20Configuration.md#wheels).

Once `llama-cpp-python` is installed, continue to the model selection.

### Step 2: Model Selection
The next step is to specify which model will be used by the integration. You may select any repository on HuggingFace that has a model in GGUF format in it.  We will use `acon96/Home-3B-v3-GGUF` for this example.  If you have less than 4GB of RAM then use `acon96/Home-1B-v3-GGUF`.

**Model Name**: Use either `acon96/Home-3B-v3-GGUF` or `acon96/Home-1B-v3-GGUF`  
**Quantization Level**: The model will be downloaded in the selected quantization level from the HuggingFace repository. If unsure which level to choose, select `Q4_K_M`.  

Pressing `Submit` will download the model from HuggingFace. The downloaded files will be stored by default in `/media/models/`.

**Note for Docker/sanboxed HA install users:** The model download may fail if it does not have the permissions to create the ```media``` folder in your Home Assistant install. To fix this, you will need to manually create the folder beside your existing ```config``` folder called ```media``` and set the permissions accordingly so that the addon can access it. If you're using Docker or similar, you may need to map the folder in your Compose file too and ```Update the Stack```. Once created and updated, you can open the model download screen again and it should now download as normal.

### Step 3: Model Configuration
This step allows you to configure how the model is "prompted". See [here](./Model%20Prompting.md) for more information on how that works.

For now, defaults for the model should have been populated. If you would like the model to be able to control devices then you must select the `Home-LLM (v1-v3)` API. This API is included to ensure compatability with the Home-LLM models that were trained before the introduction of the built in Home Assistant LLM API.

Once the desired API has been selected, scroll to the bottom and click `Submit`.

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
3. Search for, and select `Local LLM Conversation`
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

For now, defaults for the model should have been populated. If you would like the model to be able to control devices then you must select the `Assist` API.

Once the desired API has been selected, scroll to the bottom and click `Submit`.

> NOTE: The key settings in this case are that our prompt references the `{{ response_examples }}` variable and the `Enable in context learning (ICL) examples` option is turned on.

## Path 3: Using Llama-3-8B-Instruct with LM Studio
### Overview
Another model you can use if you have a GPU is Meta's Llama-3-8B-Instruct Model. This path assumes you have a machine with a GPU that already has [LM Studio](https://lmstudio.ai/) installed on it.  This path utilizes in-context learning examples, to prompt the model to produce the output that we expect.

### Step 1: Downloading and serving the Model
Llama 3 8B can be set up and downloaded on the serving machine using LM Studio by:
1. Search for `lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF` in the main interface.
2. Select and download the version of the model that is recommended for your VRAM configuration.
3. Select the 'Local Server' tab on the left side of the application.
4. Load the model by selecting it from the bar in the top middle of the screen. The server should start automatically when the model finishes loading.
5. Take note of the port that the server is running on.

### Step 2: Connect to the LM Studio API

1. In Home Assistant: navigate to `Settings > Devices and Services`
2. Select the `+ Add Integration` button in the bottom right corner
3. Search for, and select `Local LLM Conversation`
4. Select `Generic OpenAI Compatible API` from the dropdown and click `Submit`
5. Set up the connection to the API:
    - **IP Address**: Fill out IP Address for the machine hosting LM Studio
    - **Port**: enter the port that was listed in LM Studio
    - **Use HTTPS**: unchecked
    - **Model Name**: Set this to the name of the model as it appears in LM Studio. If you receive an error that the model does not exist, then select the model from the dropdown list.
    - **API Key**: leave blank
    - **API Path**: leave as `/v1`
6. Click `Submit`

### Step 3: Model Configuration
This step allows you to configure how the model is "prompted". See [here](./Model%20Prompting.md) for more information on how that works.

For now, defaults for the model should have been populated. If you would like the model to be able to control devices then you must select the `Assist` API.

Once the desired API has been selected, scroll to the bottom and click `Submit`.

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
