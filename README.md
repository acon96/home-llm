# Home LLM
This project provides the required "glue" components to control your Home Assistant installation with a completely local Large Language Model acting as a personal assistant. The goal is to provide a drop in solution to be used as a "conversation agent" component type by the Home Assistant project.

## Model
The "Home" model is a fine tuning of the Phi model series from Microsoft.  The model is able to control devices in the user's house as well as perform basic question and answering.  The fine tuning dataset is a combination of the [Cleaned Stanford Alpaca Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned) as well as a [custom synthetic dataset](./data) designed to teach the model function calling based on the device information in the context.

The model can be found on HuggingFace: https://huggingface.co/acon96/Home-3B-v1-GGUF

The model is quantized using Llama.cpp in order to enable running the model in super low resource environments that are common with Home Assistant installations such as Raspberry Pis.

The model can be used as an "instruct" type model using the ChatML prompt format. The system prompt is used to provide information about the state of the Home Assistant installation including available devices and callable services.

Example "system" prompt: 
```
<|im_start|>system You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.
Services: light.turn_off, light.turn_on, fan.turn_on, fan.turn_off
Devices:
light.office 'Office Light' = on
fan.office 'Office fan' = off
light.kitchen 'Kitchen Light' = on
<|im_end|>
```

Output from the model will consist of a response that should be relayed back to the user, along with an optional code block that will invoke different Home Assistant "services". The output format from the model for function calling is as follows:

`````
<|im_start|>assistant turning on the kitchen lights for you now
```homeassistant
light.turn_on(light.kitchen)
```
<|im_end|><|endoftext|>
`````

Due to the mix of data used during fine tuning, the model is also capable of basic instruct and QA tasks. For example, the model is able to perform basic logic tasks such as the following:

```
<|im_start|>system You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.
*snip*
<|im_end|>
<|im_start|>user if mary is 7 years old, and I am 3 years older than her. how old am I?<|im_end|>
<|im_start|>assistant If Mary is 7 years old, then you are 10 years old (7+3=10).<|im_end|><|endoftext|>
```

### Synthetic Dataset
The synthetic dataset is aimed at covering basic day to day operations in home assistant such as turning devices on and off.
The supported entity types are: light, fan, cover, lock, media_player

### Training
The model was trained as a LoRA on an RTX 3090 (24GB) using the following settings for the custom training script. The embedding weights were "saved" and trained normally along with the rank matricies in order to train the newly added tokens to the embeddings. The full model is merged together at the end.

```
python3 train.py \
    --run_name home-llm-rev11_1 \
    --base_model microsoft/phi-2 \
    --add_pad_token \
    --add_chatml_tokens \
    --bf16 \
    --train_dataset data/home_assistant_alpaca_merged_train.json \
    --test_dataset data/home_assistant_alpaca_merged_test.json \
    --learning_rate 1e-5 \
    --save_steps 1000 \
    --micro_batch_size 2 --gradient_checkpointing \
    --ctx_size 2048 \
    --use_lora --lora_rank 32 --lora_alpha 64 --lora_modules fc1,fc2,Wqkv,out_proj --lora_modules_to_save wte,lm_head.linear --lora_merge
```

The provided `custom_modeling_phi.py` has Gradient Checkpointing implemented for the MHA and MLP modules, allowing for significantly reduced VRAM usage during training.

## Home Assistant Component
In order to integrate with Home Assistant, we provide a `custom_component` that exposes the locally running LLM as a "conversation agent" that can be interacted with using a chat interface as well as integrate with Speech-to-Text and Text-to-Speech addons to enable interacting with the model by speaking.  

The component can either run the model directly as part of the Home Assistant software using llama-cpp-python, or you can run the [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) project to provide access to the LLM via an API interface. When doing this, you can host the model yourself and point the add-on at machine where the model is hosted, or you can run the model using text-generation-webui using the provided [custom Home Assistant add-on](./addon/README.md).

### Installing
1. Ensure you have either the Samba, SSH, FTP, or another add-on installed that gives you access to the `config` folder
2. If there is not already a `custom_components` folder, create one now.
3. Copy the `custom_components/llama_conversation` folder from this repo to `config/custom_components/llama_conversation` on your Home Assistant machine.
4. Restart Home Assistant using the "Developer Tools" tab -> Services -> Run `homeassistant.restart`
5. The "LLaMA Conversation" integration should show up in the "Devices" section now.

### Setting up
When setting up the component, there are 3 different "backend" options to choose from:
1. Llama.cpp with a model from HuggingFace
2. Llama.cpp with a locally provided model
3. A remote instance of text-generation-webui

**Setting up the Llama.cpp backend with a model from HuggingFace**:
TODO: need to build wheels for llama.cpp first
**Setting up the Llama.cpp backend with a locally downloaded model**:
TODO: need to build wheels for llama.cpp first

**Setting up the "remote" backend**:

You need the following settings in order to configure the "remote" backend
1. Hostname: the host of the machine where text-generation-webui API is hosted. If you are using the provided add-on then the hostname is `local-text-generation-webui`
2. Port: the port for accessing the text-generation-webui API. NOTE: this is not the same as the UI port. (Usually 5000)
3. Name of the Model: This name must EXACTLY match the name as it appears in `text-generation-webui`

On creation, the component will validate that the model is available for use.

### Configuring the component as a Conversation Agent
**NOTE: ANY DEVICES THAT YOU SELECT TO BE EXPOSED TO THE MODEL WILL BE ADDED AS CONTEXT AND POTENTIALLY HAVE THEIR STATE CHANGED BY THE MODEL. ONLY EXPOSE DEVICES THAT YOU ARE OK WITH THE MODEL MODIFYING THE STATE OF, EVEN IF IT IS NOT WHAT YOU REQUESTED. THE MODEL MAY OCCASIONALLY HALLUCINATE AND ISSUE COMMANDS TO THE WRONG DEVICE! USE AT YOUR OWN RISK.**

In order to utilize the conversation agent in HomeAssistant:
1. Navigate to "Settings" -> "Voice Assistants"
2. Select "+ Add Assistant"
3. Name the assistant whatever you want.
4. Select the "Conversation Agent" that we created previously
5. If using STT or TTS configure these now
6. Return to the "Overview" dashboard and select chat icon in the top left.

From here you can submit queries to the AI agent.

In order for any entities be available to the agent, you must "expose" them first.
1. Navigate to "Settings" -> "Voice Assistants" -> "Expose" Tab
2. Select "+ Expose Entities" in the bottom right
3. Check any entities you would like to be exposed to the conversation agent.

### Running the text-generation-webui add-on
In order to facilitate running the project entirely on the system where Home Assistant is installed, there is an experimental Home Assistant Add-on that runs the oobabooga/text-generation-webui to connect to using the "remote" backend option.

1. Ensure you have either the Samba, SSH, FTP, or another add-on installed that gives you access to the `addons` folder
2. Copy the `addon` folder from this repo to `addons/text-generation-webui` on your Home Assistant machine.
3. Go to the "Add-ons" section in settings and then pick the "Add-on Store" from the bottom right corner.
4. Select the 3 dots in the top right and click "Check for Updates" and Refresh the webpage.
5. There should now be a "Local Add-ons" section at the top of the "Add-on Store"
6. Install the `oobabooga-text-generation-webui` add-on. It will take ~15-20 minutes to build the image on a Raspberry Pi.
7. Copy any models you want to use to the `addon_configs/local_text-generation-webui/models` folder.

### Performance of running the model on a Raspberry Pi
The RPI4 4GB that I have was sitting right at 1.5 tokens/sec for prompt eval and 1.6 tokens/sec for token generation when running the `Q4_K_M` quant. I was reliably getting responses in 30-40 seconds after the initial prompt processing which took almost 5 minutes. I highly recommend if you set up text-generation-webui on a separate machine that can take advantage of a GPU.