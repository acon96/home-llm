# Home LLM
This project provides the required "glue" components to control your Home Assistant installation with a completely local Large Language Model acting as a personal assistant. The goal is to provide a drop in solution to be used as a "conversation agent" component by Home Assistant.

## Quick Start
Please see the [Setup Guide](./docs/Setup.md) for more information on installation.

## Home Assistant Component
In order to integrate with Home Assistant, we provide a `custom_component` that exposes the locally running LLM as a "conversation agent".

This component can be interacted with in a few ways:  
- using a chat interface so you can chat with it.
- integrating with Speech-to-Text and Text-to-Speech addons so you can just speak to it.

The component can either run the model directly as part of the Home Assistant software using llama-cpp-python, or you can run the [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) project to provide access to the LLM via an API interface.

When doing this, you can host the model yourself and point the add-on at machine where the model is hosted, or you can run the model using text-generation-webui using the provided [custom Home Assistant add-on](./addon).

## Model
The "Home" models are a fine tuning of the Phi model series from Microsoft and the StableLM model series from StabilityAI.  The model is able to control devices in the user's house as well as perform basic question and answering.  The fine tuning dataset is a [custom synthetic dataset](./data) designed to teach the model function calling based on the device information in the context.

The latest models can be found on HuggingFace:  
3B v3 (Based on StableLM-Zephyr-3B): https://huggingface.co/acon96/Home-3B-v3-GGUF  (Zephyr prompt format)  
1B v2 (Based on Phi-1.5): https://huggingface.co/acon96/Home-1B-v2-GGUF  (ChatML prompt format)  

<details>

<summary>Old Models</summary>  

3B v2 (Based on Phi-2): https://huggingface.co/acon96/Home-3B-v2-GGUF  (ChatML prompt format)  
3B v1 (Based on Phi-2): https://huggingface.co/acon96/Home-3B-v1-GGUF  (ChatML prompt format)  
1B v1 (Based on Phi-1.5): https://huggingface.co/acon96/Home-1B-v1-GGUF  (ChatML prompt format)  

</details>

The model is quantized using Llama.cpp in order to enable running the model in super low resource environments that are common with Home Assistant installations such as Raspberry Pis.

The model can be used as an "instruct" type model using the [ChatML](https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/ai-services/openai/includes/chat-markup-language.md) or [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/discussions/3) prompt format (depends on the model). The system prompt is used to provide information about the state of the Home Assistant installation including available devices and callable services.

Example "system" prompt: 
```
You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.
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

## Home Assistant Addon
In order to facilitate running the project entirely on the system where Home Assistant is installed, there is an experimental Home Assistant Add-on that runs the oobabooga/text-generation-webui to connect to using the "remote" backend options.  The addon can be found in the [addon/](./addon/README.md) directory.


## Version History
| Version | Description                                                                                                                                                                                      |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v0.2.6  | Bug fixes, add options for limiting chat history, HTTPS endpoint support, added zephyr prompt format.                                                                                            |
| v0.2.5  | Fix Ollama max tokens parameter, fix GGUF download from Hugging Face, update included llama-cpp-python to 0.2.32, and add parameters to function calling for dataset + component, & model update |
| v0.2.4  | Fix API key auth on model load for text-generation-webui, and add support for Ollama API backend                                                                                                 |
| v0.2.3  | Fix API key auth, Support chat completion endpoint, and refactor to make it easier to add more remote backends                                                                                   |
| v0.2.2  | Fix options window after upgrade, fix training script for new Phi model format, and release new models                                                                                           |
| v0.2.1  | Properly expose generation parameters for each backend, handle config entry updates without reloading, support remote backends with an API key                                                   |
| v0.2    | Bug fixes, support more backends, support for climate + switch devices, JSON style function calling with parameters, GBNF grammars                                                               |
| v0.1    | Initial Release                                                                                                                                                                                  |
