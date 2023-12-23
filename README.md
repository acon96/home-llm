# Home LLM
This project provides the required "glue" components to control your Home Assistant installation with a completely local Large Langage Model acting as a personal assistant. The goal is to provide a drop in solution to be used as a "conversation agent" component type by the Home Assistant project.

## Model
The "Home" model is a fine tuning of the Phi model series from Microsoft.  The model is able to control devices in the user's house as well as perform basic question and answering.  The fine tuning dataset is a combination of the [Cleaned Stanford Alpaca Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned) as well as a [custom synthetic dataset](./data) designed to teach the model function calling based on the device information in the context.

The model is quantized using Llama.cpp in order to enable running the model in super low resource environments that are common with Home Assistant installations such as Rapsberry Pis.

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

The component can either run the model directly as part of the Home Assistant software using llama-cpp-python, or you can run the [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) project to provide access to the LLM via an API interface. When doing this, you can host the model yourself and point the addon at machine where the model is hosted, or you can run the model using text-generation-webui using the provided [custom Home Assistant addon](./addon/README.md).

### Setting up
When setting up the component, there are 3 different "backend" options to choose from:
1. Llama.cpp with a model from HuggingFace
2. Llama.cpp with a locally provided model
3. A remote instance of text-generateion-webui

**Setting up the Llama.cpp backend with a model from HuggingFace**:

**Setting up the Llama.cpp backend with a locally downloaded model**:

**Setting up the "remote" backend**:
