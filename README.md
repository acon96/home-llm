# Home LLM
This project provides the required "glue" components to control your Home Assistant installation with a completely local Large Langage Model acting as a personal assistant. The goal is to provide a drop in solution to be used as a "conversation agent" component type by the Home Assistant project.

## Model
The "Home" model is a fine tuning of the Phi model series from Microsoft.  The model is able to control devices in the user's house as well as perform basic question and answering.  The fine tuning dataset is a combination of the [Cleaned Stanford Alpaca Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned) as well as a [custom curated dataset](./data) designed to teach the model function calling.

The model is quantized using Lama.cpp in order to enable running the model in super low resource environments that are common with Home Assistant installations such as Rapsberry Pis.

The model can be used as an "instruct" type model using the ChatML prompt format. The system prompt is used to provide information about the state of the Home Assistant installation including available devices and callable services.

Example "system" prompt: 
```
You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.
Services: light.turn_off, light.turn_on, fan.turn_on, fan.turn_off
Devices:
light.office 'Office Light' = on
fan.office 'Office fan' = off
light.kitchen 'Kitchen Light' = on
```

Output from the model will consist of a response that should be relayed back to the user, along with an optional code block that will invoke different Home Assistant "services". The output format from the model for function calling is as follows:

`````
turning on the kitchen lights for you now
```homeassistant
light.turn_on(light.kitchen)
```
`````

## Home Assistant Component
In order to integrate with Home Assistant, we provide a `custom_component` that exposes the locally running LLM as an "conversation agent" that can be interacted with using a chat like interface as well as integrate with Speech-to-Text as well as Text-to-Speech addons to enable interacting with the model by speaking.  

The component can either run the model directly as part of the Home Assistant software using llama-cpp-python, or you can run the [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) project to provide access to the LLM via an API interface. When doing this, you can host the model yourself and point the addon at machine where the model is hosted, or you can run the model using text-generation-webui using the provided [custom Home Assistant addon](./addon/README.md).