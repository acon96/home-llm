# Home LLM
This project provides the required components to control your Home Assistant installation with a completely local Large Langage Model acting as a personal assistant. The goal is to provide a drop in solution to be used as an "agent" component type by the Home assistant project.

## Model
The "Home" model is a fine tuning of the Phi model series from Microsoft.  The model is able to control devices in the user's house as well as perform basic question and answering.  The fine tuning dataset is a combination of the [Cleaned Stanford Alpaca Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned) as well as a [custom curated dataset](./data) designed to teach the model function calling.

The model is quantized using Lama.cpp in order to enable running the model in super low resource environments that are common with Home Assistant installations such as Rapsberry Pis.

## Home Assistant Component
In order to integrate with Home Assistant, we provide a `custom_component` that exposes the locally running LLM as an "conversation agent" that can be interacted with using a chat like interface as well as integrate with Speech-to-Text as well as Text-to-Speech addons to enable interacting with the model by speaking.  The component can either run the model directly as part of the Home Assistant software using llama-cpp-python, or you can run the [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) project to provide access to the LLM via an API interface. When doing this, you can host the model yourself and point the addon at machine where the model is hosted, or you can run the model using text-generation-webui using the provided [custom Home Assistant addon](./addon/README.md).