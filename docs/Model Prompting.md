# Model Prompting

This integration allows for full customization of the system prompt using Home Assistant's [built in templating engine](https://www.home-assistant.io/docs/configuration/templating/). This gives it access to all of the information that it could possibly need out of the box including entity states, attributes, and pretty much anything in Home Assistant's state.  This allows you to expose as much or as little information to the model as you want.

## System Prompt Template
The default system prompt is:
```
You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task ask instructed with the information provided only.
Services: {{ services }}
Devices:
{{ devices }}
```

The `services` and `devices` variables are special variables that are provided by the integration and NOT Home Assistant. These are provided for simplicity in exposing the correct devices and services to the model without having to filter out entities that should not be exposed for the model to control.
- `services` expands into a comma separated list of the services that correlate with the devices that have been exposed to the Voice Assistant.
- `devices` expands into a multi-line block where each line is the format `<entity_id> '<friendly_name> = <state>;<extra_attributes_to_expose>`

## Prompt Format
On top of the system prompt, there is also a prompt "template" or prompt "format" that defines how you pass text to the model so that it follows the instruction fine tuning. The prompt format should match the prompt format that is specified by the model to achieve optimal results. 

Currently supported prompt formats are:
1. ChatML
2. Vicuna
3. Alpaca
4. Mistral
5. None (useful for foundation models)