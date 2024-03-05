# Model Prompting

This integration allows for full customization of the system prompt using Home Assistant's [built in templating engine](https://www.home-assistant.io/docs/configuration/templating/). This gives it access to all of the information that it could possibly need out of the box including entity states, attributes, and pretty much anything in Home Assistant's state.  This allows you to expose as much or as little information to the model as you want.

## System Prompt Template
The default system prompt is:
```
You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.
Services: {{ services }}
Devices:
{{ devices }}
```

The `services` and `devices` variables are special variables that are provided by the integration and NOT Home Assistant. These are provided for simplicity in exposing the correct devices and services to the model without having to filter out entities that should not be exposed for the model to control.
- `services` expands into a comma separated list of the services that correlate with the devices that have been exposed to the Voice Assistant.
- `devices` expands into a multi-line block where each line is the format `<entity_id> '<friendly_name> = <state>;<extra_attributes_to_expose>`

### Model "Persona"
The model is trained with a few different personas. They can be activated by using their system prompt found below:

Al the Assistant - Responds politely and concisely
```
You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.
```

Blackbeard the Pirate - Sounds like a pirate
```
You are 'Blackbeard', a helpful AI Assistant that controls the devices in a house but sounds like a pirate. Complete the following task as instructed or answer the following question with the information provided only. Your response should always sound like you are a pirate.
```

Robo the Robot - Sounds like a robot
```
You are 'Robo', a helpful AI Robot that controls the devices in a house. Complete the following task as instructed or answer the following question with the information provided only. Your response should be robotic and always begin with 'Beep-Boop'.
```

## Prompt Format
On top of the system prompt, there is also a prompt "template" or prompt "format" that defines how you pass text to the model so that it follows the instruction fine tuning. The prompt format should match the prompt format that is specified by the model to achieve optimal results. 

Currently supported prompt formats are:
1. ChatML
2. Vicuna
3. Alpaca
4. Mistral
5. None (useful for foundation models)
