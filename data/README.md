---
license: mit
task_categories:
  - question-answering
  - text-generation
tags:
  - automation
  - home
  - assistant
language:
  - en
pretty_name: Home Assistant Requests
size_categories:
  - 10K<n<100k
---

# Home Assistant Requests Dataset

This dataset contains a list of requests and responses for a user interacting with a personal assistant that controls an instance of [Home Assistant](https://www.home-assistant.io/).

The dataset is generated from the different CSV "piles". The "piles" contain different chunks of requests that are assembled into a final context that is presented to the LLM. For example, `piles/<language>/pile_of_device_names.csv` contains only names of various devices to be used as part of context as well as inserted into `piles/<language>/pile_of_templated_actions.csv` and `piles/<language>/pile_of_status_requests.csv`. The logic for assembling the final dataset from the piles is contained in [generate_home_assistant_data.py](./generate_home_assistant_data.py).

## Generating the dataset from piles

`python3 generate_home_assistant_data.py --train --test --large --sharegpt`

Supported dataset splits are `--test`, `--train`, & `--sample`
Arguments to set the train dataset size are `--small`, `--medium`, `--large`, & `--xl`.
Supported formats are `--raw_corpus` (chatml formatted) & `--sharegpt`
Languages can be enabled using `--language english german french spanish`

## Merging with other instruct-datasets for training

`python3 generate_home_assistant_data.py --merge <dataset>`

Supported datasets right now are: 
- `alpaca`
- `wizardlm70k`

Please note that the supported datasets all have different licenses. Be aware that the license of the resulting data mixture might be different that the license of this dataset alone.

## Adding a new personality
In order to add a new personality, you need to define a new system prompt and new set of responses for the assistant. The system prompt is the description of the assistant's behavior that occurs at the start of the context. The responses are what is said back to the user when performing a task. The model should still respond with the correct service call no matter what the assistant's response is. The list of system prompts are stored in `pile_of_system_prompts.csv`, and the list of responses are stored in `pile_of_responses.csv`

There are 2 columns in `pile_of_system_prompts.csv`:
- `persona`: the name of the persona
- `prompt`: the system prompt to use for that persona. Recommended to put this in quotes in case the prompt also has commas in it

The response pile is a CSV with the following headers: `service,response,language,persona,short`
- `service`: the service name that we are responding to. Make sure you cover enough different services so that the model can learn how to respond in all situations.
- `response`: the text of the response. Recommended to put this in quotes in case the response also has commas in it
- `persona`: the name of the persona the response belongs to. Use the name of your persona here
- `short`: either 0 or 1. If it is 1 then the response is considered "short', and can be combined together with other "short" responses using "and". These are used for examples where there are multiple service calls

Generating the full dataset using the python script will print out a warning for any responses that are missing for a persona.

## Adding new Home Assistant functionality
TODO
<!-- In order to add new home assistant device types, you will need to add data to a handful of piles, as well as make small modifications to the `generate_home_assistant_data.py` script.
1. Add 15-30 new device names with the new type to the `pile_of_device_names.csv`. This should be an entity_id and a 'friendly name'
2. Add 
 -->
