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

The dataset is generated from the different CSV "piles". The "piles" contain different chunks of requests that are assembled into a final context that is presented to the LLM. For example, `piles/pile_of_device_names.csv` contains only names of various devices to be used as part of context as well as inserted into `piles/pile_of_templated_actions.csv` and `piles/pile_of_status_requests.csv`. The logic for assembling the final dataset from the piles is contained in [generate_home_assistant_data.py](./generate_home_assistant_data.py).

## Generating the dataset from piles

`python3 generate_home_assistant_data.py --train --test --large`

Supported dataset splits are `--test`, `--train`, & `--sample`
Arguments to set the train dataset size are `--small`, `--medium`, `--large`, & `--xl`.

## Merging with other instruct-datasets for training

`python3 generate_home_assistant_data.py --merge <dataset>`

Supported datasets right now are: 
- `alpaca`
- `wizardlm70k`

Please note that the supported datasets all have different licenses. Be aware that the license of the resulting data mixture might be different that the license of this dataset alone.

## Adding new Home Assistant functionality
Adding new functionality to the model is done by providing examples of a user asking the assistant for the 

## Adding a new personality