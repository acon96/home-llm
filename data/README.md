# Dataset

The dataset is generated from the different CSV "piles". The "piles" contain different chunks of requests that are assembled into a final context that is presented to the LLM. For example, `piles/pile_of_device_names.csv` contains only names of various devices to be used as part of context as well as inserted into `piles/pile_of_templated_actions.csv` and `piles/pile_of_status_requests.csv`. The logic for assembling the final dataset from the piles is contained in [generate_home_assistant_data.py](./generate_home_assistant_data.py).

## Generating the custom dataset

`python3 generate_home_assistant_data.py --train --test`

## Merging with Alpaca for training

`python3 generate_home_assistant_data.py --merge-alpaca`