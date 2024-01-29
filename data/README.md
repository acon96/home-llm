# Dataset

The dataset is generated from the different CSV "piles". The "piles" contain different chunks of requests that are assembled into a final context that is presented to the LLM. For example, `piles/pile_of_device_names.csv` contains only names of various devices to be used as part of context as well as inserted into `piles/pile_of_templated_actions.csv` and `piles/pile_of_status_requests.csv`. The logic for assembling the final dataset from the piles is contained in [generate_home_assistant_data.py](./generate_home_assistant_data.py).

## Generating the custom dataset

`python3 generate_home_assistant_data.py --train --test --large`

## Merging with other datasets for training

`python3 generate_home_assistant_data.py --merge <dataset>`

Supported datasets right now are: 
- `alpaca`
- `wizardlm70k`

## Potential Other Datasets to Use

### SFT
Alpaca: https://huggingface.co/datasets/yahma/alpaca-cleaned
Alpaca (Translated): https://huggingface.co/datasets/saillab/taco-datasets
WizardLM 200k: https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k
WizardLM 70k: https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k
Huggingface Ultrachat 200k: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
OpenOrca Slim Deduped (363k): https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup

### DPO
Intel Orca DPO Pairs: https://huggingface.co/datasets/Intel/orca_dpo_pairs
Huggingface Ultrachat: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized