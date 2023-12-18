import random
from datasets import load_dataset, concatenate_datasets
from generate_home_assistant_data import format_example, random_device_list, SUPPORTED_DEVICES

alpaca_dataset = load_dataset("yahma/alpaca-cleaned")["train"].train_test_split(test_size=0.1)
home_assistant_dataset = load_dataset("json", data_files={  "train": "home_assistant_train.json", "test": "home_assistant_test.json" })

random.seed(42)

def format_alpaca(example):
    question = example["instruction"]
    if example["input"]:
        question = question = "\n" + example["input"]

    answer = example["output"]

    device_list, device_types = random_device_list(max_devices=32, avoid_device_names=[])

    available_services = []
    for x in device_types:
        available_services.extend([ f"{x}.{y}" for y in SUPPORTED_DEVICES[x].services ])

    text = format_example(example={
        "states": device_list,
        "available_services": list(available_services),
        "question": question,
        "answers": [ answer ],
        "service_calls": []
    })

    result = {
        "text": text
    }

    return result

alpaca_dataset = alpaca_dataset.map(format_alpaca).remove_columns(["input", "output", "instruction"])

combined_dataset_train = concatenate_datasets([home_assistant_dataset["train"], alpaca_dataset["train"]]).shuffle(seed=42)
combined_dataset_test = concatenate_datasets([home_assistant_dataset["test"], alpaca_dataset["test"]]).shuffle(seed=42)

combined_dataset_train.to_json("home_assistant_alpaca_merged_train.json")
combined_dataset_test.to_json("home_assistant_alpaca_merged_test.json")