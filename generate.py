#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only."
CTX_SIZE = 512

def tokenize(tokenizer, prompt):
    return tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=CTX_SIZE)

def generate(model, tokenizer, prompt):
    eos_token_id = tokenizer(tokenizer.eos_token)["input_ids"][0]

    inputs = tokenize(tokenizer, prompt)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.15,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
        )
    text = tokenizer.batch_decode(outputs)
    return text

def format_example(example):
    sys_prompt = SYSTEM_PROMPT
    services_block = "Services: " + ", ".join(sorted(example["available_tools"]))
    states_block = "Devices:\n" + "\n".join(example["states"])
    question = "Request:\n" + example["question"]
    response_start = "Response:\n"

    return "\n".join([sys_prompt, services_block, states_block, question, response_start])

def main():
    request = "turn on the office lights"
    model_folder = "./models/home-llm-rev9"
    num_examples = 10

    example = {
        "states": [
            "light.kitchen_sink = on",
            "light.kitchen_lamp = on",
            "light.office_desk_lamp = on",
            "light.family_room_overhead = on",
            "fan.family_room = off",
            "lock.front_door = locked"
        ],
        "available_tools": ["turn_on", "turn_off", "toggle", "lock", "unlock" ],
        "question": request,
    }

    prompt = format_example(example)

    torch.set_default_device("cuda")
    print(f"Loading model from {model_folder}...")
    trained_model = AutoModelForCausalLM.from_pretrained(model_folder, trust_remote_code=True, torch_dtype=torch.bfloat16)
    trained_tokenizer = AutoTokenizer.from_pretrained(model_folder, trust_remote_code=True)

    print("Generating output...")
    output = generate(trained_model, trained_tokenizer, [ prompt for x in range(num_examples) ])

    for text in output:
        print("--------------------------------------------------")
        print(text.replace(trained_tokenizer.eos_token, ""))


if __name__ == "__main__":
    main()