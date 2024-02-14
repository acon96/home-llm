#!/usr/bin/env python3

import argparse, os, re, json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftConfig, PeftModel
from tqdm import tqdm

CTX_SIZE = 2048

def tokenize(tokenizer, prompt):
    return tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=CTX_SIZE)

def generate(model, tokenizer, prompts):
    inputs = tokenize(tokenizer, prompts)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    text = tokenizer.batch_decode(outputs)
    return text

def main():
    parser = argparse.ArgumentParser(description="Evaluate the function calling for a model")
    parser.add_argument("model")
    parser.add_argument("--dataset_file", default="./data/home_assistant_test.jsonl")
    parser.add_argument("--batch-size", default=8)
    parser.add_argument("--lora", default=False,action='store_const', const=True)

    args = parser.parse_args()
    lora_folder = f"./loras/{args.model}"
    model_folder = f"./models/{args.model}"

    dataset = load_dataset("json", data_files={ "train": args.dataset_file })["train"]

    print(f"Got {len(dataset)} examples to test")

    # filter out examples that are status requests
    if "text" in dataset:
        dataset = dataset.filter(lambda example: "```homeassistant" in example["text"])
    else:
        dataset = dataset.filter(lambda example: "```homeassistant" in example["conversations"][2]["value"])

    service_call_regex = re.compile(r"```homeassistant\n([\S \t\n]*?)```")

    torch.set_default_device("cuda")
    
    if args.lora:
        adapter_config = PeftConfig.from_pretrained(lora_folder)
        base_model_name = adapter_config.base_model_name_or_path
        print(f"Loading lora from {lora_folder} ({base_model_name})...")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        trained_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, padding_side='left')

        trained_model =  PeftModel.from_pretrained(base_model, lora_folder, trust_remote_code=True, torch_dtype=torch.bfloat16)

        output_folder = lora_folder
    else:
        print(f"Loading model from {model_folder}...")
        trained_model = AutoModelForCausalLM.from_pretrained(
            model_folder,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        trained_tokenizer = AutoTokenizer.from_pretrained(model_folder, trust_remote_code=True, padding_side='left')
        output_folder = model_folder

    trained_model.generation_config = GenerationConfig(
        max_new_tokens=128,
        use_cache=True,
        do_sample=True,
        temperature=0.1,
        top_k=40,
        top_p=1.0,
        repetition_penalty=1.15,
        eos_token_id=trained_model.config.eos_token_id,
        pad_token_id=trained_model.config.pad_token_id if trained_model.config.pad_token_id else trained_model.config.eos_token_id,
    )

    split = trained_tokenizer.apply_chat_template(conversation=[{"role": "assistant", "content":  r"%%%%%%%%%%%%%%%%"}], tokenize=False).split( r"%%%%%%%%%%%%%%%%")[0]

    print("Evaluating...")
    batch_size = int(args.batch_size)
    correct_answers = 0
    total_answers = 0
    color_mismatches = 0

    # pre-allocate cuda buffers
    inputs = trained_tokenizer([""] * batch_size, return_tensors="pt", max_length=CTX_SIZE, padding="max_length", truncation=True)
    inputs = {k: v.to(trained_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = trained_model(**inputs)

    failed_examples = []
    with tqdm(total=len(dataset), desc="Accuracy") as pbar:
        for batch_start in range(0, len(dataset), batch_size):
            batch = dataset[batch_start:batch_start + batch_size]
            if "text" in batch:
                prompts = [ example.split(split)[0] + split for example in batch["text"] ]
                expected_responses = [ example.split(split)[1] for example in batch["text"] ]
            else:
                prompts = []
                expected_responses = []
                for example in batch["conversations"]:
                    conversation = [ { "role": x["from"], "content": x["value"] } for x in example if x["from"] != "assistant"]
                    prompts.append(trained_tokenizer.apply_chat_template(
                        conversation=conversation,
                        max_length=CTX_SIZE,
                        truncation=True,
                        tokenize=False,
                        add_generation_prompt=True,
                    ))
                    expected_responses.append([x["value"] for x in example if x["from"] == "assistant"][0])
            output = generate(trained_model, trained_tokenizer, prompts)

            for model_output, expected_response in zip(output, expected_responses):
                response = model_output.replace(trained_tokenizer.pad_token, "").replace(trained_tokenizer.eos_token, "").split(split)[1]

                expected_service_calls = []

                for block in service_call_regex.findall(expected_response.strip()):
                    for line in block.split("\n"):
                        if len(line) == 0:
                            continue
                        expected_service_calls.append(json.loads(line))
                        total_answers = total_answers + 1
                        
                for block in service_call_regex.findall(response.strip()):
                    for line in block.split("\n"):
                        if len(line) == 0:
                            continue
                        try:
                            json_output = json.loads(line)
                        except:
                            failed_examples.append({ "expected": expected_response, "actual": response, "invalid_json": True })
                            continue

                        if json_output in expected_service_calls:
                            expected_service_calls.pop(expected_service_calls.index(json_output))
                            correct_answers = correct_answers + 1
                        elif "rgb_color" in json_output:
                            for sc in expected_service_calls:
                                sc = { **sc }
                                json_output_copy = { **json_output }
                                if not "rgb_color" in sc:
                                    continue
                                del sc["rgb_color"]
                                del json_output_copy["rgb_color"]
                                if sc == json_output_copy:
                                    correct_answers = correct_answers + 1
                                    color_mismatches = color_mismatches + 1
                                else:
                                    failed_examples.append({ "expected": expected_response, "actual": response })
                        else:
                            failed_examples.append({ "expected": expected_response, "actual": response })

            pbar.update(batch_size)
            pbar.set_description(f"Accuracy: {correct_answers/total_answers*100:.2f}% ({correct_answers}/{total_answers})")

    accuracy = correct_answers/total_answers
    print(f"Final Accuracy Rating: {accuracy*100:.2f}%")
    print(f"Color Mismatches: {color_mismatches}")

    with open(os.path.join(output_folder, "eval_results.json"), "w") as f:
        json.dump({
            "possible_answers": total_answers,
            "correct_answers": correct_answers,
            "accuracy": accuracy,
            "color_mismatches": color_mismatches,
            "failed_examples": failed_examples,
        }, f, indent=4)


if __name__ == "__main__":
    main()