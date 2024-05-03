#!/usr/bin/env python3

import argparse, os, re, json, csv, random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftConfig, PeftModel
from tqdm import tqdm

torch.set_default_device("cuda")

CTX_SIZE = 2048
TRUST_REMOTE_CODE = False


"""
python3 evaluate.py stablehome-1_6b-rev3 --batch-size 8 --all-checkpoints
python3 evaluate.py tinyhome-rev1 --batch-size 12 --all-checkpoints
python3 evaluate.py stablehome-3b-rev6 --batch-size 4 --lora --overwrite
"""

service_call_regex = re.compile(r"```homeassistant\n([\S \t\n]*?)```")
json_regex = re.compile(r"({[\S \t]*?})")
service_names_regex = re.compile(r"\b\w+\.\w+\([^)]*\)")
entity_ids_regex = re.compile(r"\b\w+\.\w+(?=\s'|\s=)")

try:
    with open("custom_components/llama_conversation/in_context_examples.csv", encoding="utf-8-sig") as f:
        in_context_examples = list(csv.DictReader(f))
except:
    in_context_examples = []

def icl_example_generator(num_examples, entity_names, service_names):
    entity_domains = set([x.split(".")[0] for x in entity_names])
    entity_names = entity_names[:]
    
    # filter out examples for disabled services
    selected_in_context_examples = []
    for x in in_context_examples:
        if x["service"] in service_names and x["service"].split(".")[0] in entity_domains:
            selected_in_context_examples.append(x)

    # if we filtered everything then just sample randomly
    if len(selected_in_context_examples) == 0:
        selected_in_context_examples = in_context_examples[:]

    random.shuffle(selected_in_context_examples)
    random.shuffle(entity_names)

    num_examples_to_generate = min(num_examples, len(selected_in_context_examples))
    if num_examples_to_generate < num_examples:
        print(f"Attempted to generate {num_examples} ICL examples for conversation, but only {len(selected_in_context_examples)} are available!")
    
    results = []
    while len(results) < num_examples_to_generate:
        if len(selected_in_context_examples) == 0:
            break
        
        chosen_example = selected_in_context_examples.pop()
        chosen_service = chosen_example["service"]
        potential_devices = [ x for x in entity_names if x.split(".")[0] == chosen_service.split(".")[0] ]

        if len(potential_devices) == 0:
            continue
        else:
            example = {
                "to_say": chosen_example["response"],
                "service": chosen_service,
                "target_device": potential_devices[0],
            }
            results.insert(0, json.dumps(example))
    
    return "\n".join(results)

def tokenize(tokenizer, prompt):
    return tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=CTX_SIZE)

def generate(model, tokenizer, prompts):
    inputs = tokenize(tokenizer, prompts)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    text = tokenizer.batch_decode(outputs)
    return text

def evaluate(output_folder, trained_model, trained_tokenizer, dataset, batch_size, use_icl):
    split = trained_tokenizer.apply_chat_template(conversation=[{"role": "assistant", "content":  r"%%%%%%%%%%%%%%%%"}], tokenize=False).split( r"%%%%%%%%%%%%%%%%")[0].replace(trained_tokenizer.bos_token, "")

    print("Evaluating...")
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

                    if use_icl:
                        new_conversation = []
                        for turn in conversation:
                            if turn["role"] == "system":
                                entity_names = entity_ids_regex.findall(turn["content"])
                                service_names = [ x.split("(")[0] for x in service_names_regex.findall(turn["content"]) ]
                                icl_examples = icl_example_generator(5, entity_names, service_names)
                                turn["content"] = turn["content"] + "Respond to the following user instruction by responding in the same format as the following examples:\n" + icl_examples
                            new_conversation.append(turn)
                        conversation = new_conversation
                    
                    prompts.append(trained_tokenizer.apply_chat_template(
                        conversation=conversation,
                        max_length=CTX_SIZE,
                        truncation=True,
                        tokenize=False,
                        add_generation_prompt=True,
                    ))

                    if use_icl:
                        response = [x["value"] for x in example if x["from"] == "assistant"][0]
                        expected_calls = service_call_regex.findall(response)
                        to_say = service_call_regex.sub("", response)
                        expected_responses.append(expected_calls[0])
                    else:
                        expected_responses.append([x["value"] for x in example if x["from"] == "assistant"][0])
            output = generate(trained_model, trained_tokenizer, prompts)

            for model_output, expected_response in zip(output, expected_responses):
                response = model_output.replace(trained_tokenizer.pad_token, "").replace(trained_tokenizer.eos_token, "").split(split)[1]

                expected_service_calls = []

                if use_icl:
                    regex_to_use = json_regex
                else:
                    regex_to_use = service_call_regex

                for block in regex_to_use.findall(expected_response.strip()):
                    for line in block.split("\n"):
                        if len(line) == 0:
                            continue
                        expected_service_calls.append(json.loads(line))
                        total_answers = total_answers + 1
                
                found_responses = regex_to_use.findall(response.strip())

                if len(expected_service_calls) == 0:
                    total_answers = total_answers + 1
                    if len(found_responses) == 0:
                        correct_answers = correct_answers + 1
                        continue
                    else:
                        failed_examples.append({ "expected": expected_response, "actual": response, "extra_response": True })
                        continue
                
                if len(found_responses) == 0:
                    failed_examples.append({ "expected": expected_response, "actual": response, "no_response_found": True })
                    continue

                for block in found_responses:
                    for line in block.split("\n"):
                        if len(line) == 0:
                            continue
                        try:
                            json_output = json.loads(line)
                        except:
                            failed_examples.append({ "expected": expected_response, "actual": response, "invalid_json": True })
                            continue

                        if use_icl:
                            json_output.pop("to_say")
                            
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

def load_model(model_name, is_lora, is_hf, load_in_8bit, checkpoint_name):
    lora_folder = f"./loras/{model_name}/"
    model_folder = f"./models/{model_name}/"
    
    # tokenizer isn't saved into checkpoint folders
    tokenizer_folder = model_folder

    if checkpoint_name:
        lora_folder = lora_folder + f"{checkpoint_name}/"
        model_folder = model_folder + f"{checkpoint_name}/"

    if is_hf:
        print(f"Loading model {model_name}...")
        trained_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=TRUST_REMOTE_CODE,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
        )

        trained_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=TRUST_REMOTE_CODE,
            padding_side='left',
        )
    elif is_lora:
        adapter_config = PeftConfig.from_pretrained(lora_folder)
        base_model_name = adapter_config.base_model_name_or_path
        print(f"Loading lora from {lora_folder} ({base_model_name})...")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=TRUST_REMOTE_CODE,
            torch_dtype=torch.bfloat16,
        )
        trained_model =  PeftModel.from_pretrained(
            base_model,
            lora_folder,
            trust_remote_code=TRUST_REMOTE_CODE,
            torch_dtype=torch.bfloat16,
        )

        trained_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=TRUST_REMOTE_CODE,
            padding_side='left',
        )
    else:
        print(f"Loading model from {model_folder}...")
        trained_model = AutoModelForCausalLM.from_pretrained(
            model_folder,
            trust_remote_code=TRUST_REMOTE_CODE,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
        )

        trained_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_folder,
            trust_remote_code=TRUST_REMOTE_CODE,
            padding_side='left',
        )

    if not trained_tokenizer.pad_token:
        trained_tokenizer.pad_token = trained_tokenizer.eos_token

    trained_model.generation_config = GenerationConfig(
        max_new_tokens=128,
        use_cache=True,
        do_sample=True,
        temperature=0.1,
        top_k=40,
        top_p=1.0,
        repetition_penalty=1.15,
        # eos_token_id=trained_model.config.eos_token_id,
        eos_token_id=128009,
        pad_token_id=trained_model.config.pad_token_id if trained_model.config.pad_token_id else trained_model.config.eos_token_id,
    )

    return trained_model, trained_tokenizer

def main():
    global in_context_examples
    parser = argparse.ArgumentParser(description="Evaluate the function calling for a model")
    parser.add_argument("model")
    parser.add_argument("--dataset-file", default="./data/home_assistant_test.jsonl")
    parser.add_argument("--batch-size", default=8)
    parser.add_argument("--lora", default=False, action='store_const', const=True)
    parser.add_argument("--all-checkpoints", default=False, action='store_const', const=True)
    parser.add_argument("--overwrite", default=False, action='store_const', const=True)
    parser.add_argument("--hf", default=False, action='store_const', const=True)
    parser.add_argument("--load-in-8bit", default=False, action='store_const', const=True)

    args = parser.parse_args()
    batch_size = int(args.batch_size)

    dataset = load_dataset("json", data_files={ "train": args.dataset_file })["train"]

    print(f"Got {len(dataset)} examples to test")

    if args.hf:
        output_folder = "./"
        trained_model, trained_tokenizer = load_model(args.model, args.lora, True, args.load_in_8bit, None)
        evaluate(output_folder, trained_model, trained_tokenizer, dataset, batch_size, True)

    else:
        model_folder = f"./loras/{args.model}/" if args.lora else f"./models/{args.model}/"

        if not os.path.isdir(model_folder):
            print(f"Model Not Found: {args.model}")
            return

        
        if not args.all_checkpoints:
            checkpoints = [None]
        else:
            checkpoints = [x for x in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, x)) and "checkpoint" in x]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
            checkpoints.append(None)

            print(f"Found {len(checkpoints) - 1} checkpoints to test (plus the final model)")

        for ckpt in checkpoints:
            if ckpt:
                output_folder = os.path.join(model_folder, ckpt)
            else:
                output_folder = model_folder
            
            output_filename = os.path.join(output_folder, "eval_results.json")
            if os.path.exists(output_filename):
                if not args.overwrite:
                    print(f"Evaluation already exists for {output_folder}. Skipping...")
                    continue

            trained_model, trained_tokenizer = load_model(args.model, args.lora, ckpt, False)
            evaluate(output_folder, trained_model, trained_tokenizer, dataset, batch_size, False)


if __name__ == "__main__":
    main()