import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task ask instructed with the information provided only."

def tokenize(tokenizer, prompt):
    return tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

def generate(model, tokenizer, prompt):
    inputs = tokenize(tokenizer, prompt)
    outputs = model.generate(**inputs, max_length=512)
    text = tokenizer.batch_decode(outputs)[0]
    return text

def format_example(example):
    sys_prompt = SYSTEM_PROMPT
    services_block = "Services: " + ", ".join(sorted(example["available_services"]))
    states_block = "Devices:\n" + "\n".join(example["states"])
    question = "Request:\n" + example["question"]
    response_start = "Response:\n"

    return "\n".join([sys_prompt, services_block, states_block, question, response_start])

def main():
    request = "turn on the office lights"
    model_folder = "./model/home-llm-rev7.1"

    example = {
        "states": [
            "light.kitchen_sink = on",
            "light.kitchen_lamp = on",
            "light.office_desk_lamp = on",
            "light.family_room_overhead = on",
            "fan.family_room = off",
            "lock.fron_door = locked"
        ],
        "available_services": ["turn_on", "turn_off", "toggle", "lock", "unlock" ],
        "question": request,
    }

    prompt = format_example(example)

    torch.set_default_device("cuda")
    trained_model = AutoModelForCausalLM.from_pretrained(model_folder, trust_remote_code=True)
    trained_tokenizer = AutoTokenizer.from_pretrained(model_folder, trust_remote_code=True)

    output = generate(trained_model, trained_tokenizer, prompt)
    print(output)


if __name__ == "__main__":
    main()