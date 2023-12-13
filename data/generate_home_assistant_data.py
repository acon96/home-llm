import json
import csv
import enum
import random
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Final, Any
from tqdm import tqdm

# #### STATES ####
STATE_ON: Final = "on"
STATE_OFF: Final = "off"
STATE_UNKNOWN: Final = "unknown"
STATE_OPEN: Final = "open"
STATE_OPENING: Final = "opening"
STATE_CLOSED: Final = "closed"
STATE_CLOSING: Final = "closing"
STATE_BUFFERING: Final = "buffering"
STATE_PLAYING: Final = "playing"
STATE_PAUSED: Final = "paused"
STATE_IDLE: Final = "idle"
STATE_STANDBY: Final = "standby"
STATE_LOCKED: Final = "locked"
STATE_UNLOCKED: Final = "unlocked"
STATE_LOCKING: Final = "locking"
STATE_UNLOCKING: Final = "unlocking"
STATE_JAMMED: Final = "jammed"
STATE_UNAVAILABLE: Final = "unavailable"
STATE_OK: Final = "ok"
STATE_PROBLEM: Final = "problem"

# class RandomValueType(enum.StrEnum):
#     NUMBER = enum.auto()
#     PERCENT = enum.auto()
#     BOOL = enum.auto()

# def get_random_value(type: RandomValueType):
#     match type:
#         case RandomValueType.NUMBER:
#             return random.randint(0, 1000)
#         case RandomValueType.PERCENT:
#             return random.random()
#         case RandomValueType.BOOL:
#             return random.random() > 0.5

@dataclass
class DeviceType:
    name: str
    possible_states: list[(str, float)]
    services: list[str]

    def all_states(self):
        return [ x[0] for x in self.possible_states ]

    def get_random_state(self):
        states = [ x[0] for x in self.possible_states ]
        weights = [ x[1] for x in self.possible_states ]
        return random.choices(states, weights=weights, k=1)[0]

SUPPORTED_DEVICES = {
    "light": DeviceType(
        name="light",
        possible_states=[
            (STATE_ON, 0.5),
            (STATE_OFF, 0.5)
        ],
        services=[
            "turn_on",
            "turn_off",
            "toggle"
        ],
    ),
    "fan": DeviceType(
        name="fan",
        possible_states=[
            (STATE_ON, 0.5),
            (STATE_OFF, 0.5)
        ],
        services=[
            "turn_on",
            "turn_off",
            "toggle",
            "increase_speed",
            "decrease_speed",
        ],
    ),
    "garage_door": DeviceType(
        name="garage_door",
        possible_states=[
            (STATE_OPEN, 0.49),
            (STATE_CLOSED, 0.49),
            (STATE_OPENING, 0.01),
            (STATE_CLOSING, 0.01)
        ],
        services=[
            "open_cover",
            "close_cover",
            "stop_cover",
            "toggle",
        ],
    ),
    "blinds": DeviceType(
        name="blinds",
        possible_states=[
            (STATE_OPEN, 0.49),
            (STATE_CLOSED, 0.49),
            (STATE_OPENING, 0.01),
            (STATE_CLOSING, 0.01)
        ],
        services=[
            "open_cover",
            "close_cover",
            "stop_cover",
            "toggle",
        ],
    ),
    "lock": DeviceType(
        name="lock",
        possible_states=[
            (STATE_LOCKED, 0.5),
            (STATE_UNLOCKED, 0.5),
        ],
        services=[
            "lock",
            "unlock",
        ],
    ),
    "media_player": DeviceType(
        name="media_player",
        possible_states=[
            (STATE_ON, 0.15),
            (STATE_OFF, 0.3),
            (STATE_IDLE, 0.1),
            (STATE_PLAYING, 0.2),
            (STATE_PAUSED, 0.15),
            (STATE_STANDBY, 0.05),
            (STATE_BUFFERING, 0.05),
        ],
        services=[
            "turn_on",
            "turn_off",
            "toggle",
            "volume_up",
            "volume_down",
            "volume_mute",
            "media_play_pause",
            "media_play",
            "media_pause",
            "media_stop",
            "media_next_track",
            "media_previous_track"
        ],
    ),
}

stacks_of_device_names = { x: [] for x in SUPPORTED_DEVICES.keys() }
with open("pile_of_device_names.csv") as f:
    reader = csv.DictReader(f)
    pile_of_device_names = list(reader)
    for device_dict in pile_of_device_names:
        try:
            device_type = device_dict["device_name"].split(".")[0]
            stacks_of_device_names[device_type].append(device_dict)
        except KeyError as ex:
            print(ex)

with open("pile_of_templated_actions.csv") as f:
    reader = csv.DictReader(f)
    pile_of_templated_actions = list(reader)

with open("pile_of_device_actions.csv") as f:
    reader = csv.DictReader(f)
    pile_of_device_actions = list(reader)

with open("pile_of_responses.csv") as f:
    reader = csv.DictReader(f)
    raw_pile_of_responses = list(reader)

    pile_of_responses = {}
    for raw in raw_pile_of_responses:
        if raw["device_type"] not in pile_of_responses:
            pile_of_responses[raw["device_type"]] = {}    
        pile_of_responses[raw["device_type"]][raw["service"]] = [ raw["response_1"], raw["response_2"], raw["response_3"] ]

with open("pile_of_status_requests.csv") as f:
    reader = csv.DictReader(f)
    pile_of_status_requests = list(reader)

def format_device_line(*, device_name: str, friendly_name: str, state: str):
    return (f"{device_name} '{friendly_name}' = {state}")

# generate a random list of devices for the context
def random_device_list(max_devices: int, avoid_device_names: list[str]):
    num_devices = random.randint(2, max_devices)

    local_device_names = { k: v[:] for k,v in stacks_of_device_names.items() }

    for avoid_device in avoid_device_names:
        avoid_type = avoid_device.split(".")[0]

        filtered_possible_devices = []
        for possible_device in local_device_names[avoid_type]:
            similarity_ratio = SequenceMatcher(None, avoid_device, possible_device["device_name"].split(".")[1]).ratio()

            if similarity_ratio < 0.4:
                filtered_possible_devices.append(possible_device)
        local_device_names[avoid_type] = filtered_possible_devices

    possible_choices = []
    for device_type in local_device_names.keys():
        possible_choices.extend(local_device_names[device_type])
    

    device_types = set()
    device_list = []
    device_lines = []
    while len(device_list) < num_devices:
        choice = random.choice(possible_choices)
        if choice["device_name"] in device_list:
            continue

        try:
            device_name = choice["device_name"]
            device_type = device_name.split(".")[0]
            friendly_name = choice["description"]

            state = SUPPORTED_DEVICES[device_type].get_random_state()
            device_lines.append(format_device_line(
                device_name=device_name,
                friendly_name=friendly_name,
                state=state
            ))
            device_list.append(device_name)
            device_types.add(device_type)
        except:
            print(f"bad device name: {choice}")

    return device_lines, list(device_types)

def generate_static_example(action: dict, max_devices: int = 32):
    question = action["english_phrase"]
    target_device = action["device_name"]
    device_type = target_device.split(".")[0]
    service_name = f"{device_type}.{action['service_name']}"
    friendly_name = target_device.split(".")[1].replace("_", " ")

    device_list, device_types = random_device_list(max_devices=max_devices, avoid_device_names=[target_device])

    # insert our target device somewhere random in the list
    
    index = random.randint(0, len(device_list))
    state = SUPPORTED_DEVICES[device_type].get_random_state()

    device_list.insert(index, format_device_line(
        device_name=target_device,
        friendly_name=friendly_name,
        state=state
    ))

    # gather a list of all available services
    available_services = []
    for x in device_types:
        available_services.extend([ f"{x}.{y}" for y in SUPPORTED_DEVICES[x].services ])

    return {
        "states": device_list,
        "available_services": list(available_services),
        "question": question.lower(),
        "answers": [ random.choice(pile_of_responses[device_type][action["service_name"]]).lower() ],
        "service_calls": [ f"{service_name}({target_device})" ]
    }

def generate_templated_example(template: dict, max_devices: int = 32):
    template_device_types: list[str] = template["device_type"].split("|")
    service_names: list[str] = [ f"{x}.{y}" for x, y in zip(template_device_types, template["service"].split("|")) ]
    question_template: str = template["english_phrase"]
    answer_template: str = template["assistant_response"]

    # choose a random device for this template
    chosen_devices = []
    for device_type in template_device_types:
        device_dict = random.choice(stacks_of_device_names[device_type])
        device_dict["type"] = device_type
        chosen_devices.append(device_dict)

    device_list, device_types = random_device_list(max_devices=max_devices, avoid_device_names=[d["device_name"] for d in chosen_devices])

    # insert our target device somewhere random in the list
    for device_dict in chosen_devices:
        index = random.randint(0, len(device_list))
        state = SUPPORTED_DEVICES[device_dict["type"]].get_random_state()
        device_name = device_dict["device_name"]
        friendly_name = device_dict["description"]

        device_list.insert(index, format_device_line(
            device_name=device_name,
            friendly_name=friendly_name,
            state=state
        ))

    # gather a list of all available services
    available_services = []
    for x in device_types:
        available_services.extend([ f"{x}.{y}" for y in SUPPORTED_DEVICES[x].services ])

    available_services.extend(service_names)

    # generate the question
    if len(template_device_types) == 1:
        question = question_template.replace("<device_name>", chosen_devices[0]["description"])
        answer = answer_template.replace("<device_name>", chosen_devices[0]["description"])
    else:
        question = question_template
        answer = answer_template
        for i in range(len(template_device_types)):
            question = question.replace(f"<device_name{(i + 1)}>", chosen_devices[i]["description"])
            answer = answer.replace(f"<device_name{(i + 1)}>", chosen_devices[i]["description"])


    # generate the list of service calls and answers
    service_calls = []
    for device_dict, service in zip(chosen_devices, service_names):
        service_calls.append(f"{service}({device_dict['device_name']})")

    return {
        "states": device_list,
        "available_services": list(available_services),
        "question": question.lower(),
        "answers": [ answer.lower() ],
        "service_calls": service_calls
    }

def generate_status_request(template: dict, max_devices: int = 32):
    device_type: str = template["device_type"]
    state_name: str = template["state"]
    question_template: str = template["english_phrase"]
    answer_template: str = template["assistant_response"]

    # choose a random device for this template
    chosen_device = random.choice(stacks_of_device_names[device_type])

    # build a random list of devices
    device_list, device_types = random_device_list(max_devices=max_devices, avoid_device_names=[ chosen_device["device_name"] ])

    # insert our target device somewhere random in the list
    index = random.randint(0, len(device_list))
    device_list.insert(index, f"{chosen_device['device_name']} = {state_name}")

    # gather a list of all available services
    available_services = []
    for x in device_types:
        available_services.extend([ f"{x}.{y}" for y in SUPPORTED_DEVICES[x].services ])

    # generate the question
    question = question_template.replace("<device_name>", chosen_device["description"])
    answer = answer_template.replace("<device_name>", chosen_device["description"])

    return {
        "states": device_list,
        "available_services": list(available_services),
        "question": question.lower(),
        "answers": [ answer.lower() ],
        "service_calls": []
    }

def format_example(example):
    sys_prompt = "You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed or answer the following question with the information provided only."
    services_block = "Services: " + ", ".join(sorted(example["available_services"]))
    states_block = "Devices:\n" + "\n".join(example["states"])
    question = "Request:\n" + example["question"]
    answers = "Response:\n" + " ".join(example["answers"])

    system_block = "\n".join(["<|im_start|>system " + sys_prompt, services_block, states_block, "<|im_end|>" ])
    user_block = "<|im_start|>user " + question + "<|im_end|>"

    assistant_block = "<|im_start|>assistant " + answers
    if len(example["service_calls"]) > 0:
        # TODO: make function calling JSON based
        code_block = "```homeassistant\n" + "\n".join(example["service_calls"]) + "\n```"
        assistant_block = assistant_block + code_block
    assistant_block = assistant_block + "<|im_end|>"
        
    example_lines = [system_block, user_block, assistant_block]
    result = "\n".join(example_lines) + "\n"
    if "<device_name" in result:
        print("bad templating")

    # replace aliases with their actual values
    result = result.replace("blinds.", "cover.")
    result = result.replace("garage_door.", "cover.")
    return result


def generate_example_file(filename: str, seed: int, *, static_factor: int, template_factor: int, status_request_factor: int):
    random.seed(seed)

    print("Generating...")

    def run_factor_times(func, examples, data, factor):
        if factor >= 1:
            for i in range(factor):
                examples.append({ "text": format_example(func(data)) })
        else:
            if random.random() < factor:
                examples.append({ "text": format_example(func(data)) })
    
    generated_examples = []
    for action in tqdm(pile_of_device_actions):
        run_factor_times(generate_static_example, generated_examples, action, static_factor)

    for templated_action in tqdm(pile_of_templated_actions):
        run_factor_times(generate_templated_example, generated_examples, templated_action, template_factor)

    for status_request in tqdm(pile_of_status_requests):
        run_factor_times(generate_status_request, generated_examples, status_request, status_request_factor)

    print(f"Generated {len(generated_examples)} examples. Saving...")
    with open(f"{filename}.json", "w") as f:
        json.dump(generated_examples, f, indent=4)

    print("Done!")

# TODO: add examples for ambiguous requests. asking a clarifying question
# TODO: make more randomized names for devices (random words or people's names)
# TODO: answer questions about more than one thing in the state list at once
# TODO: add examples for rooms/groups of devices. i.e. "turn off all the lights in the kitchen"
# TODO: expose home assistant attributes in the context
# TODO: add thermostat + switch device types
def main():
    # generate_example_file("sample", 42, static_factor=1, template_factor=1, status_request_factor=1)
    # exit()
    generate_example_file("home_assistant_train", 42, static_factor=3, template_factor=20, status_request_factor=15)
    generate_example_file("home_assistant_test", 12345, static_factor=0.25, template_factor=3, status_request_factor=2)

if __name__ == "__main__":
    main()