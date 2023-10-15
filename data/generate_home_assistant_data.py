import json
import csv
import enum
import random
from dataclasses import dataclass
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
    # TODO: needs refinement
    # "climate": DeviceType(
    #     name="climate",
    #     possible_states=[
    #         (STATE_ON, 0.95),
    #         (STATE_OFF, 0.05)
    #     ],
    #     services=[
    #         "set_temperature", parameters=[
    #             ("temperature", RandomValueType.NUMBER)
    #         ]),
    #         "turn_on",
    #         "turn_off",
    #     ],
    # ),
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

with open("pile_of_device_names.csv") as f:
    reader = csv.DictReader(f)
    pile_of_device_names = list(reader)

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

# generate a random list of devices for the context
def random_device_list(max_devices: int):
    num_devices = random.randint(int(max_devices / 2), max_devices)

    choices = random.choices(pile_of_device_names, k=num_devices)

    device_types = set()
    device_list = []
    for choice in choices:
        try:
            device_name = choice["device_name"]
            device_type = device_name.split(".")[0]

            state = SUPPORTED_DEVICES[device_type].get_random_state()
            device_list.append(f"{device_name} - {state}")
            device_types.add(device_type)
        except:
            print(f"bad device name: {choice}")

    return device_list, list(device_types)

def generate_example(question: str, target_devices: list[str], service_names: list[str]):

    device_list, device_types = random_device_list(max_devices=32)

    # insert our target device somewhere random in the list
    for device in target_devices:
        device_type = device.split(".")[0]
        index = random.randint(0, len(device_list))
        state = SUPPORTED_DEVICES[device_type].get_random_state()

        device_list.insert(index, f"{device} - {state}")

    # gather a list of all available services
    available_services = set()
    for x in device_types:
        available_services = available_services.union(set(SUPPORTED_DEVICES[x].services))

    # generate the list of service calls and answers
    service_calls = []
    answers = []
    for device, service in zip(target_devices, service_names):
        device_type = device.split(".")[0]

        service_calls.append(f"{service}({device})")
        answers.append(random.choice(pile_of_responses[device_type][service]))

    # generate a ton more examples of actions. focus in on other device types besides fans + lights
    # TODO: add examples for ambiguous requests. asking a clarifying question
    # TODO: add examples for requests about devices states without making any changes
    # TODO: better response selection for multiple action situations (make it sound more natural)

    return {
        "states": device_list,
        "available_services": list(available_services),
        "question": question,
        "answers": answers,
        "service_calls": service_calls
    }

def format_example(example):
    sys_prompt = "You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task ask instructed with the information provided only."
    services_block = "Services: " + ", ".join(example["available_services"])
    states_block = "States:\n" + "\n".join(example["states"])
    answers = " ".join(example["answers"])
    code_block = "```homeassistant\n" + "\n".join(example["service_calls"]) + "\n```done\n"
    return "\n".join([sys_prompt, services_block, states_block, example["question"], answers, code_block])

def main():
    random.seed(42)

    examples = []
    for action in tqdm(pile_of_device_actions):
        question = action["english_phrase"]
        devices = action["device_name"].split("|")
        services = action["service_name"].split("|")

        # 3 examples per question
        examples.append({ "text": format_example(generate_example(question, devices, services)) })
        examples.append({ "text": format_example(generate_example(question, devices, services)) })
        examples.append({ "text": format_example(generate_example(question, devices, services)) })

    with open("home_assistant_examples.json", "w") as f:
        json.dump(examples, f, indent=4)

    print("Done!")

if __name__ == "__main__":
    main()