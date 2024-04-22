import argparse
import json
import csv
import pandas
import numpy as np
import random
import re
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from datasets import load_dataset, concatenate_datasets
from difflib import SequenceMatcher
from typing import Final, Any, Callable, Optional
from tqdm import tqdm
import webcolors

# #### STATES ####
STATE_ON: Final = "on"
STATE_OFF: Final = "off"
STATE_ACTIVE: Final = "active"
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
STATE_CLEANING: Final = "cleaning"
STATE_DOCKED: Final = "docked"
STATE_RETURNING: Final = "returning"

# define piles for global access
pile_of_durations = None
pile_of_media_names = None
pile_of_todo_items = None
stacks_of_device_names = None
pile_of_templated_actions = None
pile_of_specific_actions = None
pile_of_responses = None
pile_of_status_requests = None
pile_of_system_prompts = None
pile_of_hallucinated_service_names = None
and_word = None

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def generate_random_date():
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2030, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 24 * 60 * 60)
    random_date_time = start_date + timedelta(days=random_days, seconds=random_seconds)
    return random_date_time

var_pattern = re.compile("<(.*?)>")
def get_included_vars(response: str):
    result = []
    for var in var_pattern.findall(response):
        if var == "device_name":
            continue
        result.append(var)

    return ",".join(sorted(result))

@dataclass
class DeviceType:
    name: str
    possible_states: list[(str, float)]
    services: dict[str, list]
    random_parameter_generator: Optional[dict[str, Callable]] = None

    def get_all_services(self, extra_exposed_attributes):
        result = []
        for service in self.services.keys():
            args = set(extra_exposed_attributes).intersection(self.services[service])
            result.append(f"{self.name}.{service}({','.join(args)})")
        return result
    
    def get_random_parameter(self, param_name):
        return self.random_parameter_generator[param_name]()

    def get_random_state(self, extra_exposed_attributes=[]):
        states = [ x[0] for x in self.possible_states ]
        weights = [ x[1] for x in self.possible_states ]
        return random.choices(states, weights=weights, k=1)[0]
    
class LightDeviceType(DeviceType):
    def __init__(self):
        super().__init__("light",
            possible_states=[
                (STATE_ON, 0.5),
                (STATE_OFF, 0.5)
            ],
            services={
                "turn_on": [ "rgb_color", "brightness" ],
                "turn_off": [],
                "toggle": []
            },
            random_parameter_generator={
                "rgb_color": lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                "brightness": lambda: random.randint(0, 100),
            }
        )

    def get_random_state(self, extra_exposed_attributes=[]):
        state = super().get_random_state(extra_exposed_attributes=extra_exposed_attributes)

        if random.random() < 0.5 and "rgb_color" in extra_exposed_attributes:
            random_rgb = self.get_random_parameter("rgb_color")
            state = state + ";" + closest_color(random_rgb) + " " + str(random_rgb)

        if random.random() < 0.7 and "brightness" in extra_exposed_attributes:
            state = state + ";" + str(self.get_random_parameter("brightness")) + "%"

        return state
    
class ClimateDeviceType(DeviceType):
    def __init__(self):
        super().__init__("climate", [], {
            "turn_on": [],
            "turn_off": [],
            "toggle": [],
            "set_temperature": ["temperature"],
            "set_humidity": ["humidity"],
            "set_fan_mode": ["fan_mode"],
            "set_hvac_mode": ["hvac_mode"],
            "set_preset_mode": ["preset_mode"]
        },
        random_parameter_generator={
            "fan_mode": lambda: random.choice(["On Low", "On High", "Auto Low", "Auto High", "Off"]),
            "temp_f": lambda: random.randint(60, 80),
            "temp_c": lambda: random.randint(15, 25),
            "humidity": lambda: random.randint(10, 90),
            "preset_mode": lambda: random.choice(["home", "eco", "away", "auto"]),
            "hvac_mode": lambda: random.choice(["heat", "cool", "heat_cool", "off", "auto", "fan_only"]),
        })

    def get_random_state(self, extra_exposed_attributes=[]):
        """state;fan_mode;temperature;humidity"""
        state = self.get_random_parameter("hvac_mode")

        if "fan_mode" in extra_exposed_attributes:
            state = state  + ";" + self.get_random_parameter("fan_mode")
        if "temperature" in extra_exposed_attributes:
            if random.random() > 0.5:
                state = state + ";" + str(self.get_random_parameter("temp_f")) + "F" 
            else:
                state = state + ";" + str(self.get_random_parameter("temp_c")) + "C"
        if "humidity" in extra_exposed_attributes:
            state = state + ";" + str(self.get_random_parameter("humidity")) + "%"

        if random.random() < 0.8 and "preset_mode" in extra_exposed_attributes:
            # if it is not "on a preset" then don't add the mode
            state = state + ";" + self.get_random_parameter("preset_mode")

        return state

class MediaPlayerDeviceType(DeviceType):
    def __init__(self):
        super().__init__("media_player", [
            (STATE_ON, 0.15),
            (STATE_OFF, 0.54),
            (STATE_IDLE, 0.1),
            (STATE_PLAYING, 0.1),
            (STATE_PAUSED, 0.05),
            (STATE_STANDBY, 0.05),
            (STATE_BUFFERING, 0.01),
        ], {
            "turn_on": [],
            "turn_off": [],
            "toggle": [],
            "volume_up": [],
            "volume_down": [],
            "volume_mute": [],
            "media_play_pause": [],
            "media_play": [],
            "media_pause": [],
            "media_stop": [],
            "media_next_track": [],
            "media_previous_track": []
        },
        random_parameter_generator={
            "media": lambda: random.choice(pile_of_media_names),
            "volume": lambda: round(random.random(), 2),
        })

    def get_random_state(self, extra_exposed_attributes=[]):
        state = super().get_random_state(extra_exposed_attributes=extra_exposed_attributes)

        if "media_title" in extra_exposed_attributes and state in [STATE_PLAYING, STATE_PAUSED, STATE_BUFFERING, STATE_ON]:
            state = state + ";" + self.get_random_parameter("media")

        if "volume_level" in extra_exposed_attributes and state != STATE_OFF:
            state = state + ";vol=" + str(self.get_random_parameter("volume"))
        return state

SUPPORTED_DEVICES = {
    "light": LightDeviceType(),
    "switch": DeviceType(
        name="switch",
        possible_states=[
            (STATE_ON, 0.5),
            (STATE_OFF, 0.5)
        ],
        services={
            "turn_on": [],
            "turn_off": [],
            "toggle": []
        },
    ),
    "fan": DeviceType(
        name="fan",
        possible_states=[
            (STATE_ON, 0.5),
            (STATE_OFF, 0.5)
        ],
        services={
            "turn_on": [],
            "turn_off": [],
            "toggle": [],
            "increase_speed": [],
            "decrease_speed": [],
        },
    ),
    "garage_door": DeviceType(
        name="garage_door",
        possible_states=[
            (STATE_OPEN, 0.49),
            (STATE_CLOSED, 0.49),
            (STATE_OPENING, 0.01),
            (STATE_CLOSING, 0.01)
        ],
        services={
            "open_cover": [],
            "close_cover": [],
            "stop_cover": [],
            "toggle": [],
        },
    ),
    "blinds": DeviceType(
        name="blinds",
        possible_states=[
            (STATE_OPEN, 0.49),
            (STATE_CLOSED, 0.49),
            (STATE_OPENING, 0.01),
            (STATE_CLOSING, 0.01)
        ],
        services={
            "open_cover": [],
            "close_cover": [],
            "stop_cover": [],
            "toggle": [],
        },
    ),
    "lock": DeviceType(
        name="lock",
        possible_states=[
            (STATE_LOCKED, 0.5),
            (STATE_UNLOCKED, 0.5),
        ],
        services={
            "lock": [],
            "unlock": [],
        },
    ),
    "media_player": MediaPlayerDeviceType(),
    "climate": ClimateDeviceType(),
    "vacuum": DeviceType(
        name="vacuum",
        possible_states=[
            (STATE_CLEANING, 0.2),
            (STATE_DOCKED, 0.6),
            (STATE_RETURNING, 0.1),
            (STATE_IDLE, 0.05),
            (STATE_PAUSED, 0.05),
        ],
        services={
            "start": [],
            "pause": [],
            "stop": [],
            "return_to_base": [],
        },
    ),
    "timer": DeviceType(
        name="timer",
        possible_states=[
            (STATE_IDLE, 0.2),
            (STATE_ACTIVE, 0.6),
            (STATE_PAUSED, 0.1),
        ],
        services={
            "start": ["duration"],
            "pause": [],
            "cancel": [],
        },
        random_parameter_generator={
            "duration": lambda: random.choice(list(pile_of_durations.keys())),
            "remaining": lambda: f"{random.randint(0, 3):02}:{random.randint(0, 60)}:{random.randint(0, 60)}"
        }
    ),
    "todo": DeviceType(
        name="todo",
        possible_states=[ (f"{i}", (1/32)) for i in range(32) ],
        services={
            "add_item": ["item"],
        },
        random_parameter_generator={
            "todo": lambda: random.choice(pile_of_todo_items),
        }
    ),
}

class NoResponseAvailableException(Exception):
    pass

class NoServicesAvailableException(Exception):
    pass

def get_random_response(*, service: str, persona: str, question_template: str, short: bool) -> str:

    required_vars = list(set([var for var in var_pattern.findall(question_template) if "device_name" not in var]))
    
    possible_results = pile_of_responses.loc[(pile_of_responses['service']==service) & 
                          (pile_of_responses['persona']==persona) &
                          (pile_of_responses['short']==(1 if short else 0)) &
                          (pile_of_responses['contains_vars']==",".join(sorted(required_vars)))
                        ]
    
    if len(possible_results) == 0:
        raise NoResponseAvailableException(f"No responses matched the provided filters: {persona}, {service}, {required_vars}, {short}")
    
    return possible_results.sample()["response"].values[0]

def format_device_line(*, device_name: str, friendly_name: str, state: str):
    return (f"{device_name} '{friendly_name}' = {state}")

# generate a random list of devices for the context
def random_device_list(max_devices: int, avoid_device_names: list[str]):
    num_devices = random.randint(2, max_devices)

    local_device_names = { k: v[:] for k,v in stacks_of_device_names.items() }

    avoid_climate = False
    for avoid_device in avoid_device_names:
        avoid_type = avoid_device.split(".")[0]

        filtered_possible_devices = []
        for possible_device in local_device_names[avoid_type]:
            similarity_ratio = SequenceMatcher(None, avoid_device, possible_device["device_name"].split(".")[1]).ratio()

            if similarity_ratio < 0.4:
                filtered_possible_devices.append(possible_device)
        local_device_names[avoid_type] = filtered_possible_devices

        if avoid_type == "climate":
            avoid_climate = True

    possible_choices = []
    for device_type in local_device_names.keys():
        possible_choices.extend(local_device_names[device_type])
    

    device_types = set()
    device_list = []
    device_lines = []
    # TODO: randomly pick attributes for this list
    extra_exposed_attributes = ["rgb_color", "brightness", "temperature", "humidity", "fan_mode", "media_title", "volume_level", "duration", "remaining", "item"]

    while len(device_list) < num_devices:
        choice = random.choice(possible_choices)
        if choice["device_name"] in device_list:
            continue

        try:
            device_name = choice["device_name"]
            device_type = device_name.split(".")[0]
            friendly_name = choice["description"]

            # don't add random thermostats. we need to be careful about how we handle multiple thermostats
            if avoid_climate and device_type == "climate":
                continue

            state = SUPPORTED_DEVICES[device_type].get_random_state(extra_exposed_attributes=extra_exposed_attributes)
            device_lines.append(format_device_line(
                device_name=device_name,
                friendly_name=friendly_name,
                state=state
            ))
            device_list.append(device_name)
            device_types.add(device_type)
        except Exception as ex:
            print(f"bad device name: {choice}")
            print(repr(ex))

    return device_lines, list(device_types), list(extra_exposed_attributes)

def generate_static_example(action: dict, persona: str, max_devices: int = 32):
    question = action["phrase"]
    service_name = action["service_name"]
    device_type = service_name.split(".")[0]
    target_device = f"{device_type}.{action['device_name']}"
    friendly_name = target_device.split(".")[1].replace("_", " ").title()

    device_list, device_types, extra_exposed_attributes = random_device_list(
        max_devices=max_devices, avoid_device_names=[target_device])

    # insert our target device somewhere random in the list
    index = random.randint(0, len(device_list))
    state = SUPPORTED_DEVICES[device_type].get_random_state(extra_exposed_attributes=extra_exposed_attributes)

    device_list.insert(index, format_device_line(
        device_name=target_device,
        friendly_name=friendly_name,
        state=state
    ))

    # gather a list of all available services
    available_services = []
    for x in set(device_types + [device_type]):
        available_services.extend(SUPPORTED_DEVICES[x].get_all_services(extra_exposed_attributes))

    response = get_random_response(
        service=action["service_name"],
        persona=persona,
        question_template="",
        short=False
    ).lower()

    response = response.replace("<device_name>", friendly_name)

    return {
        "states": device_list,
        "available_services": list(available_services),
        "question": question.lower(),
        "answers": [ response ],
        "service_calls": [ { "service": service_name, "target_device": target_device } ]
    }

def generate_templated_example(template: dict, persona: str, max_devices: int = 32):
    template_device_types: list[str] = template["device_type"].split("|")
    service_names: list[str] = [ f"{x}.{y}" for x, y in zip(template_device_types, template["service"].split("|")) ]
    question_template: str = template["phrase"]

    # choose a random device for this template
    chosen_devices = []
    for device_type in template_device_types:
        device_dict = random.choice(stacks_of_device_names[device_type])
        device_dict["type"] = device_type
        chosen_devices.append(device_dict)

    device_list, device_types, extra_exposed_attributes = random_device_list(
        max_devices=max_devices, avoid_device_names=[d["device_name"] for d in chosen_devices])

    # insert our target device somewhere random in the list
    for device_dict in chosen_devices:
        index = random.randint(0, len(device_list))
        if "<brightness>" in question_template and "brightness" not in extra_exposed_attributes:
            extra_exposed_attributes.append("brightness")
        if "<color>" in question_template and "rgb_color" not in extra_exposed_attributes:
            extra_exposed_attributes.append("rgb_color")
        if ("<temp_f>" in question_template or "<temp_c>" in question_template) \
            and "temperature" not in extra_exposed_attributes:
            extra_exposed_attributes.append("temperature")
        if "<humidity>" in question_template and "humidity" not in extra_exposed_attributes:
            extra_exposed_attributes.append("humidity")
        if "<fan_mode>" in question_template and "fan_mode" not in extra_exposed_attributes:
            extra_exposed_attributes.append("fan_mode")
        if "<duration>" in question_template and "duration" not in extra_exposed_attributes:
            extra_exposed_attributes.append("duration")

        state = SUPPORTED_DEVICES[device_dict["type"]].get_random_state(extra_exposed_attributes=extra_exposed_attributes)
        device_name = device_dict["device_name"]
        friendly_name = device_dict["description"]

        device_list.insert(index, format_device_line(
            device_name=device_name,
            friendly_name=friendly_name,
            state=state
        ))

    # gather a list of all available services with arguments
    available_services = []
    for x in set(device_types + template_device_types):
        available_services.extend(SUPPORTED_DEVICES[x].get_all_services(extra_exposed_attributes))

    # pick an appropriate response and generate the question
    if len(template_device_types) == 1:
        answer_template = get_random_response(
            service=service_names[0],
            persona=persona,
            question_template=question_template,
            short=False
        )

        question = question_template.replace("<device_name>", chosen_devices[0]["description"])
        answer = answer_template.replace("<device_name>", chosen_devices[0]["description"])
    else:        
        question = question_template
        answers = []
        for i in range(len(template_device_types)):
            question = question.replace(f"<device_name{(i + 1)}>", chosen_devices[i]["description"])
            answer = get_random_response(
                service=service_names[i],
                persona=persona,
                question_template=question_template,
                short=True
            )
            answers.append(answer.replace(f"<device_name>", chosen_devices[i]["description"]))

        answer = f" {and_word} ".join(answers)

    # generate the list of service calls and answers
    service_calls = []
    for device_dict, service in zip(chosen_devices, service_names):
        service_calls.append({ "service": service, "target_device": device_dict["device_name"] })

    if any(["climate" in service for service in service_names ]):
        climate_device_type = SUPPORTED_DEVICES["climate"]
        if "<hvac_mode>" in question:
            hvac_mode = climate_device_type.get_random_parameter("hvac_mode")
            question = question.replace("<hvac_mode>", hvac_mode)
            answer = answer.replace("<hvac_mode>", hvac_mode)
            service_calls = [ { **call, "hvac_mode": hvac_mode} for call in service_calls ]

        if "<fan_mode>" in question:
            fan_mode = climate_device_type.get_random_parameter("fan_mode")
            question = question.replace("<fan_mode>", fan_mode)
            answer = answer.replace("<fan_mode>", fan_mode)
            service_calls = [ { **call, "fan_mode": fan_mode} for call in service_calls ]

        if "<temp_f>" in question:
            temp_f = climate_device_type.get_random_parameter("temp_f")
            question = question.replace("<temp_f>", str(temp_f))
            answer = answer.replace("<temp_f>", str(temp_f))
            service_calls = [ { **call, "temperature": temp_f} for call in service_calls ]

        if "<temp_c>" in question:
            temp_c = climate_device_type.get_random_parameter("temp_c")
            question = question.replace("<temp_c>", str(temp_c))
            answer = answer.replace("<temp_c>", str(temp_c))
            service_calls = [ { **call, "temperature": temp_c} for call in service_calls ]

        if "<humidity>" in question:
            humidity = climate_device_type.get_random_parameter("humidity")
            question = question.replace("<humidity>", str(humidity))
            answer = answer.replace("<humidity>", str(humidity))
            service_calls = [ { **call, "humidity": humidity} for call in service_calls ]

    if any(["light" in service for service in service_names ]):
        light_device_type = SUPPORTED_DEVICES["light"]
        if "<brightness>" in question:
            brightness = light_device_type.get_random_parameter("brightness")
            question = question.replace("<brightness>", str(brightness))
            answer = answer.replace("<brightness>", str(brightness))
            service_calls = [ { **call, "brightness": round(brightness / 100, 2) } for call in service_calls ]

        if "<color>" in question:
            random_rgb = light_device_type.get_random_parameter("rgb_color")
            random_rgb_name = closest_color(random_rgb)
            actual_random_rgb = webcolors.name_to_rgb(random_rgb_name)
            actual_random_rgb = (actual_random_rgb.red, actual_random_rgb.green, actual_random_rgb.blue)
            question = question.replace("<color>", str(random_rgb_name))
            answer = answer.replace("<color>", str(random_rgb_name))
            service_calls = [ { **call, "rgb_color": str(actual_random_rgb) } for call in service_calls ]

    if any(["timer" in service for service in service_names ]):
        timer_device_type = SUPPORTED_DEVICES["timer"]
        if "<duration>" in question:
            duration = timer_device_type.get_random_parameter("duration")
            duration_name = pile_of_durations[duration]
            question = question.replace("<duration>", duration_name)
            answer = answer.replace("<duration>", duration_name)
            service_calls = [ { **call, "duration": str(duration) } for call in service_calls ]

    if any(["todo" in service for service in service_names ]):
        todo_device_type = SUPPORTED_DEVICES["todo"]
        if "<todo>" in question:
            todo = todo_device_type.get_random_parameter("todo")
            question = question.replace("<todo>", todo)
            answer = answer.replace("<todo>", todo)
            service_calls = [ { **call, "item": todo } for call in service_calls ]

    return {
        "states": device_list,
        "available_services": list(available_services),
        "question": question.lower(),
        "answers": [ answer.lower() ],
        "service_calls": service_calls
    }

def generate_status_request(template: dict, persona: str, max_devices: int = 32, return_target_device: bool = False):
    device_type: str = template["device_type"]
    state_name: str = template["state"]
    question_template: str = template["phrase"]
    answer_template: str = template["assistant_response"]

    # choose a random device for this template
    chosen_device = random.choice(stacks_of_device_names[device_type])

    # build a random list of devices
    device_list, device_types, extra_exposed_attributes = random_device_list(max_devices=max_devices, avoid_device_names=[ chosen_device["device_name"] ])

    # generate the question
    question = question_template.replace("<device_name>", chosen_device["description"])
    answer = answer_template.replace("<device_name>", chosen_device["description"])
    
    # insert other templated variables
    if device_type == "climate":
        climate_device_type = SUPPORTED_DEVICES["climate"]
        temp_f = climate_device_type.get_random_parameter("temp_f")
        answer = answer.replace("<temp_f>", str(temp_f))
        state_name = state_name.replace("<temp_f>", str(temp_f))

        temp_c = climate_device_type.get_random_parameter("temp_c")
        answer = answer.replace("<temp_c>", str(temp_c))
        state_name = state_name.replace("<temp_c>", str(temp_f))

        humidity = climate_device_type.get_random_parameter("humidity")
        answer = answer.replace("<humidity>", str(humidity))
        state_name = state_name.replace("<humidity>", str(temp_f))

    if device_type == "light":
        light_device_type = SUPPORTED_DEVICES["light"]

        brightness = light_device_type.get_random_parameter("brightness")
        answer = answer.replace("<brightness>", str(brightness))
        state_name = state_name.replace("<brightness>", str(brightness))

        random_rgb = light_device_type.get_random_parameter("rgb_color")
        random_rgb_name = closest_color(random_rgb)
        actual_random_rgb = webcolors.name_to_rgb(random_rgb_name)
        actual_random_rgb = (actual_random_rgb.red, actual_random_rgb.green, actual_random_rgb.blue)
        state_name = state_name.replace("<color>", str(random_rgb_name) + " " + str(actual_random_rgb))
        answer = answer.replace("<color>", str(random_rgb_name))

    if device_type == "media_player":
        media_player_device_type = SUPPORTED_DEVICES["media_player"]
        volume = media_player_device_type.get_random_parameter("volume")
        random_media = media_player_device_type.get_random_parameter("media")

        answer = answer.replace("<volume>", str(volume) + "%")
        state_name = state_name.replace("<volume>", str(volume) + "%")

        answer = answer.replace("<media>", random_media)
        state_name = state_name.replace("<media>", random_media)

    if device_type == "timer":
        timer_device_type = SUPPORTED_DEVICES["timer"]
        duration = timer_device_type.get_random_parameter("duration")
        duration_name = pile_of_durations[duration]
        remaining = timer_device_type.get_random_parameter("remaining")

        answer = answer.replace("<duration>", duration_name)
        state_name = state_name.replace("<duration>", duration)

        answer = answer.replace("<remaining>", remaining)
        state_name = state_name.replace("<remaining>", remaining)

    # insert our target device somewhere random in the list
    index = random.randint(0, len(device_list))
    device_list.insert(index, format_device_line(
        device_name=chosen_device["device_name"],
        friendly_name=chosen_device["description"],
        state=state_name
    ))

    # gather a list of all available services
    available_services = []
    for x in set(device_types + [device_type]):
        available_services.extend(SUPPORTED_DEVICES[x].get_all_services(extra_exposed_attributes))

    result = {
        "states": device_list,
        "available_services": list(available_services),
        "question": question.lower(),
        "answers": [ answer.lower() ],
        "service_calls": []
    }
    if return_target_device:
        return result, chosen_device
    else:
        return result

def generate_dpo_wrong_argument(template: dict, persona: str, max_devices: int = 32):
    """Generates examples of the model passing incorrect service call arguments"""
    while True:
        example = generate_templated_example(template, persona, max_devices)
        rejected_example = copy.deepcopy(example)

        call_idx = random.randint(0, len(example["service_calls"]) - 1)
        call = example["service_calls"][call_idx]

        target_device_type = call["target_device"].split(".")[0]

        potential_devices = [ x for x in example["states"] if x.split(".")[0] == target_device_type]
        random_device = random.choice(potential_devices).split(" ")[0]

        for device in stacks_of_device_names[call["target_device"].split(".")[0]]:
            similarity_ratio = SequenceMatcher(None, call["target_device"], device["device_name"]).ratio()

            if similarity_ratio > 0.7 and device["device_name"] != call["target_device"]:
                potential_devices.append(device["device_name"])
        
        if len(potential_devices) > 1:
            while random_device == call["target_device"]:
                random_device = random.choice(potential_devices).split(" ")[0]
        else:
            random_device = None

        # random service should probably be "related"
        available_services = [ x[:-2] for x in example["available_services"] if call["service"] not in x ]
        hallucinated_services = [ x["hallucinated_service"] for x in pile_of_hallucinated_service_names if x["real_service"] == call["service"].split(".")[1]]
        random_service = random.choice(available_services + hallucinated_services)
        random_argument = None # based on the service, add arguments that might be there like rgb, temperature, etc

        update_choices = []
        if random_device:
            update_choices.append({ "target_device": random_device })
        if random_service:
            update_choices.append({ "service": random_service })
        if random_argument:
            update_choices.append(random_argument)
        update_dict = random.choice(update_choices)

        # need to replace the response text with what the incorrect response would have been
        if len(rejected_example["service_calls"]) == 1:
            pass

        rejected_example["service_calls"][call_idx].update(update_dict)
        
        return { "accepted": example, "rejected": rejected_example }

def generate_dpo_no_service_call(template: dict, persona: str, max_devices: int = 32):
    """Generates examples of the model saying 'i'll do that for you' and generating no service calls"""
    example = generate_templated_example(template, persona, max_devices)
    rejected_example = copy.deepcopy(example)
    rejected_example["service_calls"] = []
    
    return { "accepted": example, "rejected": rejected_example }

def generate_dpo_extra_service_call(template: dict, persona: str, max_devices: int = 32):
    """Generates examples of the model adding random service calls to the end of status requests"""
    example, target_device = generate_status_request(template, persona, max_devices, return_target_device=True)
    rejected_example = copy.deepcopy(example)

    device_name = target_device["device_name"]
    device_type = device_name.split(".")[0]
    random_device_services = [ x for x in example["available_services"] if x.split(".")[0] == device_type ]

    if len(random_device_services) == 0:
        raise NoServicesAvailableException()
    
    rejected_example["service_calls"] = [{ "service": random.choice(random_device_services), "target_device": device_name }]
    
    return { "accepted": example, "rejected": rejected_example }

def generate_dpo_incorrect_persona(template: dict, persona: str, max_devices: int = 32):
    pass

def format_example_raw_chatml(example, persona):
    """Don't use this one anymore"""
    sys_prompt = pile_of_system_prompts[persona]
    services_block = "Services: " + ", ".join(sorted(example["available_services"]))
    states_block = "Devices:\n" + "\n".join(example["states"])
    question = example["question"]
    answers = " ".join(example["answers"])

    system_block = "\n".join([ "<|im_start|>system", sys_prompt, services_block, states_block ]) + "<|im_end|>"
    user_block = "\n".join([ "<|im_start|>user", question]) + "<|im_end|>"

    assistant_block = "<|im_start|>assistant\n" + answers
    if len(example["service_calls"]) > 0:
        json_calls = [ json.dumps(x) for x in example["service_calls"] ]
        code_block = "\n```homeassistant\n" + "\n".join(json_calls) + "\n```"
        assistant_block = assistant_block + code_block
    assistant_block = assistant_block + "<|im_end|>"
        
    example_lines = [system_block, user_block, assistant_block]
    result = "\n".join(example_lines)
    if "<device_name" in result:
        print("bad templating")

    # replace aliases with their actual values
    result = result.replace("blinds.", "cover.")
    result = result.replace("garage_door.", "cover.")
    return { "text": result }

def format_example_sharegpt(example, persona):
    sys_prompt = pile_of_system_prompts[persona]
    time_block = "The current time and date is " + generate_random_date().strftime("%I:%M %p on %A %B %d, %Y")
    services_block = "Services: " + ", ".join(sorted(example["available_services"]))
    states_block = "Devices:\n" + "\n".join(example["states"])
    question = example["question"]
    answers = " ".join(example["answers"])

    assistant_block = answers
    if len(example["service_calls"]) > 0:
        json_calls = [ json.dumps(x) for x in example["service_calls"] ]
        code_block = "\n```homeassistant\n" + "\n".join(json_calls) + "\n```"
        assistant_block = assistant_block + code_block

    # replace aliases with their actual values
    assistant_block = assistant_block.replace("blinds.", "cover.").replace("garage_door.", "cover.")
    states_block = states_block.replace("blinds.", "cover.").replace("garage_door.", "cover.")
    services_block = services_block.replace("blinds.", "cover.").replace("garage_door.", "cover.")

    conversation = [
        { "from": "system", "value": "\n".join([ sys_prompt, time_block, services_block, states_block ])},
        { "from": "user", "value": question },
        { "from": "assistant", "value": assistant_block },
    ]
    
    return { "conversations": conversation }

def format_example_dpo(example, persona):
    rejected_example = example["rejected"]
    example = example["accepted"]

    sys_prompt = pile_of_system_prompts[persona]
    services_block = "Services: " + ", ".join(sorted(example["available_services"]))
    states_block = "Devices:\n" + "\n".join(example["states"])
    question = example["question"]

    assistant_block = " ".join(example["answers"])
    if len(example["service_calls"]) > 0:
        json_calls = [ json.dumps(x) for x in example["service_calls"] ]
        code_block = "\n```homeassistant\n" + "\n".join(json_calls) + "\n```"
        assistant_block = assistant_block + code_block

    rejected_assistant_block = " ".join(rejected_example["answers"])
    if len(rejected_example["service_calls"]) > 0:
        json_calls = [ json.dumps(x) for x in rejected_example["service_calls"] ]
        code_block = "\n```homeassistant\n" + "\n".join(json_calls) + "\n```"
        rejected_assistant_block = rejected_assistant_block + code_block

    # replace aliases with their actual values
    assistant_block = assistant_block.replace("blinds.", "cover.").replace("garage_door.", "cover.")
    rejected_assistant_block = rejected_assistant_block.replace("blinds.", "cover.").replace("garage_door.", "cover.")
    states_block = states_block.replace("blinds.", "cover.").replace("garage_door.", "cover.")
    services_block = services_block.replace("blinds.", "cover.").replace("garage_door.", "cover.")
    
    return {
        "system": "\n".join([ sys_prompt, services_block, states_block ]),
        "question": question,
        "chosen": assistant_block,
        "rejected": rejected_assistant_block,
    }

def generate_sft_file(filename: str, seed: int, format_func: Callable, personas: list[str], *, static_factor: int, template_factor: int, status_request_factor: int):
    random.seed(seed)
    np.random.seed(seed)

    print("Generating...")

    def run_factor_times(func, examples, data, persona, factor):
        if factor >= 1:
            for i in range(factor):
                examples.append(format_func(func(data, persona), persona))
        else:
            if random.random() < factor:
                examples.append(format_func(func(data, persona), persona))
    
    generated_examples = []

    missing_responses = set()

    for person in personas:
        for action in tqdm(pile_of_specific_actions):
            try:
                run_factor_times(generate_static_example, generated_examples, action, person, static_factor)
            except NoResponseAvailableException as ex:
                missing_responses.add(str(ex))

        for templated_action in tqdm(pile_of_templated_actions):
            try:
                run_factor_times(generate_templated_example, generated_examples, templated_action, person, template_factor)
            except NoResponseAvailableException as ex:
                missing_responses.add(str(ex))

    for status_request in tqdm(pile_of_status_requests):
        run_factor_times(generate_status_request, generated_examples, status_request, "assistant", status_request_factor)

    print(f"Generated {len(generated_examples)} examples. Saving...")

    for missing in sorted(missing_responses):
        print(missing)
    
    with open(f"{filename}.jsonl", "w") as f:
        for item in generated_examples:
            json_record = json.dumps(item)
            f.write(json_record + '\n')

    print("Done!")


def generate_dpo_file(filename: str, seed: int, format_func: Callable, personas: list[str], *, wrong_argument_factor: int, no_argument_factor: int, extra_service_call_factor: int, incorrect_persona_factor: int):
    random.seed(seed)
    np.random.seed(seed)

    print("Generating...")

    def run_factor_times(func, examples, data, persona, factor):
        if factor >= 1:
            for i in range(factor):
                examples.append(format_func(func(data, persona), persona))
        else:
            if random.random() < factor:
                examples.append(format_func(func(data, persona), persona))
    
    generated_examples = []

    missing_responses = set()

    for person in personas:
        for templated_action in tqdm(pile_of_templated_actions):
            try:
                run_factor_times(generate_dpo_wrong_argument, generated_examples, templated_action, person, wrong_argument_factor)
                run_factor_times(generate_dpo_no_service_call, generated_examples, templated_action, person, no_argument_factor)
                # run_factor_times(generate_dpo_incorrect_persona, generated_examples, templated_action, person, incorrect_persona_factor)
            except NoResponseAvailableException as ex:
                missing_responses.add(str(ex))

    for status_request in tqdm(pile_of_status_requests):
        try:
            run_factor_times(generate_dpo_extra_service_call, generated_examples, status_request, "assistant", extra_service_call_factor)
        except NoServicesAvailableException as ex:
            pass # TODO: warn here?

    print(f"Generated {len(generated_examples)} DPO examples. Saving...")

    for missing in sorted(missing_responses):
        print(missing)
    
    with open(f"{filename}.jsonl", "w") as f:
        for item in generated_examples:
            json_record = json.dumps(item)
            f.write(json_record + '\n')

    print("Done!")

def format_alpaca(example, format_func: Callable):
    question = example["instruction"]
    if "input" in example and example["input"]:
        question = question = "\n" + example["input"]

    answer = example["output"]

    device_list, device_types, extra_exposed_attributes = random_device_list(
        max_devices=32, avoid_device_names=[])

    available_services = []
    for x in device_types:
        available_services.extend(SUPPORTED_DEVICES[x].get_all_services(extra_exposed_attributes))

    text = format_func(example={
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

def merge_with_dataset(dataset_name, seed, output_name, format_function, dataset_column_names, format_func):
    alpaca_dataset = load_dataset(dataset_name)["train"].train_test_split(test_size=0.1)
    home_assistant_dataset = load_dataset("json", data_files={  "train": "home_assistant_train.jsonl", "test": "home_assistant_test.jsonl" })

    random.seed(seed)
    np.random.seed(seed)

    alpaca_dataset = alpaca_dataset.map(format_function).remove_columns(dataset_column_names)

    combined_dataset_train = concatenate_datasets([home_assistant_dataset["train"], alpaca_dataset["train"]]).shuffle(seed=42)
    combined_dataset_test = concatenate_datasets([home_assistant_dataset["test"], alpaca_dataset["test"]]).shuffle(seed=42)

    combined_dataset_train.to_json(f"home_assistant_{output_name}_merged_train.jsonl")
    combined_dataset_test.to_json(f"home_assistant_{output_name}_merged_test.jsonl")

def merge_languages(filename_prefix: str, languages: list):
    all_examples = []
    for language in languages:
        with open(f"{filename_prefix}_{language}.jsonl") as f:
            all_examples.extend(f.readlines())

    with open(f"{filename_prefix}.jsonl", "w") as f:
        f.writelines(all_examples)


def load_dataset_piles(language):
    global pile_of_durations, pile_of_media_names, pile_of_todo_items, stacks_of_device_names, \
        pile_of_templated_actions, pile_of_specific_actions, pile_of_responses, pile_of_status_requests, \
        pile_of_system_prompts, pile_of_hallucinated_service_names, and_word
    
    # TODO: make this properly dynamic
    if language == "english":
        and_word = "and"
    elif language == "german":
        and_word = "und"
    elif language == "french":
        and_word = "et"
    elif language == "spanish":
        and_word = "y"
    
    with open(f"piles/{language}/pile_of_durations.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        pile_of_durations = { x["duration"]: x["name"] for x in reader }
        
    # media names are not translated
    with open(f"piles/english/pile_of_media_names.txt", encoding="utf8") as f:
        pile_of_media_names = [ x.strip() for x in f.readlines() ]

    with open(f"piles/{language}/pile_of_todo_items.txt", encoding="utf8") as f:
        pile_of_todo_items = [ x.strip() for x in f.readlines() ]

    stacks_of_device_names = { x: [] for x in SUPPORTED_DEVICES.keys() }
    with open(f"piles/{language}/pile_of_device_names.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        pile_of_device_names = list(reader)
        for device_dict in pile_of_device_names:
            try:
                device_type = device_dict["device_name"].split(".")[0]
                stacks_of_device_names[device_type].append(device_dict)
            except KeyError as ex:
                print(ex)

    with open(f"piles/{language}/pile_of_templated_actions.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        pile_of_templated_actions = list(reader)
        processed_pile_of_templated_actions = []
        for action in pile_of_templated_actions:
            try:
                multiplier = int(action["multiplier"])
            except Exception:
                raise Exception(f"line has a bad multiplier: {action}")
            for x in range(multiplier):
                processed_pile_of_templated_actions.append(action)

        pile_of_templated_actions = processed_pile_of_templated_actions

    with open(f"piles/{language}/pile_of_specific_actions.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        pile_of_specific_actions = list(reader)

    pile_of_responses = pandas.read_csv(f"piles/{language}/pile_of_responses.csv")
    pile_of_responses["contains_vars"] = pile_of_responses["response"].apply(get_included_vars)

    with open(f"piles/{language}/pile_of_status_requests.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        pile_of_status_requests = list(reader)

    with open(f"piles/{language}/pile_of_system_prompts.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        pile_of_system_prompts = { line["persona"]: line["prompt"] for line in reader }

    # service names are not translated
    with open(f"piles/english/pile_of_hallucinated_service_names.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        pile_of_hallucinated_service_names = list(reader)

# TODO: add examples for ambiguous requests. asking a clarifying question
# TODO: support rejection when asking to do a service that isn't exposed
# TODO: make more randomized names for devices (random words or people's names)
# TODO: answer questions about more than one thing in the state list at once
# TODO: add examples for rooms/groups of devices. i.e. "turn off all the lights in the kitchen"
# TODO: add time, weather, and calendar/reminders (next 3 events?)
def main():
    parser = argparse.ArgumentParser(description="Generate the full dataset from the CSV piles")
    parser.add_argument("--sample", action="store_true", help="Set this flag to enable generation of the train dataset.")
    parser.add_argument("--test", action="store_true", help="Set this flag to enable generation of the train dataset..")
    parser.add_argument("--train", action="store_true", help="Set this flag to enable generation of the train dataset.")
    parser.add_argument("--dpo", action="store_true", help="Set this flag to enable generation of the DPO dataset.")
    parser.add_argument("--merge", help="Set this flag to merge the generated datasets with the specified dataset.")
    parser.add_argument("--language", nargs="+", default=["english"], help="List of languages to generate")

    train_size_group = parser.add_mutually_exclusive_group()
    train_size_group.add_argument('--small', action='store_const', const='small', dest='size')
    train_size_group.add_argument('--medium', action='store_const', const='medium', dest='size')
    train_size_group.add_argument('--large', action='store_const', const='large', dest='size')
    train_size_group.add_argument('--xl', action='store_const', const='xl', dest='size')

    dataset_format_group = parser.add_mutually_exclusive_group()
    dataset_format_group.add_argument('--raw_corpus', action='store_const', const='raw', dest='format')
    dataset_format_group.add_argument('--sharegpt', action='store_const', const='sharegpt', dest='format')

    args = parser.parse_args()

    if not args.sample and not args.train and not args.test and not args.merge and not args.dpo:
        parser.print_usage()
        exit(-1)

    if args.size and not args.train:
        print("Train size was provided but not generating the training set!")
        exit(-1)
    
    if not args.format or args.format == "raw":
        format_func = format_example_raw_chatml
    elif args.format == "sharegpt":
        format_func = format_example_sharegpt

    for language in args.language:
        load_dataset_piles(language)
        personas = list(pile_of_system_prompts.keys())
        suffix = f"_{language}" if len(args.language) > 1 else ""

        if args.sample:
            generate_sft_file(f"sample{suffix}", 42, format_func, personas, static_factor=1, template_factor=1, status_request_factor=1)
        if args.train:
            if args.size == "small":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, personas, static_factor=1, template_factor=10, status_request_factor=8)
            elif args.size == "medium":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, personas, static_factor=5, template_factor=15, status_request_factor=12)
            elif args.size == "large":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, personas, static_factor=5, template_factor=20, status_request_factor=15)
            elif args.size == "xl":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, personas, static_factor=7, template_factor=25, status_request_factor=18)
            else:
                raise Exception(f"Unrecognized dataset size: {args.size}")
        if args.test:
            generate_sft_file(f"home_assistant_test{suffix}", 12345, format_func, personas, static_factor=0.25, template_factor=1, status_request_factor=2)

    if len(args.language) > 1:
        if args.sample:
            merge_languages("sample", args.language)
        if args.train:
            merge_languages("home_assistant_train", args.language)
        if args.test:
            merge_languages("home_assistant_test", args.language)

    if args.dpo:
        generate_dpo_file(f"home_assistant_dpo", 42, format_example_dpo, personas, wrong_argument_factor=1, no_argument_factor=1, extra_service_call_factor=1, incorrect_persona_factor=1)

    if args.merge == "alpaca":
        merge_with_dataset("yahma/alpaca-cleaned", 42, "alpaca", format_alpaca, ["input", "output", "instruction"], format_func)
    elif args.merge == "wizardlm70k":
        merge_with_dataset("WizardLM/WizardLM_evol_instruct_70k", 42, "wizardlm70k", format_alpaca, ["output", "instruction"], format_func)

if __name__ == "__main__":
    main()
