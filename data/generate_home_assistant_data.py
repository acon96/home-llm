import argparse
import json
import csv
import pandas
import numpy as np
import random
import re
import copy
import babel.dates
from dataclasses import dataclass
from datetime import datetime, timedelta
from datasets import load_dataset, concatenate_datasets
from difflib import SequenceMatcher
from typing import Final, Any, Callable, Optional
from tqdm import tqdm
import webcolors

# STATES
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

# TOOLS
TOOL_TURN_ON = "HassTurnOn"
TOOL_TURN_OFF = "HassTurnOff"
TOOL_TOGGLE = "HassToggle"
TOOL_SET_POSITION = "HassSetPosition"
TOOL_LIGHT_SET = "HassLightSet"
TOOL_SET_VOLUME = "HassSetVolume"
TOOL_MEDIA_UNPAUSE = "HassMediaUnpause"
TOOL_MEDIA_PAUSE = "HassMediaPause"
TOOL_MEDIA_NEXT = "HassMediaNext"
TOOL_MEDIA_PREVIOUS = "HassMediaPrevious"
TOOL_VACUUM_START = "HassVacuumStart"
TOOL_VACUUM_RETURN_TO_BASE = "HassVacuumReturnToBase"
TOOL_LIST_ADD_ITEM = "HassListAddItem"
TOOL_START_TIMER = "HassStartTimer"
TOOL_CANCEL_TIMER = "HassCancelTimer"
TOOL_PAUSE_TIMER = "HassPauseTimer"
TOOL_UNPAUSE_TIMER = "HassUnpauseTimer"
TOOL_INCREASE_TIMER = "HassIncreaseTimer"
TOOL_DECREASE_TIMER = "HassDecreaseTimer"
TOOL_TIMER_STATUS = "HassTimerStatus"
TOOL_CLIMATE_SET_TEMPERATURE = "HassClimateSetTemperature"
TOOL_CLIMATE_GET_TEMPERATURE = "HassClimateGetTemperature"
TOOL_SET_HUMIDITY = "HassHumidifierSetpoint"
TOOL_SET_HUMIDIFIER_MODE = "HassHumidifierMode"

# Service name to tool name mapping for backwards compatibility with CSV files
SERVICE_TO_TOOL_MAP = {
    "turn_on": TOOL_TURN_ON,
    "turn_off": TOOL_TURN_OFF,
    "toggle": TOOL_TOGGLE,
    "open_cover": TOOL_TURN_ON,
    "close_cover": TOOL_TURN_OFF,
    "stop_cover": TOOL_TOGGLE,
    "set_cover_position": TOOL_SET_POSITION,
    "lock": TOOL_TURN_ON,
    "unlock": TOOL_TURN_OFF,
    "increase_speed": TOOL_TURN_ON,
    "decrease_speed": TOOL_TURN_OFF,
    "media_play_pause": TOOL_TOGGLE,
    "media_pause": TOOL_MEDIA_PAUSE,
    "media_play": TOOL_MEDIA_UNPAUSE,
    "media_next_track": TOOL_MEDIA_NEXT,
    "media_previous_track": TOOL_MEDIA_PREVIOUS,
    "start": TOOL_VACUUM_START,
    "return_to_base": TOOL_VACUUM_RETURN_TO_BASE,
    "set_temperature": TOOL_CLIMATE_SET_TEMPERATURE,
    "set_humidity": TOOL_SET_HUMIDITY,
    "set_hvac_mode": TOOL_CLIMATE_SET_TEMPERATURE,
    "set_fan_mode": TOOL_CLIMATE_SET_TEMPERATURE,
    "set_preset_mode": TOOL_CLIMATE_SET_TEMPERATURE,
}

# Home Assistant Intent Tools Definition
HASS_TOOLS = [
    {
        "name": TOOL_TURN_ON,
        "description": "Turns on/opens/unlocks a device or entity",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the device or entity"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"},
                "domain": {"type": "array", "items": {"type": "string"}, "description": "Device domain(s)"},
                "device_class": {"type": "array", "items": {"type": "string"}, "description": "Device class(es)"}
            },
            "required": []
        }
    },
    {
        "name": TOOL_TURN_OFF,
        "description": "Turns off/closes/locks a device or entity",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the device or entity"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"},
                "domain": {"type": "array", "items": {"type": "string"}, "description": "Device domain(s)"},
                "device_class": {"type": "array", "items": {"type": "string"}, "description": "Device class(es)"}
            },
            "required": []
        }
    },
    {
        "name": TOOL_TOGGLE,
        "description": "Toggles a device or entity",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the device or entity"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"},
                "domain": {"type": "array", "items": {"type": "string"}, "description": "Device domain(s)"},
                "device_class": {"type": "array", "items": {"type": "string"}, "description": "Device class(es)"}
            },
            "required": []
        }
    },
    {
        "name": TOOL_SET_POSITION,
        "description": "Sets the position of a device or entity (e.g., blinds, covers)",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the device or entity"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"},
                "position": {"type": "integer", "description": "Position from 0-100", "minimum": 0, "maximum": 100}
            },
            "required": ["position"]
        }
    },
    {
        "name": TOOL_LIGHT_SET,
        "description": "Sets the brightness or color of a light",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the light"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"},
                "color": {"type": "string", "description": "Color name"},
                "temperature": {"type": "integer", "description": "Color temperature in Kelvin"},
                "brightness": {"type": "integer", "description": "Brightness percentage (0-100)", "minimum": 0, "maximum": 100}
            },
            "required": []
        }
    },
    {
        "name": TOOL_CLIMATE_SET_TEMPERATURE,
        "description": "Sets the target temperature of a climate device",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the climate device"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"},
                "temperature": {"type": "number", "description": "Target temperature"}
            },
            "required": ["temperature"]
        }
    },
    {
        "name": TOOL_SET_HUMIDITY,
        "description": "Sets the target humidity level of a humidifier device",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the humidifier"},
                "humidity": {"type": "integer", "description": "Target humidity percentage (0-100)", "minimum": 0, "maximum": 100}
            },
            "required": ["name", "humidity"]
        }
    },
    {
        "name": TOOL_SET_HUMIDIFIER_MODE,
        "description": "Sets the mode of a humidifier device",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the humidifier"},
                "mode": {"type": "string", "description": "Humidifier mode"}
            },
            "required": ["name", "mode"]
        }
    },
    {
        "name": TOOL_MEDIA_UNPAUSE,
        "description": "Resumes playback on a media player",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"}
            },
            "required": []
        }
    },
    {
        "name": TOOL_MEDIA_PAUSE,
        "description": "Pauses playback on a media player",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"}
            },
            "required": []
        }
    },
    {
        "name": TOOL_MEDIA_NEXT,
        "description": "Skips to the next media item on a media player",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"}
            },
            "required": []
        }
    },
    {
        "name": TOOL_SET_VOLUME,
        "description": "Sets the volume of a media player",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"},
                "volume_level": {"type": "number", "description": "Volume level (0.0-1.0)", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["volume_level"]
        }
    },
    {
        "name": TOOL_VACUUM_START,
        "description": "Starts a vacuum",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the vacuum"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"}
            },
            "required": []
        }
    },
    {
        "name": TOOL_VACUUM_RETURN_TO_BASE,
        "description": "Returns a vacuum to its base",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the vacuum"},
                "area": {"type": "string", "description": "Name of the area"},
                "floor": {"type": "string", "description": "Name of the floor"}
            },
            "required": []
        }
    },
    {
        "name": TOOL_LIST_ADD_ITEM,
        "description": "Adds an item to a todo list",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "The item to add to the list"},
                "name": {"type": "string", "description": "Name of the todo list"}
            },
            "required": ["item"]
        }
    },
    {
        "name": TOOL_START_TIMER,
        "description": "Starts a timer",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the timer"},
                "duration": {"type": "string", "description": "Timer duration (HH:MM:SS format)"}
            },
            "required": []
        }
    },
    {
        "name": TOOL_CANCEL_TIMER,
        "description": "Cancels a timer",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the timer"}
            },
            "required": []
        }
    },
    {
        "name": TOOL_PAUSE_TIMER,
        "description": "Pauses a timer",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the timer"}
            },
            "required": []
        }
    },
    {
        "name": TOOL_UNPAUSE_TIMER,
        "description": "Resumes a paused timer",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the timer"}
            },
            "required": []
        }
    }
]


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
and_words = None

def closest_color(requested_color):
    min_colors = {}
    color_names = webcolors.names("css3")
    
    for name in color_names:
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def generate_random_datetime():
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

RANDOM_PARAMETER_GENERATORS = {
    "rgb_color": lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    "brightness": lambda: random.randint(0, 100),
    "fan_mode": lambda: random.choice(["On Low", "On High", "Auto Low", "Auto High", "Off"]),
    "temp_f": lambda: random.randint(60, 80),
    "temp_c": lambda: random.randint(15, 25),
    "humidity": lambda: random.randint(10, 90),
    "preset_mode": lambda: random.choice(["home", "eco", "away", "auto"]),
    "hvac_mode": lambda: random.choice(["heat", "cool", "heat_cool", "off", "auto", "fan_only"]),
    "media": lambda: random.choice(pile_of_media_names),
    "volume": lambda: round(random.random(), 2),
    "duration": lambda: random.choice(list(pile_of_durations.keys())),
    "remaining": lambda: f"{random.randint(0, 3):02}:{random.randint(0, 60)}:{random.randint(0, 60)}",
    "todo": lambda: random.choice(pile_of_todo_items),
}

def generate_random_parameter(param_name):
    param_generator = RANDOM_PARAMETER_GENERATORS.get(param_name)

    if not param_generator:
        raise Exception(f"Unknown param to generate random value for {param_name}")
    
    return param_generator()

@dataclass
class DeviceType:
    name: str
    possible_states: list[tuple[str, float]]

    def get_random_state(self, extra_exposed_attributes=[]):
        states = [ x[0] for x in self.possible_states ]
        weights = [ x[1] for x in self.possible_states ]
        return random.choices(states, weights=weights, k=1)[0]
    
    def get_all_tools(self, extra_exposed_attributes=[]):
        """Return list of tool names available for this device type."""
        tools = [TOOL_TURN_ON, TOOL_TURN_OFF, TOOL_TOGGLE]
        return tools
    
    def get_random_parameter(self, param_name: str):
        """Generate a random parameter value."""
        return generate_random_parameter(param_name)

class LightDeviceType(DeviceType):
    def __init__(self):
        super().__init__("light",
            possible_states=[
                (STATE_ON, 0.5),
                (STATE_OFF, 0.5)
            ]
        )

    def get_random_state(self, extra_exposed_attributes=[]):
        state = super().get_random_state(extra_exposed_attributes=extra_exposed_attributes)

        if random.random() < 0.5 and "rgb_color" in extra_exposed_attributes:
            random_rgb = generate_random_parameter("rgb_color")
            state = state + ";" + closest_color(random_rgb) + " " + str(random_rgb)

        if random.random() < 0.7 and "brightness" in extra_exposed_attributes:
            state = state + ";" + str(generate_random_parameter("brightness")) + "%"

        return state
    
    def get_all_tools(self, extra_exposed_attributes=[]):
        """Return list of tool names available for lights."""
        tools = [TOOL_TURN_ON, TOOL_TURN_OFF, TOOL_TOGGLE]
        if "brightness" in extra_exposed_attributes or "rgb_color" in extra_exposed_attributes:
            tools.append(TOOL_LIGHT_SET)
        return tools
    
class ClimateDeviceType(DeviceType):
    def __init__(self):
        super().__init__("climate", [])

    def get_random_state(self, extra_exposed_attributes=[]):
        """state;fan_mode;temperature;humidity"""
        state = generate_random_parameter("hvac_mode")

        if "fan_mode" in extra_exposed_attributes:
            state = state  + ";" + generate_random_parameter("fan_mode")
        if "temperature" in extra_exposed_attributes:
            if random.random() > 0.5:
                state = state + ";" + str(generate_random_parameter("temp_f")) + "F" 
            else:
                state = state + ";" + str(generate_random_parameter("temp_c")) + "C"
        if "humidity" in extra_exposed_attributes:
            state = state + ";" + str(generate_random_parameter("humidity")) + "%"

        if random.random() < 0.8 and "preset_mode" in extra_exposed_attributes:
            # if it is not "on a preset" then don't add the mode
            state = state + ";" + generate_random_parameter("preset_mode")

        return state
    
    def get_all_tools(self, extra_exposed_attributes=[]):
        """Return list of tool names available for climate devices."""
        tools = [TOOL_TURN_ON, TOOL_TURN_OFF]
        if "temperature" in extra_exposed_attributes or "fan_mode" in extra_exposed_attributes:
            tools.append(TOOL_CLIMATE_SET_TEMPERATURE)
        if "humidity" in extra_exposed_attributes:
            tools.extend([TOOL_SET_HUMIDITY, TOOL_SET_HUMIDIFIER_MODE])
        return tools

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
        ])

    def get_random_state(self, extra_exposed_attributes=[]):
        state = super().get_random_state(extra_exposed_attributes=extra_exposed_attributes)

        if "media_title" in extra_exposed_attributes and state in [STATE_PLAYING, STATE_PAUSED, STATE_BUFFERING, STATE_ON]:
            state = state + ";" + generate_random_parameter("media")

        if "volume_level" in extra_exposed_attributes and state != STATE_OFF:
            state = state + ";vol=" + str(generate_random_parameter("volume"))
        return state
    
    def get_all_tools(self, extra_exposed_attributes=[]):
        """Return list of tool names available for media players."""
        tools = [TOOL_TURN_ON, TOOL_TURN_OFF, TOOL_MEDIA_PAUSE, TOOL_MEDIA_UNPAUSE, TOOL_MEDIA_NEXT]
        if "volume_level" in extra_exposed_attributes:
            tools.append(TOOL_SET_VOLUME)
        return tools

SUPPORTED_DEVICES = {
    "light": LightDeviceType(),
    "switch": DeviceType(
        name="switch",
        possible_states=[
            (STATE_ON, 0.5),
            (STATE_OFF, 0.5)
        ]
    ),
    "fan": DeviceType(
        name="fan",
        possible_states=[
            (STATE_ON, 0.5),
            (STATE_OFF, 0.5)
        ]
    ),
    "garage_door": DeviceType(
        name="garage_door",
        possible_states=[
            (STATE_OPEN, 0.49),
            (STATE_CLOSED, 0.49),
            (STATE_OPENING, 0.01),
            (STATE_CLOSING, 0.01)
        ]
    ),
    "blinds": DeviceType(
        name="blinds",
        possible_states=[
            (STATE_OPEN, 0.49),
            (STATE_CLOSED, 0.49),
            (STATE_OPENING, 0.01),
            (STATE_CLOSING, 0.01)
        ]
    ),
    "lock": DeviceType(
        name="lock",
        possible_states=[
            (STATE_LOCKED, 0.5),
            (STATE_UNLOCKED, 0.5),
        ]
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
        ]
    ),
    "timer": DeviceType(
        name="timer",
        possible_states=[
            (STATE_IDLE, 0.2),
            (STATE_ACTIVE, 0.6),
            (STATE_PAUSED, 0.1),
        ]
    ),
    "todo": DeviceType(
        name="todo",
        possible_states=[ (f"{i}", (1/32)) for i in range(32) ],
    ),
}

# Override get_all_tools for specific device types
def _vacuum_get_tools(extra_exposed_attributes=[]):
    return [TOOL_VACUUM_START, TOOL_VACUUM_RETURN_TO_BASE]

def _timer_get_tools(extra_exposed_attributes=[]):
    tools = [TOOL_START_TIMER, TOOL_CANCEL_TIMER, TOOL_PAUSE_TIMER, TOOL_UNPAUSE_TIMER]
    if "duration" in extra_exposed_attributes:
        tools.extend([TOOL_INCREASE_TIMER, TOOL_DECREASE_TIMER, TOOL_TIMER_STATUS])
    return tools

def _todo_get_tools(extra_exposed_attributes=[]):
    return [TOOL_LIST_ADD_ITEM]

def _cover_get_tools(extra_exposed_attributes=[]):
    tools = [TOOL_TURN_ON, TOOL_TURN_OFF, TOOL_TOGGLE]
    if "position" in extra_exposed_attributes:
        tools.append(TOOL_SET_POSITION)
    return tools

SUPPORTED_DEVICES["vacuum"].get_all_tools = _vacuum_get_tools
SUPPORTED_DEVICES["timer"].get_all_tools = _timer_get_tools
SUPPORTED_DEVICES["todo"].get_all_tools = _todo_get_tools
SUPPORTED_DEVICES["garage_door"].get_all_tools = _cover_get_tools
SUPPORTED_DEVICES["blinds"].get_all_tools = _cover_get_tools

CURRENT_DATE_PROMPT = {
    "english": "The current time and date is",
    "polish": "Aktualna godzina i data to",
    "german": "Die aktuelle Uhrzeit und das aktuelle Datum sind",
    "french": "L'heure et la date actuelles sont",
    "spanish": "La hora y fecha actuales son"
}

DEVICES_PROMPT = {
    "english": "Devices",
    "polish": "Urządzenia",
    "german": "Ger\u00e4te",
    "french": "Appareils",
    "spanish": "Dispositivos"
}

SERVICES_PROMPT = {
    "english": "Services",
    "polish": "Usługi",
    "german": "Dienste",
    "french": "Services",
    "spanish": "Servicios"
}

BABEL_LOCALE = {
    "english": "en_US",
    "polish": "pl_PL",
    "german": "de_DE",
    "french": "fr_FR",
    "spanish": "es_ES"
}

BABEL_FORMAT = {
    "english": "h:m a 'on' EEEE, MMMM d yyyy",
    "polish": "H:m 'w' EEEE, d MMMM yyyy",
    "german": "H:m EEEE, d MMMM yyyy",
    "french": "H:m EEEE, d MMMM yyyy",
    "spanish": "H:m EEEE, d 'de' MMMM 'de' yyyy"
}

USER_INSTRUCTION_PROMPT = {
    "english": "User instruction",
    "german": "Benutzeranweisung",
    "french": "Instruction de l'utilisateur ",
    "spanish": "Instrucción del usuario",
    "polish": "Instrukcja użytkownika"
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

def generate_static_example(action: dict, persona: str, max_devices: int = 32, use_service_names: bool = False):
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

    # gather a list of all available tools
    available_tools = []
    for x in set(device_types + [device_type]):
        available_tools.extend(SUPPORTED_DEVICES[x].get_all_tools(extra_exposed_attributes))
    
    # Remove duplicates while preserving order
    available_tools = list(dict.fromkeys(available_tools))

    # Map service name to tool name
    service_action = service_name.split(".")[1]
    tool_name = SERVICE_TO_TOOL_MAP.get(service_action, TOOL_TURN_ON)

    response = get_random_response(
        service=service_name,
        persona=persona,
        question_template="",
        short=False
    ).lower()

    response = response.replace("<device_name>", friendly_name)

    # Build tool call - use entity_id if service names mode, otherwise use friendly name
    if use_service_names:
        tool_call = {
            "tool_name": tool_name,
            "service_name": service_name,
            "tool_args": {"entity_id": target_device}
        }
    else:
        tool_call = {
            "tool_name": tool_name,
            "service_name": service_name,
            "tool_args": {"name": target_device}
        }

    return {
        "states": device_list,
        "available_tools": available_tools,
        "question": question.lower(),
        "answers": [ response ],
        "tool_calls": [ tool_call ]
    }

def replace_answer(list_of_answer, var, value):
    new_list = []
    for answer in list_of_answer:
        new_list.append(answer.replace(var, value))
    return new_list

def generate_templated_example(template: dict, persona: str, max_devices: int = 32, use_service_names: bool = False):
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

    # gather a list of all available tools
    available_tools = []
    for x in set(device_types + template_device_types):
        available_tools.extend(SUPPORTED_DEVICES[x].get_all_tools(extra_exposed_attributes))
    
    # Remove duplicates while preserving order
    available_tools = list(dict.fromkeys(available_tools))

    # pick an appropriate response and generate the question
    if len(template_device_types) == 1:
        answer_template = get_random_response(
            service=service_names[0],
            persona=persona,
            question_template=question_template,
            short=False
        )

        question = question_template.replace("<device_name>", chosen_devices[0]["description"])
        answer_list = [ answer_template.replace("<device_name>", chosen_devices[0]["description"]) ]
    else:
        question = question_template
        answers = []
        for i in range(len(template_device_types)):
            question = question.replace(f"<device_name{(i + 1)}>", chosen_devices[i]["description"])
            answer_response = get_random_response(
                service=service_names[i],
                persona=persona,
                question_template=question_template,
                short=True
            )
            answers.append(answer_response.replace(f"<device_name>", chosen_devices[i]["description"]))

        answer_list = []
        for word in and_words:
            answer_list.append(f" {word} ".join(answers))

    # generate the list of tool calls
    tool_calls = []
    for device_dict, service in zip(chosen_devices, service_names):
        service_action = service.split(".")[1]
        tool_name = SERVICE_TO_TOOL_MAP.get(service_action, TOOL_TURN_ON)
        tool_call = {
            "tool_name": tool_name,
            "service_name": service,
            "tool_args": {"entity_id" if use_service_names else "name": device_dict["device_name"] if use_service_names else device_dict["description"]}
        }
        tool_calls.append(tool_call)

    if any(["climate" in service for service in service_names ]):
        if "<hvac_mode>" in question:
            hvac_mode = generate_random_parameter("hvac_mode")
            question = question.replace("<hvac_mode>", hvac_mode)
            answer_list = replace_answer(answer_list, "<hvac_mode>", hvac_mode)
            # Add hvac_mode as temperature parameter for climate tool
            for call in tool_calls:
                if call["tool_name"] == TOOL_CLIMATE_SET_TEMPERATURE:
                    call["tool_args"]["hvac_mode"] = hvac_mode

        if "<fan_mode>" in question:
            fan_mode = generate_random_parameter("fan_mode")
            question = question.replace("<fan_mode>", fan_mode)
            answer_list = replace_answer(answer_list, "<fan_mode>", fan_mode)
            for call in tool_calls:
                if call["tool_name"] == TOOL_CLIMATE_SET_TEMPERATURE:
                    call["tool_args"]["fan_mode"] = fan_mode

        if "<temp_f>" in question:
            temp_f = generate_random_parameter("temp_f")
            question = question.replace("<temp_f>", str(temp_f))
            answer_list = replace_answer(answer_list, "<temp_f>", str(temp_f))
            for call in tool_calls:
                if call["tool_name"] == TOOL_CLIMATE_SET_TEMPERATURE:
                    call["tool_args"]["temperature"] = temp_f

        if "<temp_c>" in question:
            temp_c = generate_random_parameter("temp_c")
            question = question.replace("<temp_c>", str(temp_c))
            answer_list = replace_answer(answer_list, "<temp_c>", str(temp_c))
            for call in tool_calls:
                if call["tool_name"] == TOOL_CLIMATE_SET_TEMPERATURE:
                    call["tool_args"]["temperature"] = temp_c

        if "<humidity>" in question:
            humidity = generate_random_parameter("humidity")
            question = question.replace("<humidity>", str(humidity))
            answer_list = replace_answer(answer_list, "<humidity>", str(humidity))
            for call in tool_calls:
                if call["tool_name"] == TOOL_SET_HUMIDITY:
                    call["tool_args"]["humidity"] = humidity

    if any(["light" in service for service in service_names ]):
        if "<brightness>" in question:
            brightness = generate_random_parameter("brightness")
            question = question.replace("<brightness>", str(brightness))
            answer_list = replace_answer(answer_list, "<brightness>", str(brightness))
            for call in tool_calls:
                if call["tool_name"] == TOOL_LIGHT_SET:
                    call["tool_args"]["brightness"] = brightness

        if "<color>" in question:
            random_rgb = generate_random_parameter("rgb_color")
            random_rgb_name = closest_color(random_rgb)
            question = question.replace("<color>", str(random_rgb_name))
            answer_list = replace_answer(answer_list, "<color>", str(random_rgb_name))
            for call in tool_calls:
                if call["tool_name"] == TOOL_LIGHT_SET:
                    call["tool_args"]["color"] = random_rgb_name

    if any(["timer" in service for service in service_names ]):
        if "<duration>" in question:
            duration = generate_random_parameter("duration")
            duration_name = pile_of_durations[duration]
            question = question.replace("<duration>", duration_name)
            answer_list = replace_answer(answer_list, "<duration>", duration_name)
            for call in tool_calls:
                if call["tool_name"] == TOOL_START_TIMER:
                    call["tool_args"]["duration"] = str(duration)

    if any(["todo" in service for service in service_names ]):
        if "<todo>" in question:
            todo = generate_random_parameter("todo")
            question = question.replace("<todo>", todo)
            answer_list = replace_answer(answer_list, "<todo>", todo)
            for call in tool_calls:
                if call["tool_name"] == TOOL_LIST_ADD_ITEM:
                    call["tool_args"]["item"] = todo

    return {
        "states": device_list,
        "available_tools": available_tools,
        "question": question.lower(),
        "answers": [ sentence.lower() for sentence in answer_list ],
        "tool_calls": tool_calls
    }

def generate_status_request(template: dict, persona: str, max_devices: int = 32, return_target_device: bool = False, use_service_names: bool = False):
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

    # gather a list of all available tools
    available_tools = []
    for x in set(device_types + [device_type]):
        available_tools.extend(SUPPORTED_DEVICES[x].get_all_tools(extra_exposed_attributes))
    
    # Remove duplicates while preserving order
    available_tools = list(dict.fromkeys(available_tools))

    result = {
        "states": device_list,
        "available_tools": available_tools,
        "question": question.lower(),
        "answers": [ answer.lower() ],
        "tool_calls": []
    }
    if return_target_device:
        return result, chosen_device
    else:
        return result

def format_example_sharegpt(example, persona, language, use_system_role, use_service_names):
    sys_prompt = pile_of_system_prompts[persona]
    random_datetime = generate_random_datetime()
    translate_datetime = babel.dates.format_datetime(random_datetime, BABEL_FORMAT[language], locale=BABEL_LOCALE[language])
    time_block = f"{CURRENT_DATE_PROMPT[language]} {translate_datetime}" 
    
    states_block = f"{DEVICES_PROMPT[language]}:\n" + "\n".join(example["states"])
    question = example["question"]
    answers = " ".join(example["answers"])

    # replace aliases with their actual values
    states_block = states_block.replace("blinds.", "cover.").replace("garage_door.", "cover.")

    # Build assistant message with content blocks
    assistant_content = []
    
    # Add text response
    assistant_content.append({
        "type": "text",
        "text": answers
    })
    
    # Add tool use blocks if there are tool calls
    if len(example["tool_calls"]) > 0:
        for tool_call in example["tool_calls"]:
            # Use service_name if in service mode, otherwise use tool_name
            call_name = tool_call.get("service_name", tool_call["tool_name"]) if use_service_names else tool_call["tool_name"]
            assistant_content.append({
                "type": "tool_use",
                "name": call_name,
                "parameters": tool_call["tool_args"]
            })

    if use_system_role:
        conversation = [
            { 
                "role": "system", 
                "content": "\n".join([ sys_prompt, time_block, states_block ])
            },
            { 
                "role": "user", 
                "content": question 
            },
            { 
                "role": "assistant", 
                "content": assistant_content
            },
        ]
    else:
        user_instruction_words = USER_INSTRUCTION_PROMPT[language] + ":"
        conversation = [
            { 
                "role": "user", 
                "content": "\n".join([ sys_prompt, time_block, states_block, user_instruction_words, question ]) 
            },
            { 
                "role": "assistant", 
                "content": assistant_content
            },
        ]
    
    return { 
        "conversations": conversation,
        "tools": HASS_TOOLS  # Include tools as a separate top-level key
    }

def generate_sft_file(filename: str, seed: int, format_func: Callable, use_system_role: bool, use_service_names: bool, personas: list[str], language: str, *, static_factor: float, template_factor: int, status_request_factor: int):
    random.seed(seed)
    np.random.seed(seed)

    print("Generating...")

    def run_factor_times(func, examples, data, persona, factor, language):
        if factor >= 1:
            for i in range(factor):
                examples.append(format_func(func(data, persona, use_service_names=use_service_names), persona, language, use_system_role, use_service_names))
        else:
            if random.random() < factor:
                examples.append(format_func(func(data, persona, use_service_names=use_service_names), persona, language, use_system_role, use_service_names))
    
    generated_examples = []

    missing_responses = set()

    for person in personas:
        for action in tqdm(pile_of_specific_actions):
            try:
                run_factor_times(generate_static_example, generated_examples, action, person, static_factor, language)
            except NoResponseAvailableException as ex:
                missing_responses.add(str(ex))

        for templated_action in tqdm(pile_of_templated_actions):
            try:
                run_factor_times(generate_templated_example, generated_examples, templated_action, person, template_factor, language)
            except NoResponseAvailableException as ex:
                missing_responses.add(str(ex))

    for status_request in tqdm(pile_of_status_requests):
        run_factor_times(generate_status_request, generated_examples, status_request, "assistant", status_request_factor, language)

    print(f"Generated {len(generated_examples)} examples. Saving...")

    for missing in sorted(missing_responses):
        print(missing)
    
    with open(f"{filename}.jsonl", "w") as f:
        for item in generated_examples:
            json_record = json.dumps(item)
            f.write(json_record + '\n')

    print("Done!")

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
        pile_of_system_prompts, pile_of_hallucinated_service_names, and_words
    
    with open(f"piles/{language}/pile_of_and_words.csv", encoding="utf8") as f:
        and_words = [ x.strip() for x in f.readlines() ]
    
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
def main(args=None):
    parser = argparse.ArgumentParser(description="Generate the full dataset from the CSV piles")
    parser.add_argument("--sample", action="store_true", help="Set this flag to enable generation of the train dataset.")
    parser.add_argument("--test", action="store_true", help="Set this flag to enable generation of the train dataset.")
    parser.add_argument("--train", action="store_true", help="Set this flag to enable generation of the train dataset.")
    parser.add_argument("--language", nargs="+", default=["english"], help="List of languages to generate: english, german, french, spanish, polish")
    parser.add_argument("--no-system-role", action="store_true", help="Set this flag to disable the system role. It will be combined with the user role")

    train_size_group = parser.add_mutually_exclusive_group()
    train_size_group.add_argument('--small', action='store_const', const='small', dest='size')
    train_size_group.add_argument('--medium', action='store_const', const='medium', dest='size')
    train_size_group.add_argument('--large', action='store_const', const='large', dest='size')
    train_size_group.add_argument('--xl', action='store_const', const='xl', dest='size')

    parser.add_argument('--use-service-names', action='store_true', 
                        help='Use service names (e.g., light.turn_on) instead of intent tool names (e.g., HassTurnOn)')

    args = parser.parse_args(args=args)

    if not args.sample and not args.train and not args.test and not args.merge:
        parser.print_usage()
        exit(-1)

    if args.size and not args.train:
        print("Train size was provided but not generating the training set!")
        exit(-1)
    
    format_func = format_example_sharegpt

    use_system_role = not args.no_system_role
    use_service_names = args.use_service_names

    for language in args.language:
        load_dataset_piles(language)
        personas = list(pile_of_system_prompts.keys())
        suffix = f"_{language}" if len(args.language) > 1 else ""

        if args.sample:
            generate_sft_file(f"sample{suffix}", 42, format_func, use_system_role, use_service_names, personas, language, static_factor=1, template_factor=1, status_request_factor=1)
        if args.train:
            if args.size == "small":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, use_system_role, use_service_names, personas, language, static_factor=1, template_factor=10, status_request_factor=8)
            elif args.size == "medium":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, use_system_role, use_service_names, personas, language, static_factor=5, template_factor=15, status_request_factor=12)
            elif args.size == "large":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, use_system_role, use_service_names, personas, language, static_factor=5, template_factor=20, status_request_factor=15)
            elif args.size == "xl":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, use_system_role, use_service_names, personas, language, static_factor=7, template_factor=25, status_request_factor=18)
            else:
                raise Exception(f"Unrecognized dataset size: {args.size}")
        if args.test:
            generate_sft_file(f"home_assistant_test{suffix}", 12345, format_func, use_system_role, use_service_names, personas, language, static_factor=0.25, template_factor=1, status_request_factor=2)

    if len(args.language) > 1:
        if args.sample:
            merge_languages("sample", args.language)
        if args.train:
            merge_languages("home_assistant_train", args.language)
        if args.test:
            merge_languages("home_assistant_test", args.language)

if __name__ == "__main__":
    main()
