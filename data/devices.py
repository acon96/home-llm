from collections import defaultdict
import random
from dataclasses import dataclass
from typing import Final, Callable, List
from difflib import SequenceMatcher

from tools import *
from utils import PileOfDeviceType, closest_color, generate_random_parameter, get_dataset_piles

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

def format_device_line(*, device_name: str, friendly_name: str, state: str):
    return (f"{device_name} '{friendly_name}' = {state}")

@dataclass
class DeviceType:
    name: str
    possible_states: list[tuple[str, float]]

    def get_random_state(self, language: str, extra_exposed_attributes: list[str] | None = None):
        states = [ x[0] for x in self.possible_states ]
        weights = [ x[1] for x in self.possible_states ]
        return random.choices(states, weights=weights, k=1)[0]
    
    def get_all_tools(self, extra_exposed_attributes: List[str]):
        """Return list of tool names available for this device type."""
        tools = [TOOL_TURN_ON, TOOL_TURN_OFF, TOOL_TOGGLE]
        return tools
    
    def get_random_parameter(self, param_name: str, language: str):
        """Generate a random parameter value."""
        return generate_random_parameter(param_name, get_dataset_piles(language))


class LightDeviceType(DeviceType):
    def __init__(self):
        super().__init__("light",
            possible_states=[
                (STATE_ON, 0.5),
                (STATE_OFF, 0.5)
            ]
        )

    def get_random_state(self, language: str, extra_exposed_attributes: list[str] | None = None):
        extra_exposed_attributes = extra_exposed_attributes or []
        state = super().get_random_state(language, extra_exposed_attributes=extra_exposed_attributes)

        if random.random() < 0.5 and "rgb_color" in extra_exposed_attributes:
            random_rgb = generate_random_parameter("rgb_color", get_dataset_piles(language))
            state = state + ";" + closest_color(random_rgb) + " " + str(random_rgb)

        if random.random() < 0.7 and "brightness" in extra_exposed_attributes:
            state = state + ";" + str(generate_random_parameter("brightness", get_dataset_piles(language))) + "%"
        return state
    
    def get_all_tools(self, extra_exposed_attributes: List[str]):
        """Return list of tool names available for lights."""
        tools = [TOOL_TURN_ON, TOOL_TURN_OFF, TOOL_TOGGLE]
        if "brightness" in extra_exposed_attributes or "rgb_color" in extra_exposed_attributes:
            tools.append(TOOL_LIGHT_SET)
        return tools
    
class SwitchDeviceType(DeviceType):
    def __init__(self):
        super().__init__("switch", [
            (STATE_ON, 0.5),
            (STATE_OFF, 0.5)
        ])

class FanDeviceType(DeviceType):
    def __init__(self):
        super().__init__("fan", [
            (STATE_ON, 0.5),
            (STATE_OFF, 0.5)
        ])

class GarageDoorDeviceType(DeviceType):
    def __init__(self):
        super().__init__("garage_door", [
            (STATE_OPEN, 0.49),
            (STATE_CLOSED, 0.49),
            (STATE_OPENING, 0.01),
            (STATE_CLOSING, 0.01)
        ])

    def get_all_tools(self, extra_exposed_attributes: List[str]):
        tools = [TOOL_TURN_ON, TOOL_TURN_OFF, TOOL_TOGGLE]
        if "position" in extra_exposed_attributes:
            tools.append(TOOL_SET_POSITION)
        return tools

class BlindsDeviceType(DeviceType):
    def __init__(self):
        super().__init__("blinds", [
            (STATE_OPEN, 0.49),
            (STATE_CLOSED, 0.49),
            (STATE_OPENING, 0.01),
            (STATE_CLOSING, 0.01)
        ])
    
    def get_all_tools(self, extra_exposed_attributes: List[str]):
        tools = [TOOL_TURN_ON, TOOL_TURN_OFF, TOOL_TOGGLE]
        if "position" in extra_exposed_attributes:
            tools.append(TOOL_SET_POSITION)
        return tools

class LockDeviceType(DeviceType):
    def __init__(self):
        super().__init__("lock", [
            (STATE_LOCKED, 0.5),
            (STATE_UNLOCKED, 0.5),
        ])

class VacuumDeviceType(DeviceType):
    def __init__(self):
        super().__init__("vacuum", [
            (STATE_CLEANING, 0.2),
            (STATE_DOCKED, 0.6),
            (STATE_RETURNING, 0.1),
            (STATE_IDLE, 0.05),
            (STATE_PAUSED, 0.05),
        ])

    def get_all_tools(self, extra_exposed_attributes: List[str]):
        return [TOOL_VACUUM_START, TOOL_VACUUM_RETURN_TO_BASE]

class TimerDeviceType(DeviceType):
    def __init__(self):
        super().__init__("timer", [
            (STATE_IDLE, 0.2),
            (STATE_ACTIVE, 0.6),
            (STATE_PAUSED, 0.1),
        ])

    def get_all_tools(self, extra_exposed_attributes: List[str]):
        tools = [TOOL_START_TIMER, TOOL_CANCEL_TIMER, TOOL_PAUSE_TIMER, TOOL_UNPAUSE_TIMER]
        if "duration" in extra_exposed_attributes:
            tools.extend([TOOL_INCREASE_TIMER, TOOL_DECREASE_TIMER, TOOL_TIMER_STATUS])
        return tools

class TodoDeviceType(DeviceType):
    def __init__(self):
        super().__init__("todo", [ (f"{i}", (1/32)) for i in range(32) ],)

    def get_all_tools(self, extra_exposed_attributes: List[str]):
        return [TOOL_LIST_ADD_ITEM]
    
class ClimateDeviceType(DeviceType):
    def __init__(self):
        super().__init__("climate", [])

    def get_random_state(self, language: str, extra_exposed_attributes: list[str] | None = None):
        """state;fan_mode;temperature;humidity"""
        extra_exposed_attributes = extra_exposed_attributes or []
        state = generate_random_parameter("hvac_mode", get_dataset_piles(language))

        if "fan_mode" in extra_exposed_attributes:
            state = state  + ";" + generate_random_parameter("fan_mode", get_dataset_piles(language))
        if "temperature" in extra_exposed_attributes:
            if random.random() > 0.5:
                state = state + ";" + str(generate_random_parameter("temp_f", get_dataset_piles(language))) + "F" 
            else:
                state = state + ";" + str(generate_random_parameter("temp_c", get_dataset_piles(language))) + "C"
        if "humidity" in extra_exposed_attributes:
            state = state + ";" + str(generate_random_parameter("humidity", get_dataset_piles(language))) + "%"
        if random.random() < 0.8 and "preset_mode" in extra_exposed_attributes:
            # if it is not "on a preset" then don't add the mode
            state = state + ";" + generate_random_parameter("preset_mode", get_dataset_piles(language))

        return state
    
    def get_all_tools(self, extra_exposed_attributes: List[str]):
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

    def get_random_state(self, language: str, extra_exposed_attributes: list[str] | None = None):
        extra_exposed_attributes = extra_exposed_attributes or []
        state = super().get_random_state(language, extra_exposed_attributes=extra_exposed_attributes)

        if "media_title" in extra_exposed_attributes and state in [STATE_PLAYING, STATE_PAUSED, STATE_BUFFERING, STATE_ON]:
            state = state + ";" + generate_random_parameter("media", get_dataset_piles(language))
        if "volume_level" in extra_exposed_attributes and state != STATE_OFF:
            state = state + ";vol=" + str(generate_random_parameter("volume", get_dataset_piles(language))) + "%"
        return state

    def get_all_tools(self, extra_exposed_attributes: List[str]):
        """Return list of tool names available for media players."""
        tools = [TOOL_TURN_ON, TOOL_TURN_OFF, TOOL_MEDIA_PAUSE, TOOL_MEDIA_UNPAUSE, TOOL_MEDIA_NEXT]
        if "volume_level" in extra_exposed_attributes:
            tools.append(TOOL_SET_VOLUME)
        return tools


SUPPORTED_DEVICES: dict[str, DeviceType] = {
    "light": LightDeviceType(),
    "switch": SwitchDeviceType(),
    "fan": FanDeviceType(),
    "garage_door": GarageDoorDeviceType(),
    "blinds": BlindsDeviceType(),
    "lock": LockDeviceType(),
    "media_player": MediaPlayerDeviceType(),
    "climate": ClimateDeviceType(),
    "vacuum": VacuumDeviceType(),
    "timer": TimerDeviceType(),
    "todo": TodoDeviceType(),
}

# generate a random list of devices for the context
def random_device_list(max_devices: int, avoid_device_names: list[str], language: str = "english"):
    num_devices = random.randint(2, max_devices)
    piles = get_dataset_piles(language)

    local_device_names = { k: v[:] for k,v in piles.stacks_of_device_names.items() }

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

    possible_choices: list[PileOfDeviceType] = []
    for device_type in local_device_names.keys():
        possible_choices.extend(local_device_names[device_type])
    

    device_types = set()
    device_list = []
    device_lines: list[str] = []
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

            state = SUPPORTED_DEVICES[device_type].get_random_state(language, extra_exposed_attributes=extra_exposed_attributes)
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
