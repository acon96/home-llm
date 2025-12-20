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
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
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
    }},
    {"function": {
        "name": TOOL_CANCEL_TIMER,
        "description": "Cancels a timer",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the timer"}
            },
            "required": []
        }
    }},
    {"function": {
        "name": TOOL_PAUSE_TIMER,
        "description": "Pauses a timer",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the timer"}
            },
            "required": []
        }
    }},
    {"function": {
        "name": TOOL_UNPAUSE_TIMER,
        "description": "Resumes a paused timer",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the timer"}
            },
            "required": []
        }
    }}
]

SERVICE_TOOL_ALLOWED_SERVICES = ["turn_on", "turn_off", "toggle", "press", "increase_speed", "decrease_speed", "open_cover", "close_cover", "stop_cover", "lock", "unlock",
                                "start", "stop", "return_to_base", "pause", "cancel", "add_item", "set_temperature", "set_humidity", "set_fan_mode", "set_hvac_mode", "set_preset_mode"]
SERVICE_TOOL_ALLOWED_DOMAINS = ["light", "switch", "button", "fan", "cover", "lock", "media_player", "climate", "vacuum", "todo", "timer", "script"]

SERVICE_TOOLS = [
    {"function": { 
        "name": "<sample>",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to turn on"},
                "rgb_color": {"type": "string", "description": "The RGB color to set"},
                "brightness": {"type": "number", "description": "The brightness level"},
                "temperature": {"type": "number", "description": "The temperature level"},
                "humidity": {"type": "number", "description": "The humidity level"},
                "fan_mode": {"type": "string", "description": "The fan mode"},
                "hvac_mode": {"type": "string", "description": "The HVAC mode"},
                "preset_mode": {"type": "string", "description": "The preset mode"},
                "duration": {"type": "string", "description": "The amount of time to apply to the chosen timer"},
                "item": {"type": "string", "description": "The item to add to the list"}
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "light.turn_on",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to turn on"},
                "rgb_color": {"type": "string", "description": "The RGB color to set"},
                "brightness": {"type": "number", "description": "The brightness level"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "light.turn_off",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to turn off"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "light.toggle",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to toggle"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "switch.turn_on",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to turn on"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "switch.turn_off",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to turn off"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "switch.toggle",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to toggle"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "fan.turn_on",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to turn on"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "fan.turn_off",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to turn off"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "fan.toggle",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to toggle"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "fan.set_speed",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to set speed"},
                "fan_mode": {"type": "string", "description": "The fan mode"},
            },
            "required": [
                "target_device",
                "fan_mode"
            ]
        }
    }},
    {"function": {
        "name": "fan.increase_speed",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to increase speed"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "fan.decrease_speed",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to decrease speed"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "button.press",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to press"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "cover.open_cover",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to open"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "cover.close_cover",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to close"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "cover.stop_cover",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to stop"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "cover.set_cover_position",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to set position"},
                "position": {"type": "integer", "description": "Position from 0-100", "minimum": 0, "maximum": 100}
            },
            "required": [
                "target_device",
                "position"
            ]
        }
    }},
    {"function": {
        "name": "lock.unlock",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to unlock"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "lock.lock",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to lock"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "vacuum.start",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to start"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "vacuum.stop",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to stop"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "vacuum.return_to_base",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target device to return to base"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "media_player.media_play_pause",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target media player to play/pause"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "media_player.media_pause",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target media player to pause"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "media_player.media_play",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target media player to play"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "media_player.media_next_track",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target media player to skip to next track"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "media_player.media_previous_track",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target media player to skip to previous track"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "media_player.volume_set",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target media player to set volume"},
                "volume_level": {"type": "number", "description": "Volume level (0.0-1.0)", "minimum": 0.0, "maximum": 1.0}
            },
            "required": [
                "target_device",
                "volume_level"
            ]
        }
    }},
    {"function": {
        "name": "todo.add_item",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target todo list to add item to"},
                "item": {"type": "string", "description": "The item to add to the list"}
            },
            "required": [
                "target_device",
                "item"
            ]
        }
    }},
    {"function": {
        "name": "timer.start",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target timer to start"},
                "duration": {"type": "string", "description": "Timer duration (HH:MM:SS format)"}
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "timer.cancel",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target timer to cancel"},
            },
            "required": [
                "target_device"
            ]
        }
    }},
    {"function": {
        "name": "climate.set_temperature",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target climate device to set temperature"},
                "temperature": {"type": "number", "description": "Target temperature"}
            },
            "required": [
                "target_device",
                "temperature"
            ]
        }
    }},
    {"function": {
        "name": "climate.set_humidity",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target humidifier device to set humidity"},
                "humidity": {"type": "integer", "description": "Target humidity percentage (0-100)", "minimum": 0, "maximum": 100}
            },
            "required": [
                "target_device",
                "humidity"
            ]
        }
    }},
    {"function": {
        "name": "climate.set_hvac_mode",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target climate device to set HVAC mode"},
                "hvac_mode": {"type": "string", "description": "The HVAC mode"}
            },
            "required": [
                "target_device",
                "hvac_mode"
            ]
        }
    }},
    {"function": {
        "name": "climate.set_preset_mode",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target climate device to set preset mode"},
                "preset_mode": {"type": "string", "description": "The preset mode"}
            },
            "required": [
                "target_device",
                "preset_mode"
            ]
        }
    }},
    {"function": {
        "name": "climate.set_fan_mode",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "target_device": {"type": "string", "description": "The target climate device to set fan mode"},
                "fan_mode": {"type": "string", "description": "The fan mode"}
            },
            "required": [
                "target_device",
                "fan_mode"
            ]
        }
    }}
]