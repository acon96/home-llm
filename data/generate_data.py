import argparse
import json
import numpy as np
import random
from datasets import load_dataset, concatenate_datasets
from typing import Any, Callable, TypedDict
from tqdm import tqdm
import webcolors

# ensure we can import from the data/ directory
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from devices import SUPPORTED_DEVICES, format_device_line, random_device_list, \
    TOOL_TURN_ON, TOOL_CLIMATE_SET_TEMPERATURE, TOOL_SET_HUMIDITY, \
    TOOL_LIGHT_SET, TOOL_START_TIMER, TOOL_LIST_ADD_ITEM, SERVICE_TO_TOOL_MAP, \
    HASS_TOOLS, SERVICE_TOOLS
from prompting import generate_system_prompt, USER_INSTRUCTION_PROMPT
from utils import PileOfDeviceType, PileOfFailedToolcallType, PileOfRefusalsType, PileOfSpecificActionType, PileOfStatusRequestType, PileOfTemplatedActionType, PileOfType, get_random_response, generate_random_parameter, closest_color, \
    get_dataset_piles, NoResponseAvailableException


class ToolCall(TypedDict):
    tool_name: str
    service_name: str
    tool_args: dict[str, Any]


class ToolResult(TypedDict):
    tool_name: str
    tool_result: str

class AssistantTurn(TypedDict):
    answer: str
    tool_call_sequence: list[ToolCall]
    tool_results: list[ToolResult]
    train_on_turn: bool


class Example(TypedDict):
    states: list[str]
    available_tools: list[str]
    question: str
    assistant_turns: list[AssistantTurn]


def create_assistant_turn(answer: str, tool_call_sequence: list[ToolCall] | None = None, *, tool_results: list[ToolResult] | None = None, train_on_turn: bool = True) -> AssistantTurn:
    """Bundle the assistant utterance with any tool interaction for that turn."""
    return {
        "answer": answer,
        "tool_call_sequence": tool_call_sequence or [],
        "tool_results": tool_results if tool_results is not None else [],
        "train_on_turn": train_on_turn,
    }

def generate_static_example(action: PileOfSpecificActionType, persona: str, language: str, max_devices: int = 128, use_service_names: bool = False) -> Example:
    question = action["phrase"]
    service_name = action["service_name"]
    device_type = service_name.split(".")[0]
    target_device = f"{device_type}.{action['device_name']}"
    friendly_name = target_device.split(".")[1].replace("_", " ").title()
    piles = get_dataset_piles(language)

    device_list, device_types, extra_exposed_attributes = random_device_list(
        max_devices=max_devices, avoid_device_names=[target_device], language=language)

    # insert our target device somewhere random in the list
    index = random.randint(0, len(device_list))
    state = SUPPORTED_DEVICES[device_type].get_random_state(language, extra_exposed_attributes=extra_exposed_attributes)

    device_list.insert(index, format_device_line(
        device_name=target_device,
        friendly_name=friendly_name,
        state=state
    ))

    # gather a list of all available tools
    available_tools: list[str] = []
    for x in set(device_types + [device_type]):
        available_tools.extend(SUPPORTED_DEVICES[x].get_all_tools(extra_exposed_attributes))
    
    # Remove duplicates while preserving order
    available_tools = list(dict.fromkeys(available_tools))

    # Map service name to tool name
    service_action = service_name.split(".")[1]
    tool_name = SERVICE_TO_TOOL_MAP[service_action]

    response_starting, response_confirmed = get_random_response(
        piles.pile_of_responses,
        service=service_name,
        persona=persona,
        question_template="",
        short=False
    )

    answer_list = [response_confirmed]
    tool_args = {}

    question = question.replace("<device_name>", target_device)
    response_starting = response_starting.replace("<device_name>", target_device)
    answer_list = replace_answer(answer_list, "<device_name>", target_device)

    if "climate" in service_action:
        if "<hvac_mode>" in question:
            hvac_mode = generate_random_parameter("hvac_mode", piles)
            question = question.replace("<hvac_mode>", hvac_mode)
            answer_list = replace_answer(answer_list, "<hvac_mode>", hvac_mode)
            # Add hvac_mode as temperature parameter for climate tool
            tool_args["hvac_mode"] = hvac_mode

        if "<fan_mode>" in question:
            fan_mode = generate_random_parameter("fan_mode", piles)
            question = question.replace("<fan_mode>", fan_mode)
            answer_list = replace_answer(answer_list, "<fan_mode>", fan_mode)
            tool_args["fan_mode"] = fan_mode

        if "<temp_f>" in question:
            temp_f = generate_random_parameter("temp_f", piles)
            question = question.replace("<temp_f>", str(temp_f))
            answer_list = replace_answer(answer_list, "<temp_f>", str(temp_f))
            tool_args["temperature"] = temp_f

        if "<temp_c>" in question:
            temp_c = generate_random_parameter("temp_c", piles)
            question = question.replace("<temp_c>", str(temp_c))
            answer_list = replace_answer(answer_list, "<temp_c>", str(temp_c))
            tool_args["temperature"] = temp_c

        if "<humidity>" in question:
            humidity = generate_random_parameter("humidity", piles)
            question = question.replace("<humidity>", str(humidity))
            answer_list = replace_answer(answer_list, "<humidity>", str(humidity))
            tool_args["humidity"] = humidity

    if "light" in service_action:
        if "<brightness>" in question:
            brightness = generate_random_parameter("brightness", piles)
            question = question.replace("<brightness>", str(brightness))
            answer_list = replace_answer(answer_list, "<brightness>", str(brightness))
            tool_args["brightness"] = brightness

        if "<color>" in question:
            random_rgb = generate_random_parameter("rgb_color", piles)
            random_rgb_name = closest_color(random_rgb)
            question = question.replace("<color>", str(random_rgb_name))
            answer_list = replace_answer(answer_list, "<color>", str(random_rgb_name))
            tool_args["color"] = random_rgb_name

    if "timer" in service_action:
        if "<duration>" in question:
            duration = generate_random_parameter("duration", piles)
            duration_name = piles.pile_of_durations[duration]
            question = question.replace("<duration>", duration_name)
            answer_list = replace_answer(answer_list, "<duration>", duration_name)
            tool_args["duration"] = str(duration)

    if "todo" in service_action:
        if "<todo>" in question:
            todo = generate_random_parameter("todo", piles)
            question = question.replace("<todo>", todo)
            answer_list = replace_answer(answer_list, "<todo>", todo)
            tool_args["item"] = todo

    if use_service_names:
        tool_call: ToolCall = {
            "tool_name": tool_name,
            "service_name": service_name,
            "tool_args": {"entity_id": target_device, **tool_args}
        }
    else:
        tool_call: ToolCall = {
            "tool_name": tool_name,
            "service_name": service_name,
            "tool_args": {"name": target_device, **tool_args}
        }

    if "arguments" in action and action["arguments"]:
        try:
            import json
            args = json.loads(action["arguments"])
            tool_call["tool_args"].update(args)
        except Exception as e:
            print(f"Failed to parse arguments for {action}: {e}")

    final_answer = " ".join(answer_list)
    assistant_turns = [
        create_assistant_turn(response_starting, [tool_call]),
        create_assistant_turn(final_answer, [])
    ]

    return {
        "states": device_list,
        "available_tools": available_tools,
        "question": question.lower(),
        "assistant_turns": assistant_turns
    }

def replace_answer(list_of_answer: list[str], var: str, value: str):
    new_list: list[str] = []
    for answer in list_of_answer:
        new_list.append(answer.replace(var, value))
    return new_list

def generate_templated_example(template: PileOfTemplatedActionType, persona: str, language: str, max_devices: int = 128, use_service_names: bool = False) -> Example:
    template_device_types: list[str] = template["device_type"].split("|")
    service_names: list[str] = [ f"{x}.{y}" for x, y in zip(template_device_types, template["service"].split("|")) ]
    question_template: str = template["phrase"]
    piles = get_dataset_piles(language)

    # choose a random device for this template
    chosen_devices: list[PileOfDeviceType] = []
    for device_type in template_device_types:
        device_dict = random.choice(piles.stacks_of_device_names[device_type])
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

        state = SUPPORTED_DEVICES[device_dict["type"]].get_random_state(language, extra_exposed_attributes=extra_exposed_attributes)
        device_name = device_dict["device_name"]
        friendly_name = device_dict["description"]

        device_list.insert(index, format_device_line(
            device_name=device_name,
            friendly_name=friendly_name,
            state=state
        ))

    # gather a list of all available tools
    available_tools: list[str] = []
    for x in set(device_types + template_device_types):
        available_tools.extend(SUPPORTED_DEVICES[x].get_all_tools(extra_exposed_attributes))
    
    # Remove duplicates while preserving order
    available_tools = list(dict.fromkeys(available_tools))

    # pick an appropriate response and generate the question
    if len(template_device_types) == 1:
        answer_starting, answer_confirmed = get_random_response(
            piles.pile_of_responses,
            service=service_names[0],
            persona=persona,
            question_template=question_template,
            short=False
        )

        question = question_template.replace("<device_name>", chosen_devices[0]["description"])
        answer_starting = answer_starting.replace("<device_name>", chosen_devices[0]["description"])
        answer_list = [ answer_confirmed.replace("<device_name>", chosen_devices[0]["description"]) ]
    else:
        question = question_template
        answers = []
        answer_starting = ""
        for i in range(len(template_device_types)):
            question = question.replace(f"<device_name{(i + 1)}>", chosen_devices[i]["description"])
            answer_starting_part, answer_confirmed = get_random_response(
                piles.pile_of_responses,
                service=service_names[i],
                persona=persona,
                question_template=question_template,
                short=True
            )
            answer_starting += answer_starting_part.replace(f"<device_name>", chosen_devices[i]["description"]) + " "
            answers.append(answer_confirmed.replace(f"<device_name>", chosen_devices[i]["description"]))

        answer_list = []
        for word in piles.and_words:
            answer_list.append(f" {word} ".join(answers))

    # generate the list of tool calls
    tool_calls: list[ToolCall] = []
    for device_dict, service in zip(chosen_devices, service_names):
        service_action = service.split(".")[1]
        tool_name = SERVICE_TO_TOOL_MAP[service_action]
        tool_call: ToolCall = {
            "tool_name": tool_name,
            "service_name": service,
            "tool_args": {"entity_id" if use_service_names else "name": device_dict["device_name"] if use_service_names else device_dict["description"]}
        }
        tool_calls.append(tool_call)

    if any(["climate" in service for service in service_names ]):
        if "<hvac_mode>" in question:
            hvac_mode = generate_random_parameter("hvac_mode", piles)
            question = question.replace("<hvac_mode>", hvac_mode)
            answer_list = replace_answer(answer_list, "<hvac_mode>", hvac_mode)
            # Add hvac_mode as temperature parameter for climate tool
            for call in tool_calls:
                if call["tool_name"] == TOOL_CLIMATE_SET_TEMPERATURE:
                    call["tool_args"]["hvac_mode"] = hvac_mode

        if "<fan_mode>" in question:
            fan_mode = generate_random_parameter("fan_mode", piles)
            question = question.replace("<fan_mode>", fan_mode)
            answer_list = replace_answer(answer_list, "<fan_mode>", fan_mode)
            for call in tool_calls:
                if call["tool_name"] == TOOL_CLIMATE_SET_TEMPERATURE:
                    call["tool_args"]["fan_mode"] = fan_mode

        if "<temp_f>" in question:
            temp_f = generate_random_parameter("temp_f", piles)
            question = question.replace("<temp_f>", str(temp_f))
            answer_list = replace_answer(answer_list, "<temp_f>", str(temp_f))
            for call in tool_calls:
                if call["tool_name"] == TOOL_CLIMATE_SET_TEMPERATURE:
                    call["tool_args"]["temperature"] = temp_f

        if "<temp_c>" in question:
            temp_c = generate_random_parameter("temp_c", piles)
            question = question.replace("<temp_c>", str(temp_c))
            answer_list = replace_answer(answer_list, "<temp_c>", str(temp_c))
            for call in tool_calls:
                if call["tool_name"] == TOOL_CLIMATE_SET_TEMPERATURE:
                    call["tool_args"]["temperature"] = temp_c

        if "<humidity>" in question:
            humidity = generate_random_parameter("humidity", piles)
            question = question.replace("<humidity>", str(humidity))
            answer_list = replace_answer(answer_list, "<humidity>", str(humidity))
            for call in tool_calls:
                if call["tool_name"] == TOOL_SET_HUMIDITY:
                    call["tool_args"]["humidity"] = humidity

    if any(["light" in service for service in service_names ]):
        if "<brightness>" in question:
            brightness = generate_random_parameter("brightness", piles)
            question = question.replace("<brightness>", str(brightness))
            answer_list = replace_answer(answer_list, "<brightness>", str(brightness))
            for call in tool_calls:
                if call["tool_name"] == TOOL_LIGHT_SET:
                    call["tool_args"]["brightness"] = brightness

        if "<color>" in question:
            random_rgb = generate_random_parameter("rgb_color", piles)
            random_rgb_name = closest_color(random_rgb)
            question = question.replace("<color>", str(random_rgb_name))
            answer_list = replace_answer(answer_list, "<color>", str(random_rgb_name))
            for call in tool_calls:
                if call["tool_name"] == TOOL_LIGHT_SET:
                    call["tool_args"]["color"] = random_rgb_name

    if any(["timer" in service for service in service_names ]):
        if "<duration>" in question:
            duration = generate_random_parameter("duration", piles)
            duration_name = piles.pile_of_durations[duration]
            question = question.replace("<duration>", duration_name)
            answer_list = replace_answer(answer_list, "<duration>", duration_name)
            for call in tool_calls:
                if call["tool_name"] == TOOL_START_TIMER:
                    call["tool_args"]["duration"] = str(duration)

    if any(["todo" in service for service in service_names ]):
        if "<todo>" in question:
            todo = generate_random_parameter("todo", piles)
            question = question.replace("<todo>", todo)
            answer_list = replace_answer(answer_list, "<todo>", todo)
            for call in tool_calls:
                if call["tool_name"] == TOOL_LIST_ADD_ITEM:
                    call["tool_args"]["item"] = todo

    starting_answer = answer_starting.strip().lower()
    normalized_answers = [ sentence.lower() for sentence in answer_list ]
    final_answer = " ".join(normalized_answers)
    assistant_turns = [
        create_assistant_turn(starting_answer, tool_calls),
        create_assistant_turn(final_answer, [])
    ]

    return {
        "states": device_list,
        "available_tools": available_tools,
        "question": question.lower(),
        "assistant_turns": assistant_turns
    }

def generate_status_request(template: PileOfStatusRequestType, persona: str, language: str, max_devices: int = 128, return_target_device: bool = False, use_service_names: bool = False) -> Example | tuple[Example, PileOfDeviceType]:
    device_type: str = template["device_type"]
    state_name: str = template["state"]
    question_template: str = template["phrase"]
    answer_template: str = template["assistant_response"]
    piles = get_dataset_piles(language)

    # choose a random device for this template
    chosen_device = random.choice(piles.stacks_of_device_names[device_type])

    # build a random list of devices
    device_list, device_types, extra_exposed_attributes = random_device_list(max_devices=max_devices, avoid_device_names=[ chosen_device["device_name"] ])

    # generate the question
    question = question_template.replace("<device_name>", chosen_device["description"])
    answer = answer_template.replace("<device_name>", chosen_device["description"])
    
    # insert other templated variables
    if device_type == "climate":
        climate_device_type = SUPPORTED_DEVICES["climate"]
        temp_f = climate_device_type.get_random_parameter("temp_f", language)
        answer = answer.replace("<temp_f>", str(temp_f))
        state_name = state_name.replace("<temp_f>", str(temp_f))

        temp_c = climate_device_type.get_random_parameter("temp_c", language)
        answer = answer.replace("<temp_c>", str(temp_c))
        state_name = state_name.replace("<temp_c>", str(temp_c))

        humidity = climate_device_type.get_random_parameter("humidity", language)
        answer = answer.replace("<humidity>", str(humidity))
        state_name = state_name.replace("<humidity>", str(humidity))

    if device_type == "light":
        light_device_type = SUPPORTED_DEVICES["light"]

        brightness = light_device_type.get_random_parameter("brightness", language)
        answer = answer.replace("<brightness>", str(brightness))
        state_name = state_name.replace("<brightness>", str(brightness))

        random_rgb = light_device_type.get_random_parameter("rgb_color", language)
        random_rgb_name = closest_color(random_rgb)
        actual_random_rgb = webcolors.name_to_rgb(random_rgb_name)
        actual_random_rgb = (actual_random_rgb.red, actual_random_rgb.green, actual_random_rgb.blue)
        state_name = state_name.replace("<color>", str(random_rgb_name) + " " + str(actual_random_rgb))
        answer = answer.replace("<color>", str(random_rgb_name))

    if device_type == "media_player":
        media_player_device_type = SUPPORTED_DEVICES["media_player"]
        volume = media_player_device_type.get_random_parameter("volume", language)
        random_media = media_player_device_type.get_random_parameter("media", language)

        answer = answer.replace("<volume>", str(volume) + "%")
        state_name = state_name.replace("<volume>", str(volume) + "%")

        answer = answer.replace("<media>", random_media)
        state_name = state_name.replace("<media>", random_media)

    if device_type == "timer":
        timer_device_type = SUPPORTED_DEVICES["timer"]
        duration = timer_device_type.get_random_parameter("duration", language)
        duration_name = piles.pile_of_durations[duration]
        remaining = timer_device_type.get_random_parameter("remaining", language)

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
    available_tools: list[str] = []
    for x in set(device_types + [device_type]):
        available_tools.extend(SUPPORTED_DEVICES[x].get_all_tools(extra_exposed_attributes))
    
    # Remove duplicates while preserving order
    available_tools = list(dict.fromkeys(available_tools))

    assistant_turns = [create_assistant_turn(answer.lower(), [])]

    result: Example = {
        "states": device_list,
        "available_tools": available_tools,
        "question": question.lower(),
        "assistant_turns": assistant_turns
    }
    if return_target_device:
        return result, chosen_device
    else:
        return result

def generate_tool_failure_example(failure_case: PileOfFailedToolcallType, persona: str, language: str, max_devices: int = 128, use_service_names: bool = False) -> Example:
    piles = get_dataset_piles(language)
    service_name = failure_case["service_name"]
    device_type = service_name.split(".")[0]
    service_action = service_name.split(".")[1]
    target_device = failure_case["correct_device_name"]
    friendly_name = failure_case.get("correct_friendly_name", target_device.split(".")[1].replace("_", " ").title())
    bad_device = failure_case["bad_device_name"]

    question_template = failure_case["phrase"]
    question = question_template.replace("<device_name>", friendly_name).lower()

    device_list, device_types, extra_exposed_attributes = random_device_list(
        max_devices=max_devices, avoid_device_names=[target_device], language=language)

    state = SUPPORTED_DEVICES[device_type].get_random_state(language, extra_exposed_attributes=extra_exposed_attributes)
    device_list.insert(random.randint(0, len(device_list)), format_device_line(
        device_name=target_device,
        friendly_name=friendly_name,
        state=state
    ))
    if device_type not in device_types:
        device_types.append(device_type)

    available_tools: list[str] = []
    for x in set(device_types):
        available_tools.extend(SUPPORTED_DEVICES[x].get_all_tools(extra_exposed_attributes))
    available_tools = list(dict.fromkeys(available_tools))

    response_starting, response_confirmed = get_random_response(
        piles.pile_of_responses,
        service=service_name,
        persona=persona,
        question_template=question_template,
        short=False
    )
    response_starting = response_starting.replace("<device_name>", friendly_name)
    response_confirmed = response_confirmed.replace("<device_name>", friendly_name)

    tool_args_extra: dict[str, Any] = {}
    if device_type == "climate":
        if "<temp_f>" in question or "<temp_f>" in response_starting or "<temp_f>" in response_confirmed:
            temp_f = generate_random_parameter("temp_f", piles)
            question = question.replace("<temp_f>", str(temp_f))
            response_starting = response_starting.replace("<temp_f>", str(temp_f))
            response_confirmed = response_confirmed.replace("<temp_f>", str(temp_f))
            tool_args_extra["temperature"] = temp_f
        if "<temp_c>" in question or "<temp_c>" in response_starting or "<temp_c>" in response_confirmed:
            temp_c = generate_random_parameter("temp_c", piles)
            question = question.replace("<temp_c>", str(temp_c))
            response_starting = response_starting.replace("<temp_c>", str(temp_c))
            response_confirmed = response_confirmed.replace("<temp_c>", str(temp_c))
            tool_args_extra["temperature"] = temp_c

    retry_prompt = failure_case.get("retry_prompt", f"Trying again with {friendly_name}.").replace("<device_name>", friendly_name)
    error_result = failure_case.get("error_result", "Error").replace("<device_name>", friendly_name)

    tool_name = SERVICE_TO_TOOL_MAP[service_action]
    first_args = {"entity_id": bad_device} if use_service_names else {"name": bad_device}
    retry_args = {"entity_id": target_device} if use_service_names else {"name": target_device}
    first_args.update(tool_args_extra)
    retry_args.update(tool_args_extra)

    first_turn = create_assistant_turn(
        response_starting,
        [{
            "tool_name": tool_name,
            "service_name": service_name,
            "tool_args": first_args
        }],
        tool_results=[{
            "tool_name": service_name if use_service_names else tool_name,
            "tool_result": error_result
        }],
        train_on_turn=False
    )

    second_turn = create_assistant_turn(
        retry_prompt,
        [{
            "tool_name": tool_name,
            "service_name": service_name,
            "tool_args": retry_args
        }]
    )

    final_turn = create_assistant_turn(response_confirmed, [])

    return {
        "states": device_list,
        "available_tools": available_tools,
        "question": question,
        "assistant_turns": [first_turn, second_turn, final_turn]
    }

def generate_refusal_example(refusal_case: PileOfRefusalsType, persona: str, language: str, max_devices: int = 128, use_service_names: bool = False) -> Example:
    service_name = refusal_case["service_name"]
    device_type = service_name.split(".")[0]
    target_device = f"{device_type}.{refusal_case['device_name']}"
    friendly_name = refusal_case.get("friendly_name", refusal_case["device_name"].replace("_", " ").title())
    desired_state = refusal_case.get("desired_state", "")
    reason_type = refusal_case.get("reason_type", "not_available")

    device_list, device_types, extra_exposed_attributes = random_device_list(
        max_devices=max_devices, avoid_device_names=[target_device], language=language)

    if reason_type == "already_state":
        state = desired_state if desired_state else SUPPORTED_DEVICES[device_type].possible_states[0][0]
        device_list.insert(random.randint(0, len(device_list)), format_device_line(
            device_name=target_device,
            friendly_name=friendly_name,
            state=state
        ))
        if device_type not in device_types:
            device_types.append(device_type)

    available_tools: list[str] = []
    for x in set(device_types):
        available_tools.extend(SUPPORTED_DEVICES[x].get_all_tools(extra_exposed_attributes))
    available_tools = list(dict.fromkeys(available_tools))

    response_text = refusal_case["response"].replace("<device_name>", friendly_name).lower()
    question = refusal_case["phrase"].replace("<device_name>", friendly_name).lower()

    assistant_turns = [create_assistant_turn(response_text, [])]

    return {
        "states": device_list,
        "available_tools": available_tools,
        "question": question,
        "assistant_turns": assistant_turns
    }

def format_example_sharegpt(example, persona, language, use_system_role, append_user_instruction_prompt, use_service_names, tool_response_format):
    piles = get_dataset_piles(language)
    sys_prompt = generate_system_prompt(example, persona, language, piles.pile_of_system_prompts)
    question = example["question"]
    assistant_turns = example["assistant_turns"]

    if append_user_instruction_prompt:
        user_instruction_words = USER_INSTRUCTION_PROMPT[language] + ":"
        sys_prompt = "\n".join([ sys_prompt, user_instruction_words ])

    if use_system_role:
        conversation = [
            { 
                "role": "system", 
                "content": [{"type": "text", "text": sys_prompt}],
                "train_on_turn": False,
            },
            { 
                "role": "user", 
                "content": [{ "type": "text", "text": question }],
                "train_on_turn": False,
            }
        ]
    else:
        conversation = [
            { 
                "role": "user", 
                "content": [{ "type": "text", "text": "\n".join([ sys_prompt, question ]) }],
                "train_on_turn": False,
            }
        ]
    
    call_id_counter = 1
    for turn in assistant_turns:
        answer_text = turn.get("answer", "")
        assistant_block = {
            "role": "assistant",
            "content": [{ "type": "text", "text": answer_text }],
            "train_on_turn": turn.get("train_on_turn", True),
        }

        tool_call_sequence = turn.get("tool_call_sequence", [])
        formatted_calls = []
        call_names = []
        for tool_call in tool_call_sequence:
            call_name = tool_call.get("service_name", tool_call["tool_name"]) if use_service_names else tool_call["tool_name"]
            call_names.append(call_name)
            formatted_calls.append({
                "name": call_name,
                "arguments": json.dumps(tool_call["tool_args"]),
            })

        if formatted_calls:
            assistant_block["tool_calls"] = [{ "function": call } for call in formatted_calls]

        conversation.append(assistant_block)

        if formatted_calls:
            provided_results = turn.get("tool_results") or []
            step_tool_results = []

            if provided_results:
                for idx, provided in enumerate(provided_results):
                    result = dict(provided)
                    if "tool_name" not in result and call_names:
                        result["tool_name"] = call_names[min(idx, len(call_names) - 1)]
                    if "tool_call_id" not in result:
                        result["tool_call_id"] = f"call_{call_id_counter}"
                    call_id_counter += 1
                    step_tool_results.append(result)
            else:
                for call_name in call_names:
                    step_tool_results.append({
                        "tool_name": call_name,
                        "tool_call_id": f"call_{call_id_counter}",
                        "tool_result": "Success"
                    })
                    call_id_counter += 1

            if tool_response_format == "text":
                conversation.append({
                    "role": "tool",
                    "content": [{ "type": "text", "text": json.dumps(result) } for result in step_tool_results],
                    "train_on_turn": False,
                })
            elif tool_response_format == "functiongemma":
                conversation.append({
                    "role": "tool",
                    "content": [{ "name": result["tool_name"], "response": {"result": result["tool_result"]} } for result in step_tool_results],
                    "train_on_turn": False,
                })
    
    return { 
        "messages": conversation,
        "tools": SERVICE_TOOLS if use_service_names else HASS_TOOLS
    }

def generate_sft_file(
        filename: str,
        seed: int,
        format_func: Callable,
        use_system_role: bool,
        append_user_instruction_prompt: bool,
        use_service_names: bool,
        personas: list[str],
        language: str,
        tool_response_format: str,
        *,
        static_factor: float,
        template_factor: int,
        status_request_factor: int,
        failure_factor: int,
        refusal_factor: int):
    random.seed(seed)
    np.random.seed(seed)
    piles = get_dataset_piles(language)

    print("Generating...")

    def run_factor_times(func: Callable[..., Example], examples: list[Example], data, persona: str, factor: int | float, language: str):
        if factor >= 1:
            for i in range(int(factor)):
                examples.append(format_func(func(data, persona, language, use_service_names=use_service_names), persona, language, use_system_role, append_user_instruction_prompt, use_service_names, tool_response_format))
        else:
            if random.random() < factor:
                examples.append(format_func(func(data, persona, language, use_service_names=use_service_names), persona, language, use_system_role, append_user_instruction_prompt, use_service_names, tool_response_format))
    
    generated_examples: list[Example] = []

    missing_responses = set()

    for person in personas:
        for action in tqdm(piles.pile_of_specific_actions):
            try:
                run_factor_times(generate_static_example, generated_examples, action, person, static_factor, language)
            except NoResponseAvailableException as ex:
                missing_responses.add(str(ex))

        for templated_action in tqdm(piles.pile_of_templated_actions):
            try:
                run_factor_times(generate_templated_example, generated_examples, templated_action, person, template_factor, language)
            except NoResponseAvailableException as ex:
                missing_responses.add(str(ex))

        for failure_case in tqdm(piles.pile_of_failed_tool_calls):
            try:
                run_factor_times(generate_tool_failure_example, generated_examples, failure_case, person, failure_factor, language)
            except NoResponseAvailableException as ex:
                missing_responses.add(str(ex))

        for refusal_case in tqdm(piles.pile_of_refusals):
            try:
                run_factor_times(generate_refusal_example, generated_examples, refusal_case, person, refusal_factor, language)
            except NoResponseAvailableException as ex:
                missing_responses.add(str(ex))

    for status_request in tqdm(piles.pile_of_status_requests):
        run_factor_times(generate_status_request, generated_examples, status_request, "assistant", status_request_factor, language)

    print(f"Generated {len(generated_examples)} examples. Saving...")

    for missing in sorted(missing_responses):
        print(missing)
    
    cwd = os.path.dirname(__file__)
    with open(f"{cwd}/output/{filename}.jsonl", "w") as f:
        for item in generated_examples:
            json_record = json.dumps(item)
            f.write(json_record + '\n')

    print("Done!")

def merge_with_dataset(dataset_name, seed, output_name, format_function, dataset_column_names, format_func):
    alpaca_dataset = load_dataset(dataset_name)["train"].train_test_split(test_size=0.1)
    home_assistant_dataset = load_dataset("json", data_files={ "train": "home_assistant_train.jsonl", "test": "home_assistant_test.jsonl" })

    random.seed(seed)
    np.random.seed(seed)

    alpaca_dataset = alpaca_dataset.map(format_function).remove_columns(dataset_column_names)

    combined_dataset_train = concatenate_datasets([home_assistant_dataset["train"], alpaca_dataset["train"]]).shuffle(seed=42)
    combined_dataset_test = concatenate_datasets([home_assistant_dataset["test"], alpaca_dataset["test"]]).shuffle(seed=42)

    combined_dataset_train.to_json(f"home_assistant_{output_name}_merged_train.jsonl")
    combined_dataset_test.to_json(f"home_assistant_{output_name}_merged_test.jsonl")

def merge_languages(filename_prefix: str, languages: list):
    all_examples = []
    cwd = os.path.dirname(__file__)

    for language in languages:
        with open(f"{cwd}/output/{filename_prefix}_{language}.jsonl") as f:
            all_examples.extend(f.readlines())

    with open(f"{cwd}/output/{filename_prefix}.jsonl", "w") as f:
        f.writelines(all_examples)


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
    parser.add_argument("--tool-response-format", default="text", choices=["text", "functiongemma"], help="Format to use for tool responses.")

    role_tweaks = parser.add_mutually_exclusive_group()
    role_tweaks.add_argument("--no-system-role", action="store_true", help="Set this flag to disable the system role. The house context will be combined with the user role")
    role_tweaks.add_argument("--merged-system-role", action="store_true", help="Set this flag to still emit a system role, but assume it will be merged by the chat template into the user role.")

    train_size_group = parser.add_mutually_exclusive_group()
    train_size_group.add_argument('--small', action='store_const', const='small', dest='size')
    train_size_group.add_argument('--medium', action='store_const', const='medium', dest='size')
    train_size_group.add_argument('--large', action='store_const', const='large', dest='size')
    train_size_group.add_argument('--xl', action='store_const', const='xl', dest='size')

    parser.add_argument('--use-service-names', action='store_true', 
                        help='Use service names (e.g., light.turn_on) instead of intent tool names (e.g., HassTurnOn)')

    args = parser.parse_args(args=args)

    if not args.sample and not args.train and not args.test:
        parser.print_usage()
        exit(-1)

    if args.size and not args.train:
        print("Train size was provided but not generating the training set!")
        exit(-1)
    
    format_func = format_example_sharegpt

    use_system_role = not args.no_system_role
    append_user_instruction_prompt = args.merged_system_role or not args.no_system_role
    use_service_names = args.use_service_names
    tool_response_format = args.tool_response_format

    for language in args.language:
        piles = get_dataset_piles(language)
        personas = list(piles.pile_of_system_prompts.keys())
        suffix = f"_{language}" if len(args.language) > 1 else ""

        if args.sample:
            generate_sft_file(f"sample{suffix}", 42, format_func, use_system_role, append_user_instruction_prompt, use_service_names, personas, language, tool_response_format, static_factor=1, template_factor=1, status_request_factor=1, refusal_factor=1, failure_factor=1)
        if args.train:
            if args.size == "small":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, use_system_role, append_user_instruction_prompt, use_service_names, personas, language, tool_response_format, static_factor=1, template_factor=10, status_request_factor=8, refusal_factor=3, failure_factor=1)
            elif args.size == "medium":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, use_system_role, append_user_instruction_prompt, use_service_names, personas, language, tool_response_format, static_factor=5, template_factor=15, status_request_factor=12, refusal_factor=5, failure_factor=1)
            elif args.size == "large":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, use_system_role, append_user_instruction_prompt, use_service_names, personas, language, tool_response_format, static_factor=5, template_factor=20, status_request_factor=15, refusal_factor=6, failure_factor=1)
            elif args.size == "xl":
                generate_sft_file(f"home_assistant_train{suffix}", 42, format_func, use_system_role, append_user_instruction_prompt, use_service_names, personas, language, tool_response_format, static_factor=7, template_factor=25, status_request_factor=18, refusal_factor=8, failure_factor=2)
            else:
                raise Exception(f"Unrecognized dataset size: {args.size}")
        if args.test:
            generate_sft_file(f"home_assistant_test{suffix}", 12345, format_func, use_system_role, append_user_instruction_prompt, use_service_names, personas, language, tool_response_format, static_factor=0.25, template_factor=1, status_request_factor=2, refusal_factor=1, failure_factor=1)
    if len(args.language) > 1:
        if args.sample:
            merge_languages("sample", args.language)
        if args.train:
            merge_languages("home_assistant_train", args.language)
        if args.test:
            merge_languages("home_assistant_test", args.language)

if __name__ == "__main__":
    main()
