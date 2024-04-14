"""Original script by @BramNH on GitHub"""
import csv
import os
import time
import re

from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests
from tqdm import tqdm

SUPPORTED_DEVICES = [
    "light",
    "switch",
    "media_player",
    "climate",
    "vacuum",
    "todo",
    "blinds",
    "fan",
    "garage_door",
    "lock",
    "timer",
]

def format_device_name(input_str):
    return input_str.replace('-', '_').replace(' ', '_').lower()

class DatasetTranslator:
    translator: GoogleTranslator
    source_language: str
    target_language: str

    def __init__(self, source_language, target_language):
        self.source_language = source_language
        self.target_language = target_language
        self.translator = GoogleTranslator(source=source_language, target=target_language)

    def translate_all_piles(self):
        os.makedirs(f"./piles/{self.target_language}", exist_ok=True)

        self.translate_device_names()
        self.translate_templated_actions()
        self.translate_specific_actions()
        self.translate_status_requests()
        self.translate_durations()
        self.translate_responses()
        self.translate_system_prompts()
        self.translate_todo_items()

    def translate(self, phrase_to_translate) -> str:
        try:
            translated_phrase = self.translator.translate(phrase_to_translate, return_all=False)
        except TooManyRequests:
            time.sleep(5.0)
            translated_phrase = self.translator.translate(phrase_to_translate, return_all=False)

        # All <device_name> blocks are also translated,
        # so place them back in english after translation.
        res = re.findall(r"\<.*?\>", phrase_to_translate)
        i = len(res) - 1
        for match in reversed(list(re.finditer(r"\<.*?\>", translated_phrase))):
            loc = match.span()
            translated_phrase = translated_phrase.replace(
                translated_phrase[loc[0] : loc[1]], res[i], 1
            )
            i -= 1
        return translated_phrase

    def translate_device_names(self):
        print("Translating device names")
        pile_of_device_names_target = list()

        if os.path.exists(f"piles/{self.target_language}/pile_of_device_names.csv"):
            print("csv already exists! delete it to re-translate")
            return

        with open(f"piles/{self.source_language}/pile_of_device_names.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            pile_of_device_names = list(reader)
            pile_of_device_names = list(
                filter(
                    lambda x: x["device_name"].split(".")[0] in SUPPORTED_DEVICES,
                    pile_of_device_names,
                )
            )
            for device_dict in tqdm(pile_of_device_names):
                try:
                    device_type, device_name = device_dict["device_name"].split(".")
                    device_description = device_dict["description"]
                    pile_of_device_names_target.append(
                        {
                            "device_name": f"{device_type}.{format_device_name(self.translate(device_name))}",
                            "description": self.translate(device_description),
                        }
                    )
                except KeyError as ex:
                    print(ex)
        with open(f"piles/{self.target_language}/pile_of_device_names.csv", "w+", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["device_name", "description"])
            writer.writeheader()
            writer.writerows(pile_of_device_names_target)


    def translate_templated_actions(self):
        print("Translating templated actions")
        pile_of_templated_actions_target = list()

        if os.path.exists(f"piles/{self.target_language}/pile_of_templated_actions.csv"):
            print("csv already exists! delete it to re-translate")
            return

        with open(f"piles/{self.source_language}/pile_of_templated_actions.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            pile_of_templated_actions = list(reader)
            pile_of_templated_actions = list(
                filter(
                    lambda x: [t in SUPPORTED_DEVICES for t in x["device_type"].split("|")],
                    pile_of_templated_actions,
                )
            )
            for actions_dict in tqdm(pile_of_templated_actions):
                try:
                    pile_of_templated_actions_target.append(
                        {
                            "device_type": actions_dict["device_type"],
                            "service": actions_dict["service"],
                            "phrase": self.translate(actions_dict["phrase"]),
                            "multiplier": actions_dict["multiplier"],
                        }
                    )
                except KeyError as ex:
                    print(ex)
        with open(f"piles/{self.target_language}/pile_of_templated_actions.csv", "w+", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["device_type", "service", "phrase", "multiplier"]
            )
            writer.writeheader()
            writer.writerows(pile_of_templated_actions_target)


    def translate_specific_actions(self):
        print("Translating specific actions")
        pile_of_specific_actions_target = list()

        if os.path.exists(f"piles/{self.target_language}/pile_of_specific_actions.csv"):
            print("csv already exists! delete it to re-translate")
            return

        with open(f"piles/{self.source_language}/pile_of_specific_actions.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            pile_of_specific_actions = list(reader)
            pile_of_specific_actions = list(
                filter(
                    lambda x: x["service_name"].split(".")[0] in SUPPORTED_DEVICES,
                    pile_of_specific_actions,
                )
            )
            for actions_dict in tqdm(pile_of_specific_actions):
                try:
                    pile_of_specific_actions_target.append(
                        {
                            "service_name": actions_dict["service_name"],
                            "device_name": format_device_name(self.translate(actions_dict["device_name"])),
                            "phrase": self.translate(actions_dict["phrase"]),
                        }
                    )
                except KeyError as ex:
                    print(ex)
        with open(f"piles/{self.target_language}/pile_of_specific_actions.csv", "w+", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["service_name", "device_name", "phrase"]
            )
            writer.writeheader()
            writer.writerows(pile_of_specific_actions_target)


    def translate_status_requests(self):
        print("Translating status requests")
        pile_of_status_requests_target = list()

        if os.path.exists(f"piles/{self.target_language}/pile_of_status_requests.csv"):
            print("csv already exists! delete it to re-translate")
            return

        with open(f"piles/{self.source_language}/pile_of_status_requests.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            pile_of_status_requests = list(reader)
            pile_of_status_requests = list(
                filter(
                    lambda x: x["device_type"] in SUPPORTED_DEVICES,
                    pile_of_status_requests,
                )
            )
            for request_dict in tqdm(pile_of_status_requests):
                try:
                    pile_of_status_requests_target.append(
                        {
                            "device_type": request_dict["device_type"],
                            "state": request_dict["state"],
                            "phrase": self.translate(request_dict["phrase"]),
                            "assistant_response": self.translate(request_dict["assistant_response"]),
                        }
                    )
                except KeyError as ex:
                    print(ex)
        with open(f"piles/{self.target_language}/pile_of_status_requests.csv", "w+", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["device_type", "state", "phrase", "assistant_response"],
            )
            writer.writeheader()
            writer.writerows(pile_of_status_requests_target)

    def translate_durations(self):
        print("Translating durations")
        pile_of_durations_target = list()

        if os.path.exists(f"piles/{self.target_language}/pile_of_durations.csv"):
            print("csv already exists! delete it to re-translate")
            return

        with open(f"piles/{self.source_language}/pile_of_durations.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            pile_of_durations = list(reader)
            for duration_dict in tqdm(pile_of_durations):
                try:
                    pile_of_durations_target.append(
                        {
                            "duration": duration_dict["duration"],
                            "name": self.translate(duration_dict["name"])
                        }
                    )
                except KeyError as ex:
                    print(ex)
        with open(f"piles/{self.target_language}/pile_of_durations.csv", "w+", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["duration", "name"],
            )
            writer.writeheader()
            writer.writerows(pile_of_durations_target)


    def translate_responses(self):
        print("Translating responses")
        pile_of_responses_target = list()

        if os.path.exists(f"piles/{self.target_language}/pile_of_responses.csv"):
            print("csv already exists! delete it to re-translate")
            return

        with open(f"piles/{self.source_language}/pile_of_responses.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            pile_of_responses = list(reader)
            pile_of_responses = list(
                filter(
                    lambda x: x["service"].split(".")[0] in SUPPORTED_DEVICES,
                    pile_of_responses,
                )
            )
            for response_dict in tqdm(pile_of_responses):
                try:
                    pile_of_responses_target.append(
                        {
                            "service": response_dict["service"],
                            "response": self.translate(response_dict["response"]),
                            "persona": response_dict["persona"],
                            "short": response_dict["short"],
                        }
                    )
                except KeyError as ex:
                    print(ex)
        with open(f"piles/{self.target_language}/pile_of_responses.csv", "w+", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["service", "response", "persona", "short"]
            )
            writer.writeheader()
            writer.writerows(pile_of_responses_target)


    def translate_system_prompts(self):
        print("Translating system prompts")
        pile_of_system_prompts_target = list()

        if os.path.exists(f"piles/{self.target_language}/pile_of_system_prompts.csv"):
            print("csv already exists! delete it to re-translate")
            return

        with open(f"piles/{self.source_language}/pile_of_system_prompts.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            pile_of_system_prompts = list(reader)
            for prompt_dict in tqdm(pile_of_system_prompts):
                try:
                    pile_of_system_prompts_target.append(
                        {
                            "persona": prompt_dict["persona"],
                            "prompt": self.translate(prompt_dict["prompt"]),
                        }
                    )
                except KeyError as ex:
                    print(ex)
        with open(f"piles/{self.target_language}/pile_of_system_prompts.csv", "w+", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["persona", "prompt"]
            )
            writer.writeheader()
            writer.writerows(pile_of_system_prompts_target)

    def translate_todo_items(self):
        print("Translating todo items")
        pile_of_todo_items_target = list()

        if os.path.exists(f"piles/{self.target_language}/pile_of_todo_items.txt"):
            print("csv already exists! delete it to re-translate")
            return

        with open(f"piles/{self.source_language}/pile_of_todo_items.txt", encoding="utf-8") as f:
            english_phrases = f.readlines()
            for english_phrase in tqdm(english_phrases):
                pile_of_todo_items_target.append(f"{self.translate(english_phrase)}\n")
        with open(f"piles/{self.target_language}/pile_of_todo_items.txt", "w+", encoding="utf-8") as f:
            f.writelines(pile_of_todo_items_target)


# TODO: cmd line args
DatasetTranslator("english", "german").translate_all_piles()
DatasetTranslator("english", "spanish").translate_all_piles()
DatasetTranslator("english", "french").translate_all_piles()