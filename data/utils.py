import random
import re
import os
import csv
import pandas
from datetime import datetime, timedelta
import webcolors

class NoResponseAvailableException(Exception):
    pass

class NoServicesAvailableException(Exception):
    pass


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

def generate_random_parameter(param_name, piles_of_data):
    RANDOM_PARAMETER_GENERATORS = {
        "rgb_color": lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        "brightness": lambda: random.randint(0, 100),
        "fan_mode": lambda: random.choice(["On Low", "On High", "Auto Low", "Auto High", "Off"]),
        "temp_f": lambda: random.randint(60, 80),
        "temp_c": lambda: random.randint(15, 25),
        "humidity": lambda: random.randint(10, 90),
        "preset_mode": lambda: random.choice(["home", "eco", "away", "auto"]),
        "hvac_mode": lambda: random.choice(["heat", "cool", "heat_cool", "off", "auto", "fan_only"]),
        "media": lambda: random.choice(piles_of_data["pile_of_media_names"]),
        "volume": lambda: round(random.random(), 2),
        "duration": lambda: random.choice(list(piles_of_data["pile_of_durations"].keys())),
        "remaining": lambda: f"{random.randint(0, 3):02}:{random.randint(0, 60)}:{random.randint(0, 60)}",
        "todo": lambda: random.choice(piles_of_data["pile_of_todo_items"]),
    }
    param_generator = RANDOM_PARAMETER_GENERATORS.get(param_name)

    if not param_generator:
        raise Exception(f"Unknown param to generate random value for {param_name}")
    
    return param_generator()

def get_random_response(pile_of_responses, *, service: str, persona: str, question_template: str, short: bool) -> tuple[str, str]:

    required_vars = list(set([var for var in var_pattern.findall(question_template) if "device_name" not in var]))
    
    possible_results = pile_of_responses.loc[(pile_of_responses['service']==service) & 
                          (pile_of_responses['persona']==persona) &
                          (pile_of_responses['short']==(1 if short else 0)) &
                          (pile_of_responses['contains_vars']==",".join(sorted(required_vars)))
                        ]
    
    if len(possible_results) == 0:
        raise NoResponseAvailableException(f"No responses matched the provided filters: {persona}, {service}, {required_vars}, {short}")
    
    return possible_results.sample()["response_starting"].values[0], possible_results.sample()["response_confirmed"].values[0]

class DatasetPiles:
    def __init__(self, supported_devices, language="english"):
        self.language = language

        cwd = os.path.dirname(__file__)
        
        with open(f"{cwd}/piles/{language}/pile_of_and_words.csv", encoding="utf8") as f:
            self.and_words = [ x.strip() for x in f.readlines() ]
        
        with open(f"{cwd}/piles/{language}/pile_of_durations.csv", encoding="utf8") as f:
            reader = csv.DictReader(f)
            self.pile_of_durations = { x["duration"]: x["name"] for x in reader }
            
        # media names are not translated
        with open(f"{cwd}/piles/english/pile_of_media_names.txt", encoding="utf8") as f:
            self.pile_of_media_names = [ x.strip() for x in f.readlines() ]

        with open(f"{cwd}/piles/{language}/pile_of_todo_items.txt", encoding="utf8") as f:
            self.pile_of_todo_items = [ x.strip() for x in f.readlines() ]

        self.stacks_of_device_names = { x: [] for x in supported_devices }
        with open(f"{cwd}/piles/{language}/pile_of_device_names.csv", encoding="utf8") as f:
            reader = csv.DictReader(f)
            pile_of_device_names = list(reader)
            for device_dict in pile_of_device_names:
                try:
                    device_type = device_dict["device_name"].split(".")[0]
                    self.stacks_of_device_names[device_type].append(device_dict)
                except KeyError as ex:
                    print(ex)

        with open(f"{cwd}/piles/{language}/pile_of_templated_actions.csv", encoding="utf8") as f:
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

            self.pile_of_templated_actions = processed_pile_of_templated_actions

        with open(f"{cwd}/piles/{language}/pile_of_specific_actions.csv", encoding="utf8") as f:
            reader = csv.DictReader(f)
            self.pile_of_specific_actions = list(reader)

        self.pile_of_responses = pandas.read_csv(f"{cwd}/piles/{language}/pile_of_responses.csv")
        self.pile_of_responses["contains_vars"] = self.pile_of_responses["response_starting"].apply(get_included_vars)

        with open(f"{cwd}/piles/{language}/pile_of_status_requests.csv", encoding="utf8") as f:
            reader = csv.DictReader(f)
            self.pile_of_status_requests = list(reader)

        with open(f"{cwd}/piles/{language}/pile_of_system_prompts.csv", encoding="utf8") as f:
            reader = csv.DictReader(f)
            self.pile_of_system_prompts = { line["persona"]: line["prompt"] for line in reader }

        # service names are not translated
        with open(f"{cwd}/piles/english/pile_of_hallucinated_service_names.csv", encoding="utf8") as f:
            reader = csv.DictReader(f)
            self.pile_of_hallucinated_service_names = list(reader)

        failed_tool_calls_path = f"{cwd}/piles/{language}/pile_of_failed_tool_calls.csv"
        self.pile_of_failed_tool_calls = []
        if os.path.exists(failed_tool_calls_path):
            with open(failed_tool_calls_path, encoding="utf8") as f:
                reader = csv.DictReader(f)
                self.pile_of_failed_tool_calls = list(reader)

        refusals_path = f"{cwd}/piles/{language}/pile_of_refusals.csv"
        self.pile_of_refusals = []
        if os.path.exists(refusals_path):
            with open(refusals_path, encoding="utf8") as f:
                reader = csv.DictReader(f)
                self.pile_of_refusals = list(reader)

    def __getitem__(self, key):
        return getattr(self, key)

_piles_cache = {}

def get_dataset_piles(language: str) -> DatasetPiles:
    if language not in _piles_cache:
        _piles_cache[language] = DatasetPiles( [
            "light", "switch", "fan", "garage_door", "blinds",
            "lock","media_player", "climate", "vacuum", "timer", "todo",
        ], language)
    return _piles_cache[language]
