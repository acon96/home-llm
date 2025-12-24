import babel.dates

from utils import generate_random_datetime

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


def generate_system_prompt(example: dict, persona: str, language: str, pile_of_system_prompts: dict) -> str:
    sys_prompt = pile_of_system_prompts[persona]
    random_datetime = generate_random_datetime()
    translate_datetime = babel.dates.format_datetime(random_datetime, BABEL_FORMAT[language], locale=BABEL_LOCALE[language])
    time_block = f"{CURRENT_DATE_PROMPT[language]} {translate_datetime}" 

    states_block = f"{DEVICES_PROMPT[language]}:\n" + "\n".join(example["states"])
    
    # replace aliases with their actual values
    states_block = states_block.replace("blinds.", "cover.").replace("garage_door.", "cover.")

    return "\n".join([sys_prompt, time_block, states_block])