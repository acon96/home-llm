# this script attempts to figure out the correct prefix_ids and suffix_ids for the given model
# usage: python3 find_split.py <model name>
from transformers import AutoTokenizer
import sys

if len(sys.argv) > 1:
    model = sys.argv[1]
else:
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

prefix_ids = None
suffix_ids = None
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

assistant_prompt = tokenizer.apply_chat_template(
    conversation=[
        {"role": "user", "content": r"HA_REQUEST"},
        {"role": "assistant", "content": r"HA_RESPONSE"}
    ],
    tokenize=False,
    add_generation_prompt=False,
)

print("Chat template:")
print("-" * 100)
print(assistant_prompt)
print("-" * 100)

# Added real example to test the tokenizer
assistant_prompt_tokens = tokenizer.apply_chat_template(
    conversation=[
        {"role": "system", "content": r"Jeste\u015b \u201eAl\u201d, pomocnym asystentem AI, kt\u00f3ry kontroluje urz\u0105dzenia w domu. Wykonaj poni\u017csze zadanie zgodnie z instrukcj\u0105 lub odpowiedz na poni\u017csze pytanie, korzystaj\u0105c wy\u0142\u0105cznie z podanych informacji.\nAktualna godzina i data to 16:42 w niedziela, 18 marca 2029\nUs\u0142ugi: cover.close_cover(), cover.open_cover(), cover.stop_cover(), cover.toggle(), fan.decrease_speed(), fan.increase_speed(), fan.toggle(), fan.turn_off(), fan.turn_on(), cover.close_cover(), cover.open_cover(), cover.stop_cover(), cover.toggle(), light.toggle(), light.turn_off(), light.turn_on(rgb_color,brightness), lock.lock(), lock.unlock(), media_player.media_next_track(), media_player.media_pause(), media_player.media_play(), media_player.media_play_pause(), media_player.media_previous_track(), media_player.media_stop(), media_player.toggle(), media_player.turn_off(), media_player.turn_on(), media_player.volume_down(), media_player.volume_mute(), media_player.volume_up(), switch.toggle(), switch.turn_off(), switch.turn_on()\nUrz\u0105dzenia:\nlight.g\u00f3ra_gabinet_zwave '\u015awiat\u0142o w gabinecie na pi\u0119trze' = on;saddlebrown (114, 71, 52);69%\nlight.prz\u00f3d_korytarz_homekit '\u015awiat\u0142o w przednim korytarzu' = off;seagreen (47, 111, 119);3%\nswitch.master_bedroom_lights 'G\u0142\u00f3wny w\u0142\u0105cznik \u015bwiate\u0142 w sypialni' = off\nmedia_player.roku_sypialnia 'Odtwarzacz Roku w sypialni' = off\ncover.wiata_samochodowa 'Drzwi do wiaty' = closed\nlight.\u0142azienka_2 '\u015awiat\u0142o w \u0142azience' = off;tan (216, 174, 142);97%\nlock.gara\u017c 'Zamek drzwi gara\u017cowych' = locked\nlock.drzwi_dachowe 'Zamek dost\u0119pu na dachu' = locked\nlock.szopa_2 'Drugi zamek szopy' = unlocked\nlight.g\u0142\u00f3wna_sypialnia_1 '\u015awiat\u0142o w g\u0142ownej sypialni' = off\nlock.drzwi_ganku 'Zamek drzwi werandy' = locked\ncover.szopa_2 'Drugie drzwi do szopy' = closed\nlock.pok\u00f3j_kartowania 'Zamek do pokoju akt' = unlocked\nlight.gara\u017c_osram '\u015awiat\u0142o w Gara\u017cu Osram' = on;purple (148, 40, 119)\nmedia_player.home_theater_system_living_room 'Kino domowe w salonie' = off\ncover.kuchnia 'Kuchnia' = open\nlock.szklarnia 'Zamek szklarni' = locked\nlock.sypialnia_3 'Zamek w trzeciej sypialni' = locked\nfan.ganek 'Wentylator sufitowy na werandzie' = off\nlight.tylna_sypialnia_ciep\u0142e 'Tylna sypialnia Ciep\u0142e \u015bwiat\u0142o' = off;mediumslateblue (125, 83, 236);81%\nswitch.\u015bwiat\u0142a_warsztatowe 'Prze\u0142\u0105cznik \u015bwiate\u0142 warsztatowych' = off\nlight.philips_hue_2 'Philips Hue' = on;darkseagreen (161, 205, 137);72%\ncover.ikea_smart 'Inteligentne rolety z Ikei' = open"},
        {"role": "user", "content": r"opu\u015b\u0107 \u017caluzje kuchenne"},
        {"role": "assistant", "content": r"zamykam rolety, jak prosi\u0142e\u015b.\n```homeassistant\n{\"service\": \"cover.close_cover\", \"target_device\": \"cover.kuchnia\"}\n```"}
    ],
    tokenize=True,
    truncation=True,
    max_length=2048,
    add_generation_prompt=False
)

print("Chat template tokens:")
print("-" * 100)
print(assistant_prompt_tokens)
print("-" * 100)

response_prefix = assistant_prompt.split(r"HA_REQUEST")[1].split(r"HA_RESPONSE")[0]
response_suffix = assistant_prompt.split(r"HA_RESPONSE")[1]

# check for inserted system prompt and remove it
if tokenizer.eos_token in response_prefix:
    response_prefix = response_prefix.split(tokenizer.eos_token)[1].lstrip()

# some chat templates ALWAYS add the bos token
if tokenizer.bos_token in response_prefix:
    response_prefix = response_prefix.replace(tokenizer.bos_token, "")

prefix_ids = tokenizer(response_prefix, add_special_tokens=False)["input_ids"]
suffix_ids = tokenizer(response_suffix, add_special_tokens=False)["input_ids"]

prefix_ids2 = tokenizer(" " + response_prefix, add_special_tokens=False)["input_ids"]
suffix_ids2 = tokenizer(" " + response_suffix, add_special_tokens=False)["input_ids"]

prefix_ids3 = tokenizer("\n" + response_prefix, add_special_tokens=False)["input_ids"]
suffix_ids3 = tokenizer("\n" + response_suffix, add_special_tokens=False)["input_ids"]

prefix_ids4 = tokenizer(response_prefix.strip(), add_special_tokens=False)["input_ids"]
suffix_ids4 = tokenizer(response_suffix.strip(), add_special_tokens=False)["input_ids"]

print(f"Estimated tokens for {model}")
print("response prefix:")
print(response_prefix)
print("tokens with no leading whitespace:", prefix_ids)
print("tokens with leading whitespace:", prefix_ids2)
print("tokens with leading newline:", prefix_ids3)
print("tokens without leading whitespace:", prefix_ids4)

print("---------------")

print("response suffix:")
print(response_suffix)
print("tokens with no leading whitespace:", suffix_ids)
print("tokens with leading whitespace:", suffix_ids2)
print("tokens with leading newline:", suffix_ids3)
print("tokens without leading whitespace:", suffix_ids4)


def _find_mask_ranges(input_ids, prefix_ids, suffix_ids):
    """
    Returns a mask that blocks out everything but the response from the assistant
    The mask does NOT include the response_prefix but DOES include the response_suffix.
    The resulting behavior is the model uses the prefix as a prompt and the suffix as the end of text token
    """
    ranges = []
    i = 0

    while i < len(input_ids):
        try:
            # Find the start index of the prefix
            start_idx = input_ids.index(prefix_ids[0], i)
        except ValueError:
            break

        # Check if the entire prefix is present
        if input_ids[start_idx:start_idx + len(prefix_ids)] == prefix_ids:
            end_prefix_idx = start_idx + len(prefix_ids)
            start_response_idx = end_prefix_idx + 1

            # Find the start index of the suffix
            try:
                # Find the start index of the suffix
                suffix_start_idx = input_ids.index(suffix_ids[0], end_prefix_idx)
            except ValueError:
                ranges.append((start_response_idx, len(input_ids)))
                break

            # Check if the entire suffix is present
            if input_ids[suffix_start_idx:suffix_start_idx + len(suffix_ids)] == suffix_ids:
                ranges.append((start_response_idx, suffix_start_idx))
                i = suffix_start_idx + len(suffix_ids)
            else:
                i = suffix_start_idx + 1
        else:
            i = start_idx + 1

    inverse_ranges = []
    current = 0

    for start, end in sorted(ranges):
        if start > current:
            inverse_ranges.append((current, start - 1))
        current = max(current, end + 1)
    
    if current < len(input_ids):
        inverse_ranges.append((current, len(input_ids) - 1))

    return inverse_ranges

label = tokenizer.apply_chat_template(
    conversation=[
        {"role": "system", "content": "this is a system prompt"},
        {"role": "user", "content":  "a user request goes here"},
        {"role": "assistant", "content":  "the response is in here"}],
    add_generation_prompt=False,
)

def check_range(label, name, prefix_ids, suffix_ids):
    label = label[:]
    mask_ranges = _find_mask_ranges(label, prefix_ids, suffix_ids)

    for start, end in mask_ranges:
        if end - start == len(label) - 1:
            print(f"'{name}' did not find the assistant response")
        else:
            print(f"'{name}' found the assistant response!")
            print(f"\t--prefix_ids {','.join([str(x) for x in prefix_ids])}")
            print(f"\t--suffix_ids {','.join([str(x) for x in suffix_ids])}")
            break

print("---------------")
check_range(label, "no whitespace", prefix_ids, suffix_ids)
check_range(label, "leading space", prefix_ids2, suffix_ids2)
check_range(label, "leading newline", prefix_ids3, suffix_ids3)
