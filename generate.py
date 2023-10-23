import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

torch.set_default_device("cuda")
# base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
# phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
trained_model = AutoModelForCausalLM.from_pretrained("./model/home-llm-rev7.1", trust_remote_code=True)
trained_tokenizer = AutoTokenizer.from_pretrained(
    "./model/home-llm-rev7.1/tokenizer.json", trust_remote_code=True,
    config=AutoConfig.from_pretrained("./model/home-llm-rev7.1"))

def generate(model, tokenizer, inputs):
    outputs = model.generate(**inputs, max_length=512)
    text = tokenizer.batch_decode(outputs)[0]
    return text

def tokenize(tokenizer, question):
    prompt = f"{question}\nAnswer: "
    return tokenizer(prompt, return_tensors="pt", return_attention_mask=False)


# def compare(question):
#     tokenized_prompt = tokenize(phi_tokenizer, question)
#     print("--------------------------------------------------------------------")
#     print("base model:")
#     print(generate(base_model, phi_tokenizer, tokenized_prompt))
#     print("--------------------------------------------------------------------")
#     print("fine tuned model:")
#     print(generate(trained_model, phi_tokenizer, tokenized_prompt))

# prompt1 = '''My brother is 10 years older than me when I am 3 years younger than my sister who is 7. In 3 years how old will I be?'''
# prompt2 = '''Find the vertex of the parabola defined by the function y = 4x^2 + 8x - 2'''
# prompt3 = '''Joe has 3 pairs of shoes that he got for his birthday. If Joe can sell one pair of shoes for $200 but has to pay 10% tax on each sale, how much money will Joe get if he sells 2 of the pairs of shoes and keeps the last pair for himself?'''

# compare(prompt1)
# compare(prompt2)

prompt = "You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task ask instructed with the information provided only.\nServices: close_cover, decrease_speed, increase_speed, open_cover, stop_cover, toggle, turn_off, turn_on, unlock\nDevices:\nlight.upstairs_basement_zwave = on\nlight.upstairs_utility_zigbee = off\nfan.dining_area = on\nlight.back_arcade_warm = on\nlight.downstairs_sunroom_homekit = on\nblinds.patio = open\ngarage_door.basement = closed\nlight.side_hallway_hue = on\nlight.patio_1 = off\nlight.front_den_ge = on\nlight.kitchen_trashbin_cool = on\nlight.deck_left = on\nfan.honeywell_turbo = on\nlight.kitchen_floor_cool = on\nfan.living_room_center = on\nlight.kitchen_microwave_warm = on\nlight.front_arcade_ge = off\nlock.office = locked\nfan.ceiling_1 = on\nlight.kitchen_toaster_cool = on\nlight.office_2 = on\nlight.kitchen_sideboard_cool = off\nlight.upstairs_playroom_zwave = on\nlight.kitchen_sink_cool = on\ngarage_door.golf_cart = open\ngarage_door.main_1 = closed\nlight.upstairs_observatory_zigbee = on\nlight.garden_2 = off\nlight.patio = on\nlight.garden = off\nfan.study = on\nlight.kitchen_1 = off\nRequest: please unlock the office\nResponse:"

generate(trained_model, trained_tokenizer, trained_tokenizer(prompt, return_tensors="pt", return_attention_mask=False))