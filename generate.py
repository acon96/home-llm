import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
trained_model = AutoModelForCausalLM.from_pretrained("./model/training", trust_remote_code=True)
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

def generate(model, tokenizer, inputs):
    outputs = model.generate(**inputs, max_length=512)
    text = tokenizer.batch_decode(outputs)[0]
    return text

def tokenize(tokenizer, question):
    prompt = f"{question}\nAnswer: "
    return tokenizer(prompt, return_tensors="pt", return_attention_mask=False)


def compare(question):
    tokenized_prompt = tokenize(phi_tokenizer, question)
    print("--------------------------------------------------------------------")
    print("base model:")
    print(generate(base_model, phi_tokenizer, tokenized_prompt))
    print("--------------------------------------------------------------------")
    print("fine tuned model:")
    print(generate(trained_model, phi_tokenizer, tokenized_prompt))

prompt1 = '''My brother is 10 years older than me when I am 3 years younger than my sister who is 7. In 3 years how old will I be?'''
prompt2 = '''Find the vertex of the parabola defined by the function y = 4x^2 + 8x - 2'''
prompt3 = '''Joe has 3 pairs of shoes that he got for his birthday. If Joe can sell one pair of shoes for $200 but has to pay 10% tax on each sale, how much money will Joe get if he sells 2 of the pairs of shoes and keeps the last pair for himself?'''

compare(prompt1)
compare(prompt2)