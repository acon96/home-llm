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
    conversation=[{"role": "assistant", "content":  r"%%%%%%%%%%%%%%%%"}],
    tokenize=False,
    add_generation_prompt=False,
).split( r"%%%%%%%%%%%%%%%%")

response_prefix = assistant_prompt[0]
response_suffix = assistant_prompt[1]

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

print(f"Estimated tokens for {model}")
print("response prefix:")
print(response_prefix)
print("tokens with no leading whitespace:", prefix_ids)
print("tokens with leading whitespace:", prefix_ids2)
print("tokens with leading newline:", prefix_ids3)

print("---------------")

print("response suffix:")
print(response_suffix)
print("tokens with no leading whitespace:", suffix_ids)
print("tokens with leading whitespace:", suffix_ids2)
print("tokens with leading newline:", suffix_ids3)
