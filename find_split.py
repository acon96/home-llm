# this script attempts to figure out the correct prefix_ids and suffix_ids for the given model
# usage: python3 find_split.py <model name>
from transformers import AutoTokenizer
from jinja2.exceptions import TemplateError
import sys

if len(sys.argv) > 1:
    model = sys.argv[1]
else:
    print(f"Usage: {sys.argv[0]} <model name>")
    exit(-1)

prefix_ids = None
suffix_ids = None
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

test_prompt = tokenizer.apply_chat_template(
    conversation=[
        {"role": "user", "content": r"HA_REQUEST"},
        {"role": "assistant", "content": r"HA_RESPONSE"}
    ],
    tokenize=False,
    add_generation_prompt=False,
)

print("Chat template:")
print("-" * 100)
print(test_prompt)
print("-" * 100)

# Added real example to test the tokenizer
test_prompt_tokens = tokenizer.apply_chat_template(
    conversation=[
        {"role": "system", "content": "this is a system prompt"},
        {"role": "user", "content":  "a user request goes here"},
        {"role": "assistant", "content":  "the response is in here"}
    ],
    tokenize=True,
    add_generation_prompt=False
)

print("Chat template tokens:")
print("-" * 100)
print(test_prompt_tokens)
print("-" * 100)

try:
    assistant_prompt = tokenizer.apply_chat_template(
        conversation=[{"role": "assistant", "content":  r"%%%%%%%%%%%%%%%%"}],
        tokenize=False,
        add_generation_prompt=False,
    ).split( r"%%%%%%%%%%%%%%%%")
except TemplateError:
    user_prompt = tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": r"$$$$$$$$$$$$$$$$"}
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    # some prompt templates require user/assistant alternating
    assistant_prompt = tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": r"$$$$$$$$$$$$$$$$"},
            {"role": "assistant", "content":  r"%%%%%%%%%%%%%%%%"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    ).split(r"$$$$$$$$$$$$$$$$")[-1].strip().split(r"%%%%%%%%%%%%%%%%")

response_prefix = assistant_prompt[0]
response_suffix = assistant_prompt[1]

# check for inserted system prompt and remove it
if tokenizer.eos_token in response_prefix:
    response_prefix = response_prefix.split(tokenizer.eos_token)[-1].lstrip()

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
print("tokens with stripped whitespace:", prefix_ids4)

print("-" * 100)

print("response suffix:")
print(response_suffix)
print("tokens with no leading whitespace:", suffix_ids)
print("tokens with leading whitespace:", suffix_ids2)
print("tokens with leading newline:", suffix_ids3)
print("tokens with stripped whitespace:", suffix_ids4)


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

try:
    label = tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": "this is a system prompt"},
            {"role": "user", "content":  "a user request goes here"},
            {"role": "assistant", "content":  "the response is in here"}
        ],
        add_generation_prompt=False,
    )
except TemplateError:
    # some chat templates don't have a system prompt option
    label = tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content":  "a user request goes here"},
            {"role": "assistant", "content":  "the response is in here"}
        ],
        add_generation_prompt=False,
    )

def check_range(label, name, prefix_ids, suffix_ids):
    label = label[:]
    mask_ranges = _find_mask_ranges(label, prefix_ids, suffix_ids)

    found = False
    for start, end in mask_ranges:
        if end - start == len(label) - 1:
            print(f"'{name}' did not find the assistant response")
        else:
            found = True
            # label[start:end] = [-100] * (end - start)

    # assistant_tokens = [x for x in label if x != -100]
    # decoded_string = tokenizer.decode(assistant_tokens)
    # expected_decoded_string = "the response is in here" + tokenizer.decode(suffix_ids)
    # if decoded_string == expected_decoded_string:
    #     found = True

    if found:
        print(f"'{name}' found the assistant response!")
        print(f"\t--prefix_ids {','.join([str(x) for x in prefix_ids])}")
        print(f"\t--suffix_ids {','.join([str(x) for x in suffix_ids])}")
    # else:
    #     print(f"'{decoded_string}' != '{expected_decoded_string}'")

print("-" * 100)
check_range(label, "no added whitespace", prefix_ids, suffix_ids)
check_range(label, "leading space", prefix_ids2, suffix_ids2)
check_range(label, "leading newline", prefix_ids3, suffix_ids3)
check_range(label, "stripped whitespace", prefix_ids4, suffix_ids4)
