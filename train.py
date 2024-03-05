#!/usr/bin/env python3

import math
import copy
import torch
import os
import random
from torch.utils.data import SequentialSampler, Subset, RandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    PreTrainedTokenizerFast, HfArgumentParser, GPTQConfig, AutoConfig
from transformers.trainer_utils import EvalPrediction
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Sized, Iterator

"""
Phi Modules: 
- MLP: fc1,fc2
- MHA: q_proj,v_proj,k_proj,dense
- Embeddings: embed_tokens (input) lm_head (output)
StableLM Modules: 
- MLP: up_proj,down_proj,gate_proj
- MHA: q_proj,v_proj,k_proj,o_proj
- Embeddings: embed_tokens (input) lm_head (output)
"""

"""
python3 train.py \
    --run_name Home-3B-v2_ha-GGUF \
    --base_model microsoft/phi-2 \
    --add_pad_token \
    --add_chatml_tokens \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --learning_rate 1e-5 \
    --save_steps 1000 \
    --micro_batch_size 2 --gradient_checkpointing \
    --ctx_size 2048 \
    --use_lora --lora_rank 32 --lora_alpha 64 --lora_modules fc1,fc2,q_proj,v_proj,dense --lora_modules_to_save embed_tokens,lm_head --lora_merge
"""

"""
python3 train.py \
    --run_name home-1b-rev6 \
    --base_model microsoft/phi-1_5 \
    --add_pad_token \
    --add_chatml_tokens \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 1e-5 \
    --micro_batch_size 4 --gradient_checkpointing \
    --ctx_size 2048 --save_steps 200
"""

"""
python3 train.py \
    --run_name stablehome-1_6b-rev2 \
    --base_model stabilityai/stablelm-2-zephyr-1_6b \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 1e-5 \
    --micro_batch_size 2 --gradient_checkpointing \
    --ctx_size 2048 --save_steps 200 --save_total_limit 6
"""

"""
python3 train.py \
    --run_name stablehome-3b-rev5 \
    --base_model stabilityai/stablelm-zephyr-3b \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 1e-5 \
    --micro_batch_size 4 --gradient_checkpointing \
    --ctx_size 2048 \
    --save_steps 400 --save_total_limit 20 \
    --use_lora --lora_rank 64 --lora_alpha 128 --lora_modules up_proj,down_proj,q_proj,v_proj,o_proj --lora_merge
"""

"""
python3 train.py \
    --run_name home-7b-rev2 \
    --base_model TheBloke/Llama-2-7B-GPTQ \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --load_as_gptq --use_lora --gradient_checkpointing \
    --add_pad_token --bf16 --micro_batch_size 4 --learning_rate 2e-5
"""

@dataclass
class TrainingRunArguments:
    run_name: str = field(metadata={"help": "The folder to save the output model under"})
    base_model: str = field(metadata={"help": "The base model to load for fine-tuning"})
    train_dataset: str = field(metadata={"help": "The JSON file containing the training dataset"})
    test_dataset: str = field(default=None, metadata={"help": "The JSON file containing the evaluation dataset"})
    ctx_size: int = field(default=2048, metadata={"help": "The number of tokens to pad & truncate the input examples to"})
    bf16: bool = field(default=False, metadata={"help": "If set, the model will the loaded and trained in bf16 instead of fp16"})
    batch_size: int = field(default=8, metadata={"help": "The simulated 'batch size' that we will train on. will tweak gradient accumulations steps"})
    micro_batch_size: int = field(default=2, metadata={"help": "The actual batch size that will fit into VRAM on this machine"})
    eval_batch_size: int = field(default=1, metadata={"help": "The batch size for generation used while performing evaluation"})
    epochs: int = field(default=1, metadata={"help": "The number of times to train the model on each example"})
    learning_rate: float = field(default=1e-5, metadata={"help": "The starting learning rate (speed at which the model trains)"})
    learning_rate_schedule: str = field(default="cosine", metadata={"help": "How fast the learning rate is reduced during training"})
    weight_decay: float = field(default=0.1, metadata={"help": ""})
    gradient_clip: float = field(default=1.0, metadata={"help": ""})
    resume_from_checkpoint: str = field(default="", metadata={"help": "The name of the checkpoint to resume training from"})
    eval_steps: int = field(default=200, metadata={"help": "The number of steps in between evaluations of the model; set to -1 to evaluate every epoch"})
    save_steps: int = field(default=-1, metadata={"help": "The number of steps in between model checkpoints; set to -1 to save every epoch"})
    save_total_limit: int = field(default=1, metadata={"help": "The number of recent checkpoints of the model to save (not including the final model)"})
    group_by_length: bool = field(default=False, metadata={"help": "If enabled, the training data will be grouped by length to optimize use of padding"})
    pre_allocate_cuda_buffers: bool = field(default=True, metadata={"help": "If enabled, runs a forward and backward pass on the model before training to force pytorch to allocate the correct size CUDA buffers up front"})
    
    # Quantization
    load_in_8bit: bool = field(default=False, metadata={"help": "Set to load the base model in 8-bit mode using bitsandbytes"})
    load_in_4bit: bool = field(default=False, metadata={"help": "Set to load the base model in 4-bit mode using bitsandbytes"})
    load_as_gptq: bool = field(default=False, metadata={"help": "Set to load the base model as a GPTQ using AutoGPTQ"})
    
    # lora config
    use_lora: bool = field(default=False, metadata={"help": "If set, then the trained model will be a LoRA"})
    lora_rank: int = field(default=4)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_modules: str = field(default=None)
    lora_modules_to_save: str = field(default=None, metadata={"help": "Additional modules to save"})
    lora_merge: bool = field(default=False, metadata={"help": "If set, the Lora will be merged back into the base model an saved"})

    add_pad_token: bool = field(default=False, metadata={"help": "If set, a pad token will be added to the tokenizer's vocabulary"})
    add_chatml_tokens: bool = field(default=False, metadata={"help": "If set, tokens for the ChatML format will be added specifically"})
    add_chatml_prompt_template: bool = field(default=False, metadata={"help": "If set, the ChatML prompt template will be set as the model's Jinja2 template"})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enables gradient checkpointing which saves quite a lot of VRAM"})

    run_tensorboard: bool = field(default=False, metadata={"help": "If set, will tensorboard in the background to monitor training progress"})

parser = HfArgumentParser([TrainingRunArguments])
training_run_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

if sum([training_run_args.load_in_8bit, training_run_args.load_in_4bit, training_run_args.load_as_gptq]) > 1:
    raise Exception("Please select exactly one of 'load_in_8bit', 'load_in_4bit', or 'load_as_gptq")

print(f"Loading model '{training_run_args.base_model}'...")

model_kwargs = {}
if training_run_args.load_in_8bit:
    model_kwargs["load_in_8bit"] = True
elif training_run_args.load_in_4bit:
    model_kwargs["load_in_4bit"] = True
elif training_run_args.load_as_gptq:
    model_kwargs["quantization_config"] = GPTQConfig(bits=4, disable_exllama=True)


if training_run_args.bf16:
    model_kwargs["torch_dtype"] = torch.bfloat16
else:
    model_kwargs["torch_dtype"] = torch.float16

# model_kwargs["resid_pdrop"] = 0.0
# model_kwargs["revision"] = "accfee56d8988cae60915486310362db5831b1bd"
model_kwargs["use_cache"] = False

def find_max_vram(min_buffer_mib=800):
    total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    suggestion = round((total_mem - 1000) / 1000) * 1000
    suggestion = min(suggestion, total_mem - min_buffer_mib)

    print(f"Model will target using {suggestion}MiB of VRAM")
    max_memory = {0: f'{suggestion}MiB'}

    return max_memory if len(max_memory) > 0 else None

model = AutoModelForCausalLM.from_pretrained(
    training_run_args.base_model,
    trust_remote_code=True,
    device_map="auto",
    max_memory=find_max_vram(),
    **model_kwargs
)
tokenizer = AutoTokenizer.from_pretrained(training_run_args.base_model, trust_remote_code=True)

if training_run_args.add_pad_token:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    model.config.pad_token_id = tokenizer.pad_token_id

if training_run_args.add_chatml_tokens:
    tokenizer.add_special_tokens({
        'bos_token': '<|im_start|>',
        'eos_token': '<|im_end|>'
    })

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

if training_run_args.add_chatml_prompt_template:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

embeddings_len = math.ceil(len(tokenizer) / 32) * 32
if model.get_input_embeddings().num_embeddings < embeddings_len:
    model.resize_token_embeddings(embeddings_len)
else:
    model.tie_weights()

# model.tie_weights()

if training_run_args.use_lora:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    print("Creating LoRA for model...")
    target_modules = training_run_args.lora_modules.split(",") if training_run_args.lora_modules else None
    modules_to_save = training_run_args.lora_modules_to_save.split(",") if training_run_args.lora_modules_to_save else None
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=training_run_args.lora_rank,
        lora_alpha=training_run_args.lora_alpha,
        lora_dropout=training_run_args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    if training_run_args.load_in_8bit or training_run_args.load_in_4bit or training_run_args.load_as_gptq:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_run_args.gradient_checkpointing
        )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()

    model.print_trainable_parameters()
    

base_dir = "loras" if training_run_args.use_lora else "models"
model_dir = f"./{base_dir}/{training_run_args.run_name}"

training_kwargs = {}

if training_run_args.test_dataset:
    training_kwargs.update({
        "per_device_eval_batch_size": training_run_args.eval_batch_size,
        "evaluation_strategy": ("steps" if training_run_args.eval_steps != -1 else "epoch"),
        "eval_steps": (training_run_args.eval_steps if training_run_args.eval_steps != -1 else None),
        "bf16_full_eval": training_run_args.bf16,
    })

training_args = TrainingArguments(
    per_device_train_batch_size=training_run_args.micro_batch_size,
    gradient_accumulation_steps=training_run_args.batch_size//training_run_args.micro_batch_size,
    gradient_checkpointing=training_run_args.gradient_checkpointing,
    weight_decay=training_run_args.weight_decay,
    max_grad_norm=training_run_args.gradient_clip,
    save_strategy=("steps" if training_run_args.save_steps != -1 else "epoch"),
    save_steps=(training_run_args.save_steps if training_run_args.save_steps != -1 else None),
    save_safetensors=True,
    logging_steps=5, 
    output_dir=model_dir,
    num_train_epochs=training_run_args.epochs,
    save_total_limit=training_run_args.save_total_limit,
    report_to="tensorboard",
    learning_rate=training_run_args.learning_rate,
    lr_scheduler_type=training_run_args.learning_rate_schedule,
    log_level="info",
    bf16=training_run_args.bf16,
    group_by_length=training_run_args.group_by_length,
    skip_memory_metrics=True,
    **training_kwargs,
    # include_inputs_for_metrics=True,
)

class DataCollatorForSupervisedFineTuning(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: AutoTokenizer
    prompt_split: str
    response_prefix: str
    response_suffix: str
    prefix_ids: list[int]
    suffix_ids: list[int]

    def __init__(self, *, tokenizer: AutoTokenizer):
        
        self.tokenizer = tokenizer
        assistant_prompt = tokenizer.apply_chat_template(conversation=[{"role": "assistant", "content":  r"%%%%%%%%%%%%%%%%"}], tokenize=False).split( r"%%%%%%%%%%%%%%%%")
        self.response_prefix = assistant_prompt[0]
        self.response_suffix = assistant_prompt[1]

        self.prefix_ids = self.tokenizer(self.response_prefix, add_special_tokens=False)["input_ids"]
        self.suffix_ids = self.tokenizer(self.response_suffix, add_special_tokens=False)["input_ids"]

    def _find_mask_ranges(self, input_ids):
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
                start_idx = input_ids.index(self.prefix_ids[0], i)
            except ValueError:
                break

            # Check if the entire prefix is present
            if input_ids[start_idx:start_idx + len(self.prefix_ids)] == self.prefix_ids:
                end_prefix_idx = start_idx + len(self.prefix_ids)
                start_response_idx = end_prefix_idx + 1

                # Find the start index of the suffix
                try:
                    # Find the start index of the suffix
                    suffix_start_idx = input_ids.index(self.suffix_ids[0], end_prefix_idx)
                except ValueError:
                    ranges.append((start_response_idx, len(input_ids)))
                    break

                # Check if the entire suffix is present
                if input_ids[suffix_start_idx:suffix_start_idx + len(self.suffix_ids)] == self.suffix_ids:
                    ranges.append((start_response_idx, suffix_start_idx))
                    i = suffix_start_idx + len(self.suffix_ids)
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
    
    def _pad(self, examples, pad_value):
        longest = max([len(ex) for ex in examples])
        result = []
        for example in examples:
            cur_len = len(example)
            result.append(example + [pad_value] * (longest - cur_len))

        return result

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = copy.deepcopy(input_ids)

        for label in labels:
            mask_ranges = self._find_mask_ranges(label)
            for start, end in mask_ranges:
                if end - start == len(label):
                    print("warning! example had no assistant response in it!")
                label[start:end] = [-100] * (end - start)

        input_ids = torch.LongTensor(self._pad(input_ids, self.tokenizer.pad_token_id))
        labels = torch.LongTensor(self._pad(labels, -100))

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

print("Loading dataset...")
data_files = { "train": training_run_args.train_dataset }
if training_run_args.test_dataset:
    data_files["test"] = training_run_args.test_dataset
datasets = load_dataset("json", data_files=data_files)

def tokenize_raw_example(batch):
    return tokenizer(
        text=batch["text"],
        max_length=training_run_args.ctx_size,
        truncation=True,
        add_special_tokens=False,
    )

def tokenize_sharegpt_example(batch):
    # TODO: figure out how to properly batch this
    result = []
    for example in batch["conversations"]:
        conversation = [ { "role": x["from"], "content": x["value"] }  for x in example ]
        result.append(tokenizer.apply_chat_template(
            conversation=conversation,
            max_length=training_run_args.ctx_size,
            truncation=True,
        ))

    return {"input_ids": result}

print("Tokenizing datasets...")

if "text" in datasets["train"].column_names:
    tokenize_function = tokenize_raw_example
    columns_to_remove = ["text"]
elif "conversations" in datasets["train"].column_names:
    tokenize_function = tokenize_sharegpt_example
    columns_to_remove = ["conversations"]
else:
    raise Exception("Unknown dataset input format (not raw corpus or sharegpt)")

tokenized_test_dataset = None
tokenized_train_dataset = datasets["train"].map(tokenize_function, batched=True).remove_columns(columns_to_remove)
if training_run_args.test_dataset:
    tokenized_test_dataset = datasets["test"].map(tokenize_function, batched=True).remove_columns(columns_to_remove)

example_lengths = [ len(example) for example in tokenized_train_dataset["input_ids"] ]
tokens_in_train_set, longest_example = sum(example_lengths), max(example_lengths)
print(f"Train dataset has {int(tokens_in_train_set / 1000000)}M tokens. Longest Example: {longest_example} tokens")

data_collator = DataCollatorForSupervisedFineTuning(tokenizer=tokenizer)

class CustomTrainer(Trainer):
    """Implement different training tweaks"""
    def __init__(self, random_eval_sample_pct=0.1, learning_rate_overshoot=1.15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_eval_sample_pct = random_eval_sample_pct
        self.evaluate_full_dataset = False
        self.learning_rate_overshoot = learning_rate_overshoot

    def evaluate_all(self):
        self.evaluate_full_dataset = True
        super().evaluate()
        self.evaluate_full_dataset = False

    # Randomly sample the eval dataset
    def _get_eval_sampler(self, eval_dataset):
        if self.evaluate_full_dataset:
            return SequentialSampler(eval_dataset)
        else:
            num_samples = int(self.random_eval_sample_pct * len(eval_dataset))
            random_indices = random.sample(range(len(eval_dataset)), num_samples)
            subset_eval_dataset = Subset(eval_dataset, random_indices)
            return SequentialSampler(subset_eval_dataset)
        
    def _get_train_sampler(self):
        if self.args.group_by_length:
            return super()._get_train_sampler()
        
        return RandomSampler(self.train_dataset, generator=torch.Generator(device='cpu'))
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Saw this in the chinchilla paper. It says not to go over 25% overshoot
        Should speed up training by skipping the final fine tuning part that doesn't affect accuracy much
        """
        return super().create_scheduler(int(num_training_steps * self.learning_rate_overshoot), optimizer=optimizer)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
)

tensorboard_process = None
def kill_tensorboard():
    tensorboard_process.kill()

if training_run_args.run_tensorboard:
    import subprocess, atexit
    tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", model_dir])
    atexit.register(kill_tensorboard)

# pre-allocate cuda buffers by running a forwards and backwards pass with the largest possible example length
# the trainer dumps the cuda buffers before we start... need to figure out how to disable that
# if training_run_args.pre_allocate_cuda_buffers:
#     print("Allocating CUDA buffers...")
#     inputs = tokenizer([""] * training_args.per_device_train_batch_size, return_tensors="pt", max_length=longest_example, padding="max_length", truncation=True)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     inputs["labels"] = inputs["input_ids"]
#     outputs = model(**inputs)
#     loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
#     loss.backward()
#     model.zero_grad()

try:
    checkpoint = training_run_args.resume_from_checkpoint
    if checkpoint:
        trainer.train(checkpoint)
    else:
        trainer.train()

    if training_run_args.train_dataset:
        trainer.evaluate_all()

    if training_run_args.use_lora and training_run_args.lora_merge:
        trainer.save_model() # save lora

        merged_model = model.merge_and_unload(progressbar=True)
        merged_model_dir = f"./models/{training_run_args.run_name}"
        merged_model.save_pretrained(merged_model_dir, safe_serialization=True, max_shard_size="2GB")
        
        tokenizer.save_pretrained(merged_model_dir)
    else:
        trainer.save_model()
        tokenizer.save_pretrained(model_dir)

    if tensorboard_process:
        input("Training is finished. Press enter to quit tensorboard after the viewing results.")
        tensorboard_process.kill()
except Exception as e:
    print("Something bad happened! Try and save it?")
    import code, traceback
    traceback.print_exc()
    code.interact(local=locals())