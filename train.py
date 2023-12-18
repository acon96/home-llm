import math
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, HfArgumentParser, GPTQConfig, AutoConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

# torch.set_default_device("cuda")

"""
Phi Modules: fc1,fc2,Wqkv,out_proj,wte,lm_head.linear
"""

"""
python3 train.py \
    --run_name home-llm-rev11 \
    --base_model microsoft/phi-2 \
    --add_pad_token \
    --add_chatml_tokens \
    --bf16 \
    --train_dataset data/home_assistant_alpaca_merged_train.json \
    --test_dataset data/home_assistant_alpaca_merged_test.json \
    --learning_rate 1e-6 \
    --save_steps 1000 \
    --micro_batch_size 2 --gradient_checkpointing \
    --use_lora --lora_rank 16 --lora_modules fc1,fc2,Wqkv,out_proj --lora_modules_to_save wte,lm_head.linear --lora_merge
"""

"""
python3 train.py \
    --run_name home-llm-rev9.1 \
    --base_model microsoft/phi-1_5 \
    --disable_attention_mask \
    --add_pad_token \
    --bf16 \
    --train_dataset data/home_assistant_train.json \
    --test_dataset data/home_assistant_test.json
"""

"""
python3 train.py \
    --run_name home-llama2-7b-rev2 \
    --base_model TheBloke/Llama-2-7B-GPTQ \
    --train_dataset data/home_assistant_train.json \
    --test_dataset data/home_assistant_test.json \
    --load_as_gptq --use_lora --gradient_checkpointing \
    --add_pad_token --bf16 --micro_batch_size 4 --learning_rate 2e-5
"""

@dataclass
class TrainingRunArguments:
    run_name: str = field(metadata={"help": "The folder to save the output model under"})
    train_dataset: str = field(metadata={"help": "The JSON file containing the training dataset"})
    test_dataset: str = field(metadata={"help": "The JSON file containing the evaluation dataset"})
    base_model: str = field(metadata={"help": "The base model to load for fine-tuning"})
    ctx_size: int = field(default=512, metadata={"help": "The number of tokens to pad & truncate the input examples to"})
    bf16: bool = field(default=False, metadata={"help": "If set, the model will the loaded and trained in bf16 instead of fp16"})
    batch_size: int = field(default=8, metadata={"help": "The simulated 'batch size' that we will train on. will tweak gradient accumulations steps"})
    micro_batch_size: int = field(default=2, metadata={"help": "The actual batch size that will fit into VRAM on this machine"})
    epochs: int = field(default=1, metadata={"help": "The number of times to train the model on each example"})
    learning_rate: float = field(default=1e-5, metadata={"help": "The starting learning rate (speed at which the model trains)"})
    learning_rate_schedule: str = field(default="cosine", metadata={"help": "How fast the learning rate is reduced during training"})
    resume_from_checkpoint: str = field(default="", metadata={"help": "The name of the checkpoint to resume training from"})
    eval_steps: int = field(default=100, metadata={"help": "The number of steps in between evaluations of the model"})
    save_steps: int = field(default=-1, metadata={"help": "The number of steps in between model checkpoints; set to -1 to save every epoch"})
    
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
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enables gradient checkpointing which saves quite a lot of VRAM"})

    run_tensorboard: bool = field(default=False, metadata={"help": "If set, will tensorboard in the background to monitor training progress"})

parser = HfArgumentParser([TrainingRunArguments])
training_run_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

if sum([training_run_args.load_in_8bit, training_run_args.load_in_4bit, training_run_args.load_as_gptq]) > 1:
    raise Exception("Please select exactly one of 'load_in_8bit', 'load_in_4bit', or 'load_as_gptq")

# TODO: write a proper evaluation script

print(f"Loading model '{training_run_args.base_model}'...")

model_kwargs = {}
if training_run_args.load_in_8bit:
    model_kwargs["load_in_8bit"] = True
elif training_run_args.load_in_4bit:
    model_kwargs["load_in_4bit"] = True
elif training_run_args.load_as_gptq:
    model_kwargs["quantization_config"] = GPTQConfig(bits=4, disable_exllama=True)


# if training_run_args.bf16:
#     model_kwargs["torch_dtype"] = torch.bfloat16
# else:
#     model_kwargs["torch_dtype"] = torch.float16

def find_max_vram(min_buffer_mib=800):
    total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    suggestion = round((total_mem - 1000) / 1000) * 1000
    suggestion = min(suggestion, total_mem - min_buffer_mib)

    print(f"Model will target using {suggestion}MiB of VRAM")
    max_memory = {0: f'{suggestion}MiB'}

    return max_memory if len(max_memory) > 0 else None

if True:
    config = AutoConfig.from_pretrained(training_run_args.base_model, trust_remote_code=True)
    config.flash_attn = True
    config.flash_rotary = True
else:
    config = None

model = AutoModelForCausalLM.from_pretrained(
    training_run_args.base_model,
    config=config,
    trust_remote_code=True,
    device_map="auto",
    max_memory=find_max_vram(),
    local_files_only=True,
    **model_kwargs
)
tokenizer = AutoTokenizer.from_pretrained(training_run_args.base_model, trust_remote_code=True)

if training_run_args.add_pad_token:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

if training_run_args.add_chatml_tokens:
    tokenizer.add_tokens(["<|im_start|>", "<|im_end|>"])

# TODO: figure out how to actually use the modified tokenizer when loading the base model + lora
embeddings_len = math.ceil(len(tokenizer) / 32) * 32
if model.get_input_embeddings().num_embeddings < embeddings_len:
    model.resize_token_embeddings(embeddings_len)
else:
    model.tie_weights()

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
    model.print_trainable_parameters()

base_dir = "loras" if training_run_args.use_lora else "models"
model_dir = f"./{base_dir}/{training_run_args.run_name}"

training_args = TrainingArguments(
    per_device_train_batch_size=training_run_args.micro_batch_size,
    per_device_eval_batch_size=training_run_args.micro_batch_size,
    gradient_accumulation_steps=training_run_args.batch_size/training_run_args.micro_batch_size,
    gradient_checkpointing=training_run_args.gradient_checkpointing,
    evaluation_strategy="steps",
    eval_steps=training_run_args.eval_steps,
    save_strategy=("steps" if training_run_args.save_steps != -1 else "epoch"),
    save_steps=(training_run_args.save_steps if training_run_args.save_steps != -1 else None),
    logging_steps=5,
    output_dir=model_dir,
    num_train_epochs=training_run_args.epochs,
    save_total_limit=1,
    # dataloader_pin_memory=False,
    report_to="tensorboard",
    learning_rate=training_run_args.learning_rate,
    lr_scheduler_type=training_run_args.learning_rate_schedule,
    log_level="info",
    bf16=training_run_args.bf16,
    bf16_full_eval=training_run_args.bf16,
)

@dataclass
class DataCollatorForSupervisedFineTuning(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: AutoTokenizer
    ctx_length: int
    prompt_split: str

    def _tokenize(self, examples):
        return tokenizer(
                text=examples,
                max_length=self.ctx_length,
                truncation=True,
            )["input_ids"]
    
    def _pad(self, examples, pad_value):
        longest = max([len(ex) for ex in examples])
        result = []
        for example in examples:
            cur_len = len(example)
            result.append(example + [pad_value] * (longest - cur_len))

        return result

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        examples = [ instance["input_ids"] + self.tokenizer.eos_token for instance in instances ]
        prompts = [ self.prompt_split + example.split(self.prompt_split)[0] for example in examples ]
        input_ids = self._tokenize(examples)
        input_prompt_lengths = [ len(tokenized_prompt_ids) for tokenized_prompt_ids in self._tokenize(prompts)]

        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, input_prompt_lengths):
            label[:source_len] = [ -100 ]  * source_len

        input_ids = torch.LongTensor(self._pad(input_ids, self.tokenizer.pad_token_id))
        labels = torch.LongTensor(self._pad(labels, -100))

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

datasets = load_dataset("json", data_files={ "train": training_run_args.train_dataset, "test": training_run_args.test_dataset })
datasets = datasets.rename_column("text", "input_ids")
data_collator = DataCollatorForSupervisedFineTuning(tokenizer=tokenizer, ctx_length=training_run_args.ctx_size, prompt_split="<|im_start|>assistant")

import random
from torch.utils.data import SequentialSampler, Subset, RandomSampler
class RandomEvalSubsetTrainer(Trainer):
    def __init__(self, random_eval_sample_pct=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_eval_sample_pct = random_eval_sample_pct
        self.evaluate_full_dataset = False

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
        return RandomSampler(self.train_dataset, generator=torch.Generator(device='cpu'))

trainer = RandomEvalSubsetTrainer(
    model=model,
    args=training_args,
    # train_dataset=tokenized_train_dataset,
    # eval_dataset=tokenized_test_dataset,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
    data_collator=data_collator,
)

tensorboard_process = None
def kill_tensorboard():
    tensorboard_process.kill()

if training_run_args.run_tensorboard:
    import subprocess, atexit
    tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", model_dir])
    atexit.register(kill_tensorboard)

try:
    checkpoint = training_run_args.resume_from_checkpoint
    if checkpoint:
        trainer.train(checkpoint)
    else:
        trainer.train()

    trainer.evaluate_all()

    trainer.save_model()

    if training_run_args.use_lora and training_run_args.lora_merge:
        merged_model = model.merge_and_unload(progressbar=True)
        merged_model_dir = f"./models/{training_run_args.run_name}"
        merged_model.save_pretrained(merged_model_dir, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(merged_model_dir)
    else:
        tokenizer.save_pretrained(model_dir)

    if tensorboard_process:
        input("Training is finished. Press enter to quit tensorboard after the viewing results.")
        tensorboard_process.kill()
except Exception as e:
    print("Something bad happened! Try and save it?")
    import code, traceback
    traceback.print_exc()
    code.interact(local=locals())