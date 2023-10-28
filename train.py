import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, HfArgumentParser, GPTQConfig
from datasets import load_dataset
from dataclasses import dataclass, field

torch.set_default_device("cuda")

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
    --run_name home-llama2-13b-rev1 \
    --base_model TheBloke/Llama-2-13B-GPTQ \
    --train_dataset data/home_assistant_train.json \
    --test_dataset data/home_assistant_test.json \
    --load_as_gptq --use_lora --lora_rank 16 --gradient_checkpointing \
    --add_pad_token --bf16 --micro_batch_size 1 --learning_rate 2e-5
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
    learning_rate_schedule: str = field(default="cosine", metadata={"help": "How fast the learning rate is reduced during training"})\
    
    # Quantization
    load_in_8bit: bool = field(default=False, metadata={"help": "Set to load the base model in 8-bit mode using bitsandbytes"})
    load_in_4bit: bool = field(default=False, metadata={"help": "Set to load the base model in 4-bit mode using bitsandbytes"})
    load_as_gptq: bool = field(default=False, metadata={"help": "Set to load the base model as a GPTQ using AutoGPTQ"})
    
    use_lora: bool = field(default=False, metadata={"help": "If set, then the trained model will be a LoRA"})
    lora_rank: int = field(default=4)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)

    disable_attention_mask: bool = field(default=False, metadata={"help": "If set, disables the attention mask generated to ignore pad tokens."})
    add_pad_token: bool = field(default=False, metadata={"help": "If set, a pad token will be added to the tokenizer's vocabulary"})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enables gradient checkpointing which saves quite a lot of VRAM"})

    run_tensorboard: bool = field(default=True, metadata={"help": "If set, will tensorboard in the background to monitor training progress"})

parser = HfArgumentParser([TrainingRunArguments])
training_run_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

if sum([training_run_args.load_in_8bit, training_run_args.load_in_4bit]) > 1:
    raise Exception("Please select exactly one of 'bf16', 'load_in_8bit', 'load_in_4bit', or 'load_as_gptq")

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

# TODO: figure out how to actually use the modified tokenizer when loading the base model + lora
embeddings_len = math.ceil(len(tokenizer) / 32) * 32
if model.get_input_embeddings().num_embeddings < embeddings_len:
    model.resize_token_embeddings(embeddings_len)
else:
    model.tie_weights()

if training_run_args.use_lora:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    print("Creating LoRA for model...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=training_run_args.lora_rank,
        lora_alpha=training_run_args.lora_alpha,
        lora_dropout=training_run_args.lora_dropout,
        target_modules=None,
    )
    if training_run_args.load_in_8bit or training_run_args.load_in_4bit or training_run_args.load_as_gptq:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_run_args.gradient_checkpointing
        )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

def tokenize_function(example):
    result = tokenizer(example['text'] + tokenizer.eos_token,
                       padding=True,
                       max_length=training_run_args.ctx_size,
                       truncation=True)
    
    return result

print("Tokenizing Dataset...")
datasets = load_dataset("json", data_files={  "train": training_run_args.train_dataset, "test": training_run_args.test_dataset })
tokenized_train_dataset = datasets["train"].map(tokenize_function, remove_columns=datasets["train"].column_names)
tokenized_test_dataset = datasets["test"].map(tokenize_function, remove_columns=datasets["test"].column_names)

base_dir = "loras" if training_run_args.use_lora else "models"
model_dir = f"./{base_dir}/{training_run_args.run_name}"

training_args = TrainingArguments(
    per_device_train_batch_size=training_run_args.micro_batch_size,
    per_device_eval_batch_size=training_run_args.micro_batch_size,
    gradient_accumulation_steps=training_run_args.batch_size/training_run_args.micro_batch_size,
    # evaluation_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    # save_strategy="steps",
    # save_steps=1000,
    logging_steps=5,
    output_dir=model_dir,
    num_train_epochs=training_run_args.epochs,
    save_total_limit=1,
    dataloader_pin_memory=False,
    report_to="tensorboard",
    learning_rate=training_run_args.learning_rate,
    lr_scheduler_type=training_run_args.learning_rate_schedule,
    log_level="info",
    bf16=training_run_args.bf16,
    bf16_full_eval=training_run_args.bf16,
)

class NoAttentionMaskDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        result = super().torch_call(examples)
        del result["attention_mask"]
        return result

if training_run_args.disable_attention_mask:
    data_collator = NoAttentionMaskDataCollator(tokenizer, mlm=False)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# TODO: ignore user input when training by masking the input properly
# @dataclass
# class CustomDataCollator:
#     tokenizer: AutoTokenizer
#     train_ctx_size: int
#     def __call__(self, features, **kwargs):
#         for feature in features:
# data_collator = CustomDataCollator(tokenizer=tokenizer)

import random
from torch.utils.data import SequentialSampler, Subset
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

trainer = RandomEvalSubsetTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
)

tensorboard_process = None
if training_run_args.run_tensorboard:
    import subprocess
    tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", model_dir])

checkpoint = None

if checkpoint:
    trainer.train(checkpoint)
else:
    trainer.train()

trainer.evaluate_all()

trainer.save_model()

if not training_run_args.use_lora:
    tokenizer.save_pretrained(model_dir)

if tensorboard_process:
    input("Training is finished. Press enter to quit tensorboard after the viewing results.")
    tensorboard_process.kill()