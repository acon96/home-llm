import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, HfArgumentParser
from datasets import load_dataset
from dataclasses import dataclass, field

torch.set_default_device("cuda")

"""
python3 train.py \
    --run_name home-llm-rev9.1 \
    --base_model microsoft/phi-1_5 \
    --train_dataset data/home_assistant_train.json \
    --test_dataset data/home_assistant_test.json
"""

@dataclass
class TrainingRunArguments:
    run_name: str = field(metadata={"help": "The folder to save the output model under"})
    train_dataset: str = field(metadata={"help": "The JSON file containing the training dataset"})
    test_dataset: str = field(metadata={"help": "The JSON file containing the evaluation dataset"})
    base_model: str = field(metadata={"help": "The base model to load for fine-tuning"})
    ctx_size: int = field(default=512, metadata={"help": "The number of tokens to pad & truncate the input examples to"})
    bf16: bool = field(default=True, metadata={"help": "If set, the model will the loaded and trained in bf16 instead of fp16"})
    batch_size: int = field(default=8, metadata={"help": "The simulated 'batch size' that we will train on. will tweak gradient accumulations steps"})
    micro_batch_size: int = field(default=2, metadata={"help": "The actual batch size that will fit into VRAM on this machine"})
    epochs: int = field(default=1, metadata={"help": "The number of times to train the model on each example"})
    learning_rate: float = field(default=1e-5, metadata={"help": "The starting learning rate (speed at which the model trains)"})
    learning_rate_schedule: str = field(default="cosine", metadata={"help": "How fast the learning rate is reduced during training"})
    load_in_8bit: bool = field(default=False, metadata={"help": "Set to load the base model in 8-bit mode using bitsandbytes"})
    
    use_lora: bool = field(default=False, metadata={"help": "If set, then the trained model will be a LoRA"})
    lora_rank: int = field(default=4)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)

parser = HfArgumentParser([TrainingRunArguments])
training_run_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

if training_run_args.load_in_8bit and training_run_args.bf16:
    raise Exception("Cannot use load_in_8bit and bf16 flags at the same time!")

# TODO: write a proper evaluation script

print(f"Loading model '{training_run_args.base_model}'...")

model_kwargs = {}
if training_run_args.load_in_8bit:
    model_kwargs["load_int_8bit"] = True
elif training_run_args.bf16:
    model_kwargs["torch_dtype"] = torch.bfloat16
else:
    model_kwargs["torch_dtype"] = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    training_run_args.base_model,
    trust_remote_code=True,
    **model_kwargs
)
tokenizer = AutoTokenizer.from_pretrained(training_run_args.base_model, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

training_callbacks = None
# if training_run_args.use_lora:
#     raise NotImplementedError("Need to fix the callback thing still")

if training_run_args.use_lora:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    print("Creating LoRA for model...")
    class SavePeftModelCallback(transformers.TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_folder_name = f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            checkpoint_folder = os.path.join(args.output_dir, checkpoint_folder_name)
            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)
            return control
    training_callbacks = [SavePeftModelCallback]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=training_run_args.lora_rank,
        lora_alpha=training_run_args.alpha,
        lora_dropout=training_run_args.droput,
        target_modules=None,
    )
    if training_run_args.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

def tokenize_function(example):
    result = tokenizer(example['text'] + tokenizer.eos_token,
                       return_attention_mask=False,
                       padding=True, max_length=training_run_args.ctx_size, truncation=True)
    
    return result

print("Tokenizing Dataset...")
datasets = load_dataset("json", data_files={  "train": training_run_args.train_dataset, "test": training_run_args.test_dataset })
tokenized_train_dataset = datasets["train"].map(tokenize_function, remove_columns=datasets["train"].column_names)
tokenized_test_dataset = datasets["test"].map(tokenize_function, remove_columns=datasets["test"].column_names)

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
    output_dir=f"./models/{training_run_args.run_name}",
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

data_collator = NoAttentionMaskDataCollator(tokenizer, mlm=False)

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
    callbacks=training_callbacks,
)

checkpoint = None

if checkpoint:
    trainer.train(checkpoint)
else:
    trainer.train()

trainer.evaluate_all()

# trainer.save_state()
trainer.save_model()
tokenizer.save_pretrained(f"./models/{training_run_args.run_name}")