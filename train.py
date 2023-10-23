import torch
import torch.nn.functional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import evaluate
import numpy as np

torch.set_default_device("cuda")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# TODO: convert this in to cmd line args
TRAIN_CTX_SIZE = 512 # The number of tokens to pad + truncate the input examples to
BATCH_SIZE = 8 # The simulated "batch size" that we will train on. will tweak gradient accumulations steps
MICRO_BATCH_SIZE = 2 # The actual batch size that will fit into VRAM on this machine
TRAINING_EPOCHS = 1 # The number of times to train the model on each example
LEARNING_RATE_START = 1e-4 # The starting learning rate (speed at which the model trains)
LEARNING_RATE_SCHEDULE = "cosine" # How fast the learning rate is reduced during training
RUN_NAME = "home-llm-rev7.3"
OUTPUT_DIR =f"./models/{RUN_NAME}"

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True).to(dtype=torch.bfloat16, device="cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer(tokenizer.pad_token)["input_ids"][0]
# tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
# model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

def tokenize_function(example):
    result = tokenizer(example['text'],
                       return_attention_mask=False,
                       padding=True, max_length=TRAIN_CTX_SIZE, truncation=True)
    
    return result


datasets = load_dataset("json", data_files={ "train": "data/home_assistant_train.json", "test": "data/home_assistant_test.json" })
tokenized_train_dataset = datasets["train"].map(tokenize_function, remove_columns=datasets["train"].column_names)
tokenized_test_dataset = datasets["test"].map(tokenize_function, remove_columns=datasets["test"].column_names)

training_args = TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    per_device_eval_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=BATCH_SIZE/MICRO_BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # evaluation_strategy="steps",
    # save_strategy="steps",
    # eval_steps=1000,
    # save_steps=1000,
    logging_steps=5,
    output_dir=OUTPUT_DIR,
    num_train_epochs=TRAINING_EPOCHS,
    save_total_limit=1,
    dataloader_pin_memory=False,
    report_to="tensorboard",
    learning_rate=LEARNING_RATE_START,
    lr_scheduler_type=LEARNING_RATE_SCHEDULE,
    log_level="info",
    bf16=True,
    bf16_full_eval=True,
)

class NoAttentionMaskDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        result = super().torch_call(examples)
        del result["attention_mask"]
        return result

data_collator = NoAttentionMaskDataCollator(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
)

checkpoint = None

if checkpoint:
    trainer.train(checkpoint)
else:
    trainer.train()

trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)