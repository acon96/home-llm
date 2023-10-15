import torch
import torch.nn.functional
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import evaluate
import numpy as np

torch.set_default_device("cuda")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

TRAIN_CTX_SIZE = 512

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True).to(dtype=torch.bfloat16, device="cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# pad_token_id = tokenizer(tokenizer.pad_token)["input_ids"][0]
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(example):
    result = tokenizer(example['text'],
                       return_attention_mask=False,
                       padding=True, max_length=TRAIN_CTX_SIZE, truncation=True)
    
    return result


datasets = load_dataset("json", data_files="data/home_assistant_examples.json")
datasets = datasets["train"].train_test_split(test_size=0.1)
tokenized_train_dataset = datasets["train"].map(tokenize_function, remove_columns=datasets["train"].column_names)
tokenized_test_dataset = datasets["test"].map(tokenize_function, remove_columns=datasets["test"].column_names)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # evaluation_strategy="steps",
    # save_strategy="steps",
    # eval_steps=1000,
    # save_steps=1000,
    logging_steps=10,
    output_dir="./models/training",
    num_train_epochs=4,
    save_total_limit=2,
    dataloader_pin_memory=False,
    report_to="tensorboard",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
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

accuracy_netric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_netric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

checkpoint = None

if checkpoint:
    trainer.train(checkpoint)
else:
    trainer.train()

trainer.save_model()