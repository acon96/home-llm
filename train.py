#!/usr/bin/env python3

import math
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, \
    HfArgumentParser, GPTQConfig
from trl import DPOTrainer
from datasets import load_dataset

from train_util import TrainingRunArguments, DataCollatorForSupervisedFineTuning, CustomSFTTrainer, \
    UploadToS3Callback

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
    --run_name stablehome-1_6b-rev3 \
    --base_model stabilityai/stablelm-2-zephyr-1_6b \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 1e-5 --batch_size 32 \
    --micro_batch_size 2 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 --save_steps 100 --save_total_limit 20
"""

"""
python3 train.py \
    --run_name stablehome-3b-rev8 \
    --base_model stabilityai/stablelm-zephyr-3b \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 1e-5 --batch_size 128 --epochs 2 \
    --micro_batch_size 8 --gradient_checkpointing \
    --ctx_size 2048 \
    --save_steps 50 --save_total_limit 20 --eval_steps 100 --logging_steps 2 \
    --use_lora --lora_rank 64 --lora_alpha 128 --lora_modules up_proj,down_proj,q_proj,v_proj,o_proj --lora_merge
"""

"""
accelerate launch --config_file fsdp_config.yaml train.py \
    --run_name stablehome-3b-rev9 \
    --base_model stabilityai/stablelm-zephyr-3b \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --learning_rate 1e-5 --batch_size 64 --epochs 1 \
    --micro_batch_size 2 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 \
    --save_steps 50 --save_total_limit 5 --eval_steps 100 --logging_steps 2
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

"""
python3 train.py \
    --run_name tinyhome-rev2 \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 2e-5 --batch_size 32 \
    --micro_batch_size 4 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 --save_steps 100 --save_total_limit 10
"""

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
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total_mem = (torch.cuda.get_device_properties(i).total_memory / (1024 * 1024))
        suggestion = round((total_mem - 1000) / 1000) * 1000
        suggestion = min(suggestion, total_mem - min_buffer_mib)

        print(f"Model will target using {suggestion}MiB of VRAM on GPU {i}")
        max_memory[i] = f'{suggestion}MiB'

    return max_memory

model = AutoModelForCausalLM.from_pretrained(
    training_run_args.base_model,
    trust_remote_code=True,
    # device_map="auto",
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
    logging_steps=training_run_args.logging_steps, 
    output_dir=model_dir,
    num_train_epochs=training_run_args.epochs,
    save_total_limit=training_run_args.save_total_limit,
    report_to="tensorboard",
    learning_rate=training_run_args.learning_rate,
    lr_scheduler_type=training_run_args.learning_rate_schedule,
    warmup_ratio=training_run_args.learning_rate_warmup,
    log_level="info",
    bf16=training_run_args.bf16,
    group_by_length=training_run_args.group_by_length,
    **training_kwargs,
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

training_callbacks = []
if training_run_args.sync_to_bucket:
    training_callbacks.append(UploadToS3Callback(
        s3_bucket=training_run_args.sync_to_bucket,
        s3_prefix=training_run_args.run_name,
        save_total_limit=training_run_args.save_total_limit
    ))

if not training_run_args.dpo:
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
    tokenized_train_dataset = datasets["train"].map(tokenize_function, batched=True, num_proc=os.cpu_count()).remove_columns(columns_to_remove)
    if training_run_args.test_dataset:
        tokenized_test_dataset = datasets["test"].map(tokenize_function, batched=True, num_proc=os.cpu_count()).remove_columns(columns_to_remove)

    example_lengths = [ len(example) for example in tokenized_train_dataset["input_ids"] ]
    tokens_in_train_set, longest_example = sum(example_lengths), max(example_lengths)
    print(f"Train dataset has {int(tokens_in_train_set / 1000000)}M tokens. Longest Example: {longest_example} tokens")

    data_collator = DataCollatorForSupervisedFineTuning(tokenizer=tokenizer)
    # fix for tinyllama not detecting split properly
    # data_collator = DataCollatorForSupervisedFineTuning(
    #     tokenizer=tokenizer,
    #     prefix_ids=[29966, 29989, 465, 22137, 29989, 29958, 13],
    #     suffix_ids=[2],
    # )

    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        callbacks=training_callbacks,
    )
else:
    max_prompt_length = 0

    train_dataset = datasets["train"].map(lambda x: { "prompt_len": len(x["system"]) })
    test_dataset = datasets["test"]

    max_prompt_length = max(train_dataset["prompt_len"])

    trainer = DPOTrainer(
        model,
        args=training_args,
        beta=training_run_args.beta,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
        max_length=training_run_args.ctx_size,
        max_prompt_length=max_prompt_length,
        generate_during_eval=True,
        callbacks=training_callbacks,
    )

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

except Exception as ex:
    if len(torch.cuda.device_count()) > 1:
        raise ex # this doesn't play nice with FSDP so don't even try
    
    print("Something bad happened! Try and save it?")
    import code, traceback
    traceback.print_exc()
    code.interact(local=locals())