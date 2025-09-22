#!/usr/bin/env python3

import math
import copy
import torch
import os
import random
import time
import traceback
from torch.utils.data import SequentialSampler, Subset, RandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    HfArgumentParser, GPTQConfig, AutoConfig, TrainerCallback, BitsAndBytesConfig
from transformers.integrations.integration_utils import TensorBoardCallback
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


IS_DDP_ENABLED = "LOCAL_RANK" in os.environ
MULTI_GPU_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
MULTI_GPU_RANK = int(os.environ.get("RANK", "0"))
IS_MULTI_GPU = os.environ.get("RANK") != None
IS_MASTER_PROCESS = MULTI_GPU_RANK == 0

@dataclass
class TrainingRunArguments:
    run_name: str = field(metadata={"help": "The folder to save the output model under"})
    base_model: str = field(metadata={"help": "The base model to load for fine-tuning"})
    train_dataset: str = field(metadata={"help": "The JSON file containing the training dataset"})
    test_dataset: str = field(default=None, metadata={"help": "The JSON file containing the evaluation dataset"})
    dataset_processing_threads: int = field(default=None, metadata={"help": "The number of threads to use to tokenize the dataset"})
    ctx_size: int = field(default=2048, metadata={"help": "The number of tokens to pad & truncate the input examples to"})
    bf16: bool = field(default=False, metadata={"help": "If set, the model will the loaded and trained in bf16 instead of fp32"})
    batch_size: int = field(default=8, metadata={"help": "The simulated 'batch size' that we will train on. will tweak gradient accumulations steps"})
    micro_batch_size: int = field(default=2, metadata={"help": "The actual batch size that will fit into VRAM on this machine"})
    epochs: int = field(default=1, metadata={"help": "The number of times to train the model on each example"})
    learning_rate: float = field(default=1e-5, metadata={"help": "The starting learning rate (speed at which the model trains)"})
    learning_rate_schedule: str = field(default="cosine", metadata={"help": "How fast the learning rate is reduced during training"})
    learning_rate_warmup: float = field(default=0.0, metadata={"help": "The starting learning rate (speed at which the model trains)"})
    weight_decay: float = field(default=0.1, metadata={"help": "Weight Decay rate for regularization. Rate to reduce all neuron weights towards zero."})
    # dropout: float = field(default=0.01, metadata={"help": "Dropout percent for regularization. Determines the fraction of neurons randomly deactivated during training."})
    gradient_clip: float = field(default=1.0, metadata={"help": "Maximum gradient norm for clipping to prevent exploding gradients during training."})
    resume_from_checkpoint: str = field(default="", metadata={"help": "The name of the checkpoint to resume training from"})
    eval_steps: int = field(default=200, metadata={"help": "The number of steps in between evaluations of the model; set to -1 to evaluate every epoch"})
    save_steps: int = field(default=-1, metadata={"help": "The number of steps in between model checkpoints; set to -1 to save every epoch"})
    save_total_limit: int = field(default=1, metadata={"help": "The number of recent checkpoints of the model to save (not including the final model)"})
    logging_steps: int = field(default=5, metadata={"help": "Sets the number of steps in between log output for the training run"})
    group_by_length: bool = field(default=False, metadata={"help": "If enabled, the training data will be grouped by length to optimize use of padding. Runs from longest to shortest examples."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enables gradient checkpointing to saves VRAM at the cost of re-computing activations during the backwards pass"})
    pre_allocate_cuda_buffers: bool = field(default=True, metadata={"help": "If enabled, runs a forward and backward pass on the model before training to force pytorch to allocate the correct size CUDA buffers up front"})
    
    # Quantization
    load_in_8bit: bool = field(default=False, metadata={"help": "Set to load the base model in 8-bit mode using bitsandbytes"})
    load_in_4bit: bool = field(default=False, metadata={"help": "Set to load the base model in 4-bit mode using bitsandbytes"})
    load_as_gptq: bool = field(default=False, metadata={"help": "Set to load the base model as a GPTQ using AutoGPTQ"})
    
    # lora config
    use_lora: bool = field(default=False, metadata={"help": "If set, then the trained model will be a LoRA"})
    lora_rank: int = field(default=4, metadata={"help": "Rank which determines LoRA matrix size. Rank typically starts at 8 but can go up to 256. Higher ranks can store more information but increase the computational and memory cost of LoRA."})
    lora_alpha: int = field(default=32, metadata={"help": "Alpha a scaling factor for updates. Alpha directly impacts the adapters contribution and is often set to 1x or 2x the rank value."})
    lora_dropout: float = field(default=0.05)
    lora_modules: str = field(default=None, metadata={"help": "Target modules: LoRA can be applied to various model components, including attention mechanisms (Q, K, V matrices), output projections, feed-forward blocks, and linear output layers. While initially focused on attention mechanisms, extending LoRA to other components has shown benefits. However, adapting more modules increases the number of trainable parameters and memory needs."})
    lora_modules_to_save: str = field(default=None, metadata={"help": "Additional modules to save"})
    lora_merge: bool = field(default=False, metadata={"help": "If set, the Lora will be merged back into the base model an saved"})

    # dpo config
    dpo: bool = field(default=False, metadata={"help": "If set, performs Direct Preference Optimization instead of Supervised Fine Tuning"})
    beta: float = field(default=0.1, metadata={"help": "The implicit reward value used during DPO training"})
    dpo_loss: str = field(default="sigmoid", metadata={"help": "The loss type to use during DPO training"})

    # token options
    add_pad_token: bool = field(default=False, metadata={"help": "If set, a pad token will be added to the tokenizer's vocabulary"})
    add_chatml_tokens: bool = field(default=False, metadata={"help": "If set, tokens for the ChatML format will be added specifically"})
    add_chatml_prompt_template: bool = field(default=False, metadata={"help": "If set, the ChatML prompt template will be set as the model's Jinja2 template"})
    prefix_ids: str = field(default=None, metadata={"help": "Determine the prefix tokens that surround the response from the assistant for SFT if model can not correctly recognize response."})
    suffix_ids: str = field(default=None, metadata={"help": "Determine the suffix tokens that surround the response from the assistant for SFT if model can not correctly recognize response."})

    # custom trainer tweaks
    sync_to_bucket: str = field(default=None, metadata={"help": "If set, checkpoints will be synced to the s3 bucket specified by this argument"})
    bucket_save_limit: int = field(default=None, metadata={"help": "The number of recent checkpoints of the model to save in S3 (not including the final model)"})
    flops_baseline: str = field(default=None, metadata={"help": "The baseline flops for the GPUs used for the training run. Outputs MFU"})


class UploadToS3Callback(TrainerCallback):
    def __init__(self, s3_bucket, s3_prefix, save_total_limit=None):
        import boto3
        self.s3_client = boto3.client('s3')
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.save_total_limit = save_total_limit

    def on_save(self, args, state, control, **kwargs):

        # Upload current checkpoint
        checkpoint = f"checkpoint-{state.global_step}"
        output_dir = f"{args.output_dir}/{checkpoint}"
        
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                s3_path = os.path.join(self.s3_prefix, checkpoint, os.path.relpath(local_path, start=output_dir))
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_path)
                print(f"Uploaded {local_path} to s3://{self.s3_bucket}/{s3_path}")

        # Delete prior checkpoints from S3
        if self.save_total_limit:
            s3_checkpoints = self.list_s3_checkpoints()
            if len(s3_checkpoints) > self.save_total_limit:
                sorted_checkpoints = sorted(s3_checkpoints)
                to_delete = sorted_checkpoints[:-self.save_total_limit]
                for checkpoint in to_delete:
                    self.delete_checkpoint_from_s3(checkpoint)

    def list_s3_checkpoints(self):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix + '/', Delimiter='/')
        return [prefix.get('Prefix').rstrip('/').split('/')[-1] for page in page_iterator for prefix in page.get('CommonPrefixes', [])]

    def delete_checkpoint_from_s3(self, checkpoint_name):
        resp = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=os.path.join(self.s3_prefix, checkpoint_name))
        for obj in resp.get('Contents', []):
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=obj['Key'])
            print(f"Deleted s3://{self.s3_bucket}/{obj['Key']}")

class MFUCallback(TrainerCallback):
    def __init__(self, peak_flops):
        self.total_iterations = 0
        self.start_time = time.time()
        self.flops_promised = peak_flops
        self.last_total_flos = 0

    def on_log(self, args, state, control, **kwargs):
        if state.global_step == 0:  # Avoid computation at the very beginning
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Calculate and log MFU
        new_flops = state.total_flos - self.last_total_flos
        kwargs['logs']['mfu'] = round(new_flops / elapsed_time / self.flops_promised, 4)

        self.start_time = current_time
        self.last_total_flos = state.total_flos


def ddp_print(*args, **kwargs):
    if not IS_DDP_ENABLED or IS_MASTER_PROCESS:
        print(*args, **kwargs)

def find_max_vram(min_buffer_mib=800):
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        gpu_properties = torch.cuda.get_device_properties(i)
        total_memory_mib = (gpu_properties.total_memory / (1000 * 1000))
        suggestion = max(total_memory_mib - 1000, min_buffer_mib)

        ddp_print(f"GPU {i}: {gpu_properties.name}, Total Memory: {gpu_properties.total_memory / (1024**3):.2f} GB")
        ddp_print(f"Model will target using {suggestion}MiB of VRAM on GPU {i}")
        max_memory[i] = f'{suggestion}MiB'

    return max_memory


class DataCollatorForSupervisedFineTuning(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: AutoTokenizer
    prompt_split: str
    response_prefix: str
    response_suffix: str
    prefix_ids: list[int]
    suffix_ids: list[int]

    def __init__(self, *, tokenizer: AutoTokenizer, prefix_ids: Optional[list[int]] = None, suffix_ids: Optional[list[int]] = None):
        
        self.tokenizer = tokenizer
        if not prefix_ids and not suffix_ids:
            assistant_prompt = tokenizer.apply_chat_template(
                conversation=[{"role": "assistant", "content":  r"%%%%%%%%%%%%%%%%"}], 
                tokenize=False).split( r"%%%%%%%%%%%%%%%%")
            
            self.response_prefix = assistant_prompt[0]
            self.response_suffix = assistant_prompt[1]

            # check for inserted system prompt and remove it
            if tokenizer.eos_token in self.response_prefix:
                self.response_prefix = self.response_prefix.split(tokenizer.eos_token)[1].lstrip()

            # some chat templates ALWAYS add the bos token
            if tokenizer.bos_token in self.response_prefix:
                self.response_prefix = self.response_prefix.replace(tokenizer.bos_token, "")

        if prefix_ids:
            self.prefix_ids = prefix_ids
        else:
            self.prefix_ids = self.tokenizer(self.response_prefix, add_special_tokens=False)["input_ids"]

        if suffix_ids:
            self.suffix_ids = suffix_ids
        else:
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
                if end - start == len(label) - 1:
                    print("warning! example had no assistant response in it!")
                    print(input_ids)
                label[start:end] = [-100] * (end - start)

        input_ids = torch.LongTensor(self._pad(input_ids, self.tokenizer.pad_token_id or self.tokenizer.eos_token_id))
        labels = torch.LongTensor(self._pad(labels, -100))

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
        )


def tokenize_raw_example(batch, tokenizer=None, training_run_args=None):
    return tokenizer(
        text=batch["text"],
        max_length=training_run_args.ctx_size,
        truncation=True,
        add_special_tokens=False,
    )

def tokenize_sharegpt_example(batch, tokenizer=None, training_run_args=None):
    # TODO: figure out how to properly batch this
    result = []
    for example in batch["conversations"]:
        conversation = [ { "role": x["from"], "content": x["value"] }  for x in example ]
        result.append(
            tokenizer.apply_chat_template(
                conversation=conversation,
                max_length=training_run_args.ctx_size,
                truncation=True,
            )
        )

    return {"input_ids": result}

def template_dpo_example(batch, tokenizer=None, training_run_args=None):
    # TODO: figure out how to properly batch this
    result = []
    for example in zip(batch["system"], batch["question"]):
        conversation = [ 
            { "role": "system", "content": example[0] },
            { "role": "user", "content": example[1] },
        ]
        result.append(
            tokenizer.apply_chat_template(
                conversation=conversation,
                max_length=training_run_args.ctx_size,
                truncation=True,
                tokenize=False,
                add_generation_prompt=True
            )
        )

    return {"prompt": result}


class CustomSFTTrainer(Trainer):
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
        Should improve training efficiency by skipping the final fine tuning part that doesn't affect accuracy much
        """
        return super().create_scheduler(int(num_training_steps * self.learning_rate_overshoot), optimizer=optimizer)
    
    def floating_point_ops(self, inputs):
        config = self.model.config
        examples_length = len(inputs["input_ids"][0])
        batch_size = len(inputs["input_ids"])

        # mfu is approximated using throughput and param count
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # there are 3 passes of a NN (fwd, bwd, delta) - we multiply by 3 ie 2 * 3 * n_param
        # this gets us FLOPs / token
        flops_per_token = 2 * sum(p.numel() for p in self.model.parameters())
        flops_per_seq = flops_per_token * examples_length

        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_seq = config.num_hidden_layers * 2 * 2 * (config.hidden_size * (examples_length**2))

        # there are 2 ops in bwd pass and 1 in fwd pass so we mult by 3
        result = (3 * flops_per_seq + 3 * attn_flops_per_seq) * batch_size
        return result


def do_training_run(training_run_args: TrainingRunArguments):
    # validate args + build model kwargs
    if sum([training_run_args.load_in_8bit, training_run_args.load_in_4bit, training_run_args.load_as_gptq]) > 1:
        raise Exception("Please select exactly one of 'load_in_8bit', 'load_in_4bit', or 'load_as_gptq")

    model_kwargs = {}
    if training_run_args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif training_run_args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif training_run_args.load_as_gptq:
        model_kwargs["quantization_config"] = GPTQConfig(bits=4, disable_exllama=True)

    if training_run_args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif training_run_args.use_lora and "quantization_config" not in model_kwargs:
        model_kwargs["torch_dtype"] = torch.float16
    else:
        # auto detect 'best' format with fallback to fp32
        model_kwargs["torch_dtype"] = "auto"

    # model_kwargs["resid_pdrop"] = training_run_args.dropout
    model_kwargs["use_cache"] = False

    if not IS_DDP_ENABLED:
        model_kwargs["device_map"] = "auto"

    # load the model
    ddp_print(f"Loading model '{training_run_args.base_model}'...")

    model = AutoModelForCausalLM.from_pretrained(
        training_run_args.base_model,
        max_memory=find_max_vram(),
        token=os.environ.get("HF_TOKEN"),
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(training_run_args.base_model, token=os.environ.get("HF_TOKEN"))

    # mess with tokens + prompt template
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

    # resize embeddings if added tokens require it
    embeddings_len = math.ceil(len(tokenizer) / 32) * 32
    if model.get_input_embeddings().num_embeddings < embeddings_len:
        model.resize_token_embeddings(embeddings_len)
    else:
        model.tie_weights()

    # create LoRA model if config says so
    original_model = model
    peft_config = None
    if training_run_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        ddp_print("Creating LoRA for model...")
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

    # set up HuggingFace Trainer args
    training_kwargs = {}

    if training_run_args.test_dataset:
        training_kwargs.update({
            "per_device_eval_batch_size": training_run_args.micro_batch_size,
            "eval_strategy": ("steps" if training_run_args.eval_steps != -1 else "epoch"),
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
        report_to='none',
        learning_rate=training_run_args.learning_rate,
        lr_scheduler_type=training_run_args.learning_rate_schedule,
        warmup_ratio=training_run_args.learning_rate_warmup,
        log_level="info",
        bf16=training_run_args.bf16,
        group_by_length=training_run_args.group_by_length,
        # include_num_input_tokens_seen=True,
        **training_kwargs,
    )

    # set up trainer callbacks
    training_callbacks = []
    if training_run_args.sync_to_bucket:
        training_callbacks.append(UploadToS3Callback(
            s3_bucket=training_run_args.sync_to_bucket,
            s3_prefix=training_run_args.run_name,
            save_total_limit=training_run_args.bucket_save_limit if training_run_args.bucket_save_limit else training_run_args.save_total_limit
        ))

    if training_run_args.flops_baseline:
        # A100 40/80GB GPU bfloat16 peak flops is 312 TFLOPS (312e12)
        # 4090 24GB GPU bfloat16 peak flops is 165.2 TFLOPS (1652e11)
        # A40 48GB GPU bfloat16 peak flops is 149.7 TFLOPS (149.7e11)
        # 3090 24GB GPU bfloat16 peak flops is 71 TFLOPS (71e12)
        training_callbacks.append(MFUCallback(peak_flops=float(training_run_args.flops_baseline)))

    # log to tensorboard (but after MFU)
    training_callbacks.append(TensorBoardCallback())

    if not training_run_args.dpo:
        ddp_print("Loading dataset...")
        data_files = { "train": training_run_args.train_dataset }
        if training_run_args.test_dataset:
            data_files["test"] = training_run_args.test_dataset
        datasets = load_dataset("json", data_files=data_files)
        
        # prepare the dataset
        ddp_print("Tokenizing datasets...")

        if "text" in datasets["train"].column_names:
            tokenize_function = tokenize_raw_example
            columns_to_remove = ["text"]
        elif "conversations" in datasets["train"].column_names:
            tokenize_function = tokenize_sharegpt_example
            columns_to_remove = ["conversations"]
        else:
            raise Exception("Unknown dataset input format (not raw corpus or sharegpt)")

        tokenized_test_dataset = None
        num_proc = None
        if training_run_args.dataset_processing_threads:
            num_proc = training_run_args.dataset_processing_threads // MULTI_GPU_WORLD_SIZE
        tokenized_train_dataset = datasets["train"].map(tokenize_function, batched=True, num_proc=num_proc, fn_kwargs={"tokenizer": tokenizer, "training_run_args": training_run_args}).remove_columns(columns_to_remove)
        if training_run_args.test_dataset:
            tokenized_test_dataset = datasets["test"].map(tokenize_function, batched=True, num_proc=num_proc, fn_kwargs={"tokenizer": tokenizer, "training_run_args": training_run_args}).remove_columns(columns_to_remove)

        example_lengths = [ len(example) for example in tokenized_train_dataset["input_ids"] ]
        tokens_in_train_set, longest_example = sum(example_lengths), max(example_lengths)
        ddp_print(f"Train dataset has {int(tokens_in_train_set / 1000000)}M tokens. Longest Example: {longest_example} tokens")
        
        provided_prefix_ids = None
        provided_suffix_ids = None
        try:
            if training_run_args.prefix_ids:
                provided_prefix_ids = [ int(x) for x in training_run_args.prefix_ids.split(",") ]
            if training_run_args.suffix_ids:
                provided_suffix_ids = [ int(x) for x in training_run_args.suffix_ids.split(",") ]
        except ValueError as ex:
            print(f"Error parsing prefix_ids or suffix_ids: '{ex}'")
            exit(-1)

        trainer = CustomSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_test_dataset,
            data_collator=DataCollatorForSupervisedFineTuning(
                tokenizer=tokenizer,
                prefix_ids=provided_prefix_ids,
                suffix_ids=provided_suffix_ids,
            ),
            callbacks=training_callbacks,
        )
    else:
        raise NotImplementedError("DPO Trainer doesn't work yet!")
        # from trl import DPOTrainer
        # max_prompt_length = 0

        # train_dataset = datasets["train"].map(lambda x: { "prompt_len": len(x["system"]) })

        # test_dataset = None
        # if training_run_args.test_dataset:
        #     test_dataset = datasets["test"]

        # max_prompt_length = max(train_dataset["prompt_len"])

        # print("Templating DPO Examples...")
        # templated_test_dataset = None
        # templated_train_dataset = train_dataset.map(template_dpo_example, batched=True).remove_columns(["system", "question"])
        # if training_run_args.test_dataset:
        #     templated_test_dataset = datasets["test"].map(template_dpo_example, batched=True).remove_columns(["system", "question"])

        # # tokenizer.model_input_names = [ "chosen_input_ids" ]

        # # group_by_length doesn't work here
        # # templated_train_dataset = templated_train_dataset.sort("prompt_len", reverse=True)

        # training_args.length_column_name = "prompt_len"
        # model.enable_input_require_grads()

        # trainer = DPOTrainer(
        #     model,
        #     ref_model=None,
        #     # ref_model=original_model,
        #     peft_config=peft_config,
        #     args=training_args,
        #     beta=training_run_args.beta,
        #     loss_type=training_run_args.dpo_loss,
        #     train_dataset=templated_train_dataset,
        #     eval_dataset=templated_test_dataset,
        #     tokenizer=tokenizer,
        #     max_length=training_run_args.ctx_size,
        #     max_prompt_length=max_prompt_length,
        #     truncation_mode="keep_start",
        #     callbacks=training_callbacks,
        # )

    try:
        trainer.train(resume_from_checkpoint=training_run_args.resume_from_checkpoint if training_run_args.resume_from_checkpoint else None)

        if training_run_args.test_dataset:
            trainer.evaluate_all()

        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

        if training_run_args.use_lora and training_run_args.lora_merge:
            trainer.save_model() # save lora

            merged_model = model.merge_and_unload(progressbar=True)
            merged_model_dir = f"./models/{training_run_args.run_name}"
            merged_model.save_pretrained(merged_model_dir, safe_serialization=True, max_shard_size="2GB")
            
            tokenizer.save_pretrained(merged_model_dir)
        else:
            trainer.save_model()
            tokenizer.save_pretrained(model_dir)

        if training_run_args.sync_to_bucket:
            import boto3
            s3_client = boto3.client('s3')

            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    s3_path = os.path.join(training_run_args.run_name, os.path.relpath(local_path, start="."))
                    s3_client.upload_file(local_path, training_run_args.sync_to_bucket, s3_path)
                    print(f"Uploaded {local_path} to s3://{training_run_args.sync_to_bucket}/{s3_path}")

    except Exception as ex:
        if trainer.is_fsdp_enabled:
            raise ex # this doesn't play nice with FSDP so don't even try
        
        traceback.print_exc()
        
        if input("Something bad happened! Try and save it? (Y/n) ").lower().startswith("y"):
            trainer._save_checkpoint(model, None)
            print("Saved Checkpoint!")
    
if __name__ == "__main__":
    parser = HfArgumentParser([TrainingRunArguments])
    training_run_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    do_training_run(training_run_args)
