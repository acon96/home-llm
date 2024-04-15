import copy
import torch
import os
import random
from torch.utils.data import SequentialSampler, Subset, RandomSampler
from transformers import TrainerCallback, AutoTokenizer, Trainer
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import boto3
import os
import shutil

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
    logging_steps: int = field(default=5, metadata={"help": "Sets the number of steps in between log output for the training run"})
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

    dpo: bool = field(default=False, metadata={"help": "If set, performs Direct Preference Optimization instead of Supervised Fine Tuning"})
    beta: float = field(default=0.1, metadata={"help": "The implicit reward value used during DPO training"})

    add_pad_token: bool = field(default=False, metadata={"help": "If set, a pad token will be added to the tokenizer's vocabulary"})
    add_chatml_tokens: bool = field(default=False, metadata={"help": "If set, tokens for the ChatML format will be added specifically"})
    add_chatml_prompt_template: bool = field(default=False, metadata={"help": "If set, the ChatML prompt template will be set as the model's Jinja2 template"})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enables gradient checkpointing which saves quite a lot of VRAM"})

    sync_to_bucket: str = field(default=None, metadata={"help": "If set, checkpoints will be synced to the s3 bucket specified by this argument"})

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
        Should speed up training by skipping the final fine tuning part that doesn't affect accuracy much
        """
        return super().create_scheduler(int(num_training_steps * self.learning_rate_overshoot), optimizer=optimizer)

class UploadToS3Callback(TrainerCallback):
    def __init__(self, s3_bucket, s3_prefix, save_total_limit=None):
        self.s3_client = boto3.client('s3')
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.save_total_limit = save_total_limit

    def on_save(self, args, state, control, **kwargs):
        output_dir = kwargs['output_dir']
        checkpoint = os.path.basename(output_dir)
        
        # Upload current checkpoint
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                s3_path = os.path.join(self.s3_prefix, checkpoint, os.path.relpath(local_path, start=output_dir))
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_path)
                print(f"Uploaded {local_path} to s3://{self.s3_bucket}/{s3_path}")

        # Manage checkpoints in S3
        if self.save_total_limit:
            s3_checkpoints = self.list_s3_checkpoints()
            if len(s3_checkpoints) > self.save_total_limit:
                sorted_checkpoints = sorted(s3_checkpoints)
                to_delete = sorted_checkpoints[:-self.save_total_limit]
                for checkpoint in to_delete:
                    self.delete_checkpoint_from_s3(checkpoint)

        # Clean local checkpoints, keeping only the most recent
        all_checkpoints = [os.path.join(args.output_dir, d) for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))]
        if all_checkpoints:
            latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
            for checkpoint_dir in all_checkpoints:
                if checkpoint_dir != latest_checkpoint:
                    shutil.rmtree(checkpoint_dir)
                    print(f"Deleted local checkpoint {checkpoint_dir}")

    def list_s3_checkpoints(self):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix, Delimiter='/')
        return [prefix.get('Prefix').rstrip('/').split('/')[-1] for page in page_iterator for prefix in page.get('CommonPrefixes', [])]

    def delete_checkpoint_from_s3(self, checkpoint_name):
        resp = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=os.path.join(self.s3_prefix, checkpoint_name))
        for obj in resp.get('Contents', []):
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=obj['Key'])
            print(f"Deleted s3://{self.s3_bucket}/{obj['Key']}")