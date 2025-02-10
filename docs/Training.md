# Training new model

## Initial requirements

Before you start training you model you should prepare data. One off the way to prepare data is run script `data/generate_home_assistant_data.py`. Currently this script support english, french, german, polish and spanish language. This is not the only way and you can prepare your own data. More details can be found in the file [HOW TO PREPARE DATA](../data/README.md).


## Training

Before you will prepare your local environment and install all necessary libraries. You will need a graphics card with significant VRAM (16GB+) to effectively train the new model. If you have less VRAM you may be able to perform a LoRA fine tuning, but this will be less effective.

Start by installing system dependencies (assumes Ubuntu):
`sudo apt-get install python3-dev`

Then create a Python virtual environment and install all necessary libraries:
```
python3 -m venv .train_data
source ./.train_data/bin/activate
pip3 install datasets==2.20.0 dataclasses==0.6 transformers==4.43.3 torch==2.4.0 accelerate==0.33.0 tensorboard==2.17.0 peft==0.12.0 bitsandbytes==0.43.3 trl==0.9.6
```

### Prepare your model

Select which model you need to train: e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Remember that the choice of model depends on the VRAM your graphics card. If the model is larger, it is worth using the LoRA configuration which will allow you to train your model.

Run train.py script to start Fine Tuning with selected model and prepared data. It is worth experimenting with several parameters to choose the best model. Few example with some params you can find below.

### Example training runs

#### Phi Series
**Phi Modules (for lora fine-tuning)**
- MLP: fc1,fc2
- MHA: q_proj,v_proj,k_proj,dense
- Embeddings: embed_tokens (input) lm_head (output)

```console
python3 train.py \
    --run_name home-phi3-mini-rev1 \
    --base_model microsoft/Phi-3-mini-4k-instruct \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 5e-6 --batch_size 32 \
    --micro_batch_size 8 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 --save_steps 100 --save_total_limit 10
```

```console
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
```

```console
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
```

#### StableLM series
**StableLM modules (for lora fine-tuning)**
- MLP: up_proj,down_proj,gate_proj
- MHA: q_proj,v_proj,k_proj,o_proj
- Embeddings: embed_tokens (input) lm_head (output)

```console
python3 train.py \
    --run_name stablehome-1_6b-rev3 \
    --base_model stabilityai/stablelm-2-zephyr-1_6b \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 1e-5 --batch_size 32 \
    --micro_batch_size 2 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 --save_steps 100 --save_total_limit 20
```

```console
accelerate launch --config_file fsdp_config.yaml train.py \
    --run_name stablehome-3b-rev10 \
    --base_model stabilityai/stablelm-zephyr-3b \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --learning_rate 1e-5 --batch_size 64 --epochs 1 \
    --micro_batch_size 2 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 \
    --save_steps 50 --save_total_limit 10 --eval_steps 100 --logging_steps 2
```

```console
python3 train.py \
    --run_name stablehome-3b-rev9-dpo \
    --base_model ./models/stablehome-3b-rev9/ \
    --bf16 \
    --train_dataset data/home_assistant_dpo.jsonl \
    --learning_rate 2e-7 --batch_size 16 --epochs 1 \
    --dpo --beta 0.1 --dpo_loss sigmoid \
    --micro_batch_size 1 --gradient_checkpointing \
    --ctx_size 2048 \
    --save_steps 50 --save_total_limit 10 --eval_steps 100 --logging_steps 2
```

```console
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
```

#### Llama 3 8B Instruct:

```console
python3 train.py \
    --run_name llamahome-8b-rev1 \
    --base_model NousResearch/Meta-Llama-3-8B-Instruct \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 1e-5 --learning_rate_warmup 0.03 --batch_size 64 --epochs 1 \
    --micro_batch_size 2 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 \
    --save_steps 25 --save_total_limit 20 --eval_steps 100 --logging_steps 2 \
    --use_lora --lora_rank 32 --lora_alpha 64 --lora_modules up_proj,down_proj,q_proj,v_proj,o_proj
```

#### Llama 2 7B:

```console
python3 train.py \
    --run_name home-7b-rev2 \
    --base_model TheBloke/Llama-2-7B-GPTQ \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --load_as_gptq --use_lora --gradient_checkpointing \
    --add_pad_token --bf16 --micro_batch_size 4 --learning_rate 2e-5
```

#### Bielik 7B Instruct:

```console
python3 train.py \
    --run_name mistralhome-bielik-rev1 \
    --base_model speakleash/Bielik-7B-Instruct-v0.1 \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 1e-5 --learning_rate_warmup 0.03 --batch_size 64 --epochs 1 \
    --micro_batch_size 4 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 \
    --save_steps 50 --save_total_limit 20 --eval_steps 200 --logging_steps 1 \
    --use_lora --lora_rank 32 --lora_alpha 64 --lora_modules up_proj,down_proj,q_proj,v_proj,o_proj --load_in_4bit
```

#### TinyLlama:

```console
python3 train.py \
    --run_name tinyhome-rev4 \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 2e-5 --batch_size 32 \
    --micro_batch_size 8 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 --save_steps 100 --save_total_limit 10
```

#### Qwen2 0.5B Instruct:

```console
python3 train.py \
    --run_name tinyhome-qwen-rev3 \
    --base_model Qwen/Qwen2-0.5B-Instruct \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 2e-5 --batch_size 64 \
    --micro_batch_size 8 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 --save_steps 1000
```

#### polka 1.1b chat:

```console
python3 train.py \
    --run_name tinyhome-polish-rev1 \
    --base_model eryk-mazus/polka-1.1b-chat \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 2e-5 --batch_size 32 \
    --micro_batch_size 8 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 --save_steps 100 --save_total_limit 10
```

```console
python3 train.py \
    --run_name tinyhome-rev2-dpo \
    --base_model ./models/tinyhome-rev2/ \
    --bf16 \
    --train_dataset data/home_assistant_dpo.jsonl \
    --learning_rate 5e-7 --batch_size 16 --epochs 1 \
    --dpo --beta 0.1 --dpo_loss sigmoid --learning_rate_warmup 0.03 \
    --micro_batch_size 2 --gradient_checkpointing \
    --ctx_size 2048 \
    --save_steps 50 --save_total_limit 10 --eval_steps 100 --logging_steps 2
```

#### Llama 3.2 3B Instruct
```
python3 generate_home_assistant_data.py --train --test --large --sharegpt --language english german french spanish

python3 train.py \
    --run_name Home-Llama-3.2-3B-rev1 \
    --base_model meta-llama/Llama-3.2-3B-Instruct \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 1e-5 --learning_rate_warmup 0.03 --batch_size 64 --epochs 1 \
    --micro_batch_size 2 \
    --ctx_size 2048 \
    --save_steps 200 --save_total_limit 3 --eval_steps 100 --logging_steps 2
```

### Problems

Training a model is not an easy thing. Therefore, we are not able to cover all the problems encountered during training. Here we will try to add known problems and solutions on how to deal with them.

1. Problem with incorrect recognizing prefix_ids and suffix_ids

When you ran the script and in the logs you will see this kind of logs:

```console
...
[[128000, 128006, 9125, 128007, 271, 41, 18223, 7545, ... Lots more numbers ..., 128009, 128006, 78191, 128007, 271]]
warning! example had no assistant response in it!
...
```

The script is printing the warning along with the tokenized version of the example that is being trained on. This is to help determine what the correct prefix_ids and suffix_ids.

The idea is that the training script need to build a mask (array) that is True for all of the tokens that the assistant would respond with, and False for all of the tokens that the user inputted or was included in the system prompt. We don't want to train the model to reproduce those tokens because it is a waste of computation power and can also confuse the model. This is the biggest difference between pre-training an LLM and performing Supervised Fine Tuning. In pre-training you train the model on the entire example (mask is True for all tokens).

The training script here attempts to auto-detect which tokens are for the assistant, but that is not trivial, and sometimes you need to manually provide the tokens that start an assistant response, and the tokens that end an assistant response.

For a model like TinyLlama that uses Zephyr format, the prefix is `<|assistant|>\n` and the suffix is `</s>`. That ends up equating to `[29966, 29989, 465, 22137, 29989, 29958, 13]` and `[2]` as the prefix and suffix tokens respectively (the suffix token is actually just the end of sentence/eos token in this case but is not always true for all chat models)

The other issue is that tokenizers perform differently based on if a token is preceded by white-space or if it is adjacent to the token that came before it. For example, check out https://gpt-tokenizer.dev/ to mess around with the GPT tokenizers. Try tokenizing the word `Computerwiz`. You will see that it returns 2 tokens: `[50411, 146049]` split up with `Computer` and `wiz`. Now if you split the word up with a space as Computer wiz, you would expect there to be 3 tokens now, the same 2 tokens from before separated by the "space" token. Instead you get back 2 tokens `[50411, 121731]`. The first token is the same, but the second token has "consumed" the space we inserted and is now a totally different token. This means that figuring out the exact prefix and suffix IDs can be a bit hard to do without the full prompt assembled and all of the spaces, newlines, and tabs that are part of the full chat template.

There is a script included in this repo that shows this and can assist in determining the correct prefix and suffix tokens for your model: [find_split.py](/find_split.py)

Example to use the script:

```console
python3 find_split.py NousResearch/Meta-Llama-3.1-8B-Instruct
```

On the console you will see this output:

```console
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50.9k/50.9k [00:00<00:00, 516kB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9.09M/9.09M [00:17<00:00, 510kB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 296/296 [00:00<00:00, 1.06MB/s]
Estimated tokens for NousResearch/Meta-Llama-3.1-8B-Instruct
response prefix:
<|start_header_id|>assistant<|end_header_id|>


tokens with no leading whitespace: [128006, 78191, 128007, 271]
tokens with leading whitespace: [220, 128006, 78191, 128007, 271]
tokens with leading newline: [198, 128006, 78191, 128007, 271]
---------------
response suffix:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>


tokens with no leading whitespace: [128009, 128006, 78191, 128007, 271]
tokens with leading whitespace: [220, 128009, 128006, 78191, 128007, 271]
tokens with leading newline: [198, 128009, 128006, 78191, 128007, 271]
---------------
'no whitespace' found the assistant response!
	--prefix-ids 128006,78191,128007,271
	--suffix-ids 128009,128006,78191,128007,271
'leading space' did not find the assistant response
'leading newline' did not find the assistant response
```

After that you should add extra parameters to you script for properly set those params. This is Supervised Fine Tuning params for correctly hide requests for the model. You probably see several examples of tokens which script give you, it is worth looking for the indicated tokens in your log and checking which value will be correct (You should be able to find them). In case above the correct values is:

```
prefix_ids = [128006, 78191, 128007, 271]
suffix_ids = [128009, 128006, 78191, 128007, 271]
```

You can add the provided arguments to your training run:

```
python3 train.py \
    ...
    --prefix_ids 128006,78191,128007,271 \
    --suffix_ids 128009,128006,78191,128007,271
```

#### Known prefix and suffix IDs

tinyllama:
```console
python3 train.py \
    ...
    --prefix_ids 29966,29989,465,22137,29989,29958,13 \
    --suffix_ids 2
```

qwen2:
```console
python3 train.py \
    ...
    --prefix_ids 151644,77091,198 \
    --suffix_ids 151645,198
```

polka-1.1:
```console
python3 train.py \
    ...
    --prefix_ids 43883,20255,13 \
    --suffix_ids 43882,29871,13
```

Llama-3-8B-Instruct:
```console
python3 train.py \
    ...
    --prefix-ids 128006,78191,128007,271 \
    --suffix-ids 128009,128006,78191,128007,271
```

## Worth reading:

* [Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth](https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html)
* [Fine-tune Mistral-7b with Direct Preference Optimization](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html)
* [Quantize Llama models with GGUF and llama.cpp](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html)