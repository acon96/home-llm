# Training new model

## Initial requirements

Before you start training you model you should prepare data. One off the way to prepare data is run script `data/generate_home_assistant_data.py`. Currently this script support english, french, german, polish and spanish language. This is not the only way and you can prepare your own data. More details can be found in the file [HOW TO PREPARE DATA](../data/README.md).


## Training

Before you will prepare your local environment and install all necessary library. Remember that you need a graphics card to effectively train the new model.

Start by installing system dependencies:
`sudo apt-get install python3-dev`

Then create a Python virtual environment and install all necessary library:
```
python3 -m venv .train_data
source ./.train_data/bin/activate
pip3 install datasets==2.20.0 dataclasses==0.6 transformers==4.43.3 torch==2.4.0 accelerate==0.33.0 tensorboard==2.17.0 peft==0.12.0 bitsandbytes==0.43.3 trl==0.9.6
```

### Prepare your model

Select model which you need to train your model: eg `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Remember that the choice of model depends on the VRAM your graphic card. If the model is larger, it is worth using the LoRA configuration which will allow you to train your model.

Run train.py script to start Fine Tuning with selected model and prepared data. It is worth experimenting with several parameters to choose the best model. Few example with some params you can find below:


The 3B model was trained as a full fine-tuning on 2x RTX 4090 (48GB). Training time took approximately 28 hours. It was trained on the `--large` dataset variant.

```console
accelerate launch --config_file fsdp_config.yaml train.py \
    --run_name home-3b \
    --base_model stabilityai/stablelm-zephyr-3b \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --learning_rate 1e-5 --batch_size 64 --epochs 1 \
    --micro_batch_size 2 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 \
    --save_steps 50 --save_total_limit 10 --eval_steps 100 --logging_steps 2
```

The 1B model was trained as a full fine-tuning on an RTX 3090 (24GB). Training took approximately 2 hours. It was trained on the `--medium` dataset variant.

```console
python3 train.py \
    --run_name home-1b \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --bf16 \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --learning_rate 2e-5 --batch_size 32 \
    --micro_batch_size 8 --gradient_checkpointing --group_by_length \
    --ctx_size 2048 --save_steps 100 --save_total_limit 10
```

Phi Modules: 
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

StableLM Modules: 
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

Llama 3 8B Instruct:

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

Llama 2 7B:

```console
python3 train.py \
    --run_name home-7b-rev2 \
    --base_model TheBloke/Llama-2-7B-GPTQ \
    --train_dataset data/home_assistant_train.jsonl \
    --test_dataset data/home_assistant_test.jsonl \
    --load_as_gptq --use_lora --gradient_checkpointing \
    --add_pad_token --bf16 --micro_batch_size 4 --learning_rate 2e-5
```

Bielik 7B Instruct:

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

TinyLlama:

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

Qwen2 0.5B Instruct:

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

polka 1.1b chat:

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

### Problems

Training a model is not an easy thing. Therefore, we are not able to cover all the problems encountered during training. Here we will try to add known problems and solutions on how to deal with them.

1. Problem with incorrect recognising prefix_ids and suffix_ids

When you ran the script and in the logs you will see this kind of logs:

```console
...
[[128000, 128006, 9125, 128007, 271, 41, 18223, 7545, 15179, 2149, 9520, 95811, 49221, 439, 599, 306, 336, 15592, 11, 42942, 97134, 34967, 4433, 77781, 36188, 289, 118447, 13, 24100, 40637, 1662, 281, 21446, 6077, 82, 3059, 71748, 19700, 1167, 40811, 11044, 1167, 9680, 3178, 66, 37937, 28445, 72086, 89, 4415, 281, 21446, 6077, 82, 3059, 95771, 19700, 11, 33054, 55081, 96351, 16999, 45500, 89594, 1167, 7661, 45553, 6179, 34478, 627, 92040, 940, 3458, 10087, 89, 2259, 602, 828, 311, 220, 975, 25, 1272, 289, 30286, 299, 3315, 11, 220, 24, 112611, 2649, 220, 2366, 17, 198, 3642, 4697, 51640, 25, 3504, 4757, 39187, 1535, 3504, 5949, 39187, 1535, 3504, 13530, 39187, 1535, 3504, 24582, 1535, 10182, 995, 766, 276, 7450, 968, 276, 7450, 705, 10182, 995, 1552, 75026, 3283, 75026, 705, 10182, 995, 1552, 54803, 7450, 1535, 10182, 995, 623, 9915, 7450, 1535, 10182, 995, 54625, 7, 35658, 705, 10182, 24582, 1535, 10182, 46734, 13965, 1535, 10182, 46734, 4570, 1535, 8571, 2337, 20542, 17374, 1535, 8571, 1896, 20542, 17374, 1535, 8571, 24582, 1535, 8571, 46734, 13965, 1535, 8571, 46734, 4570, 1535, 3504, 4757, 39187, 1535, 3504, 5949, 39187, 1535, 3504, 13530, 39187, 1535, 3504, 24582, 1535, 3177, 24582, 1535, 3177, 46734, 13965, 1535, 3177, 46734, 4570, 67423, 6855, 8568, 37730, 705, 5409, 21679, 1535, 5409, 48281, 1535, 3772, 15892, 26142, 11507, 29645, 1535, 3772, 15892, 26142, 61089, 1535, 3772, 15892, 26142, 22932, 1535, 3772, 15892, 26142, 22932, 61089, 1535, 3772, 15892, 26142, 53652, 29645, 1535, 3772, 15892, 26142, 19577, 1535, 3772, 15892, 24582, 1535, 3772, 15892, 46734, 13965, 1535, 3772, 15892, 46734, 4570, 1535, 3772, 15892, 41584, 14325, 1535, 3772, 15892, 41584, 722, 1088, 1535, 3772, 15892, 41584, 8401, 1535, 3480, 24582, 1535, 3480, 46734, 13965, 1535, 3480, 46734, 4570, 1535, 9198, 23049, 1535, 9198, 41155, 1535, 9198, 5069, 49248, 705, 12067, 1388, 5752, 5508, 340, 55615, 77781, 36188, 512, 7551, 15892, 2180, 321, 3153, 62, 3042, 36000, 47232, 364, 22378, 86, 450, 269, 74603, 1167, 7661, 7545, 86, 3978, 2963, 27960, 13414, 6, 284, 1022, 198, 4238, 962, 337, 3919, 86437, 86, 60477, 9509, 22389, 8390, 364, 33, 84995, 818, 297, 7545, 86, 3978, 2963, 648, 7500, 4653, 4415, 656, 273, 6, 284, 389, 26, 485, 7992, 320, 2614, 11, 220, 1187, 11, 220, 10290, 1237, 2287, 14062, 17825, 98116, 5407, 89, 1100, 316, 11044, 19699, 62, 2868, 38977, 258, 47011, 364, 44717, 21951, 1100, 316, 11044, 19699, 297, 577, 24409, 58197, 613, 6, 284, 220, 1032, 198, 17790, 14778, 7545, 86, 3978, 2963, 648, 62, 540, 4653, 6551, 318, 78380, 364, 3617, 3059, 45500, 14088, 22212, 83008, 6629, 4697, 289, 105125, 6077, 261, 3893, 6, 284, 1022, 198, 4238, 1276, 64, 821, 1286, 722, 1149, 525, 22389, 8390, 364, 129, 248, 86, 10574, 23762, 289, 503, 1091, 6729, 18472, 4697, 30086, 4657, 4415, 656, 273, 6, 284, 389, 68336, 48162, 320, 13341, 11, 220, 1591, 11, 220, 13302, 1237, 3391, 14062, 31278, 5314, 1412, 21557, 62, 16, 364, 54, 306, 4010, 859, 735, 1412, 7907, 6, 284, 389, 198, 7551, 15892, 10583, 3059, 45500, 14088, 22212, 1107, 20942, 364, 3617, 3059, 45500, 14088, 22212, 23643, 6, 284, 28747, 26, 12821, 28, 15, 13, 2304, 198, 1039, 1444, 1590, 11949, 266, 35045, 22645, 364, 57, 373, 74, 656, 10244, 2642, 6780, 25981, 11949, 266, 61482, 6, 284, 36718, 198, 4238, 516, 44711, 646, 1739, 275, 364, 43, 23465, 924, 6410, 13780, 289, 40828, 648, 6, 284, 389, 26, 13223, 11912, 320, 3264, 11, 220, 7743, 11, 220, 8899, 1237, 2137, 14062, 4238, 1258, 78, 4697, 1704, 1590, 11949, 266, 37975, 364, 43, 23465, 25981, 11949, 266, 13780, 289, 342, 5169, 6077, 84, 6, 284, 389, 26, 12481, 85, 32012, 320, 8899, 11, 220, 1187, 11, 220, 12326, 1237, 4044, 14062, 31278, 516, 1739, 275, 62, 17, 364, 54, 306, 4010, 5382, 924, 6410, 24665, 6, 284, 389, 198, 7551, 15892, 13269, 20762, 62, 20, 364, 63365, 5119, 20, 6, 284, 1022, 198, 17790, 14778, 7545, 86, 3978, 2963, 648, 49410, 1394, 1910, 84, 364, 3617, 3059, 45500, 14088, 22212, 83008, 6629, 4697, 4983, 1394, 1910, 84, 6, 284, 389, 198, 17825, 87555, 1677, 13255, 623, 13327, 5267, 5010, 34478, 62, 540, 4653, 364, 21119, 9265, 81946, 5267, 5010, 34478, 7500, 4653, 6, 284, 220, 24, 198, 3773, 1326, 4697, 42413, 64, 646, 1100, 532, 21557, 364, 39972, 1169, 88, 289, 66789, 12543, 36900, 274, 1100, 532, 7907, 6, 284, 8036, 198, 31278, 870, 2596, 9336, 530, 324, 754, 364, 51, 9225, 363, 306, 4010, 859, 40715, 9336, 6, 284, 1022, 198, 4238, 5314, 1412, 21557, 14725, 669, 648, 79, 30678, 364, 42, 1412, 21557, 4821, 356, 648, 79, 81793, 93127, 86, 10574, 23762, 6, 284, 1022, 54847, 324, 59770, 320, 2946, 11, 220, 10697, 11, 220, 11727, 1237, 2983, 14062, 31278, 32663, 19700, 62, 17, 364, 42987, 23762, 7545, 22212, 93562, 6780, 6, 284, 389, 198, 7551, 15892, 5314, 1412, 21557, 50840, 364, 46, 8207, 45804, 42930, 9038, 689, 597, 1412, 2734, 13546, 6, 284, 389, 26, 44584, 3509, 22037, 11348, 26, 12821, 28, 15, 13, 22, 198, 4238, 13, 4720, 331, 364, 129, 248, 86, 10574, 23762, 4415, 7661, 35989, 26738, 6, 284, 389, 26, 27178, 12481, 320, 1419, 11, 220, 1627, 11, 220, 10828, 340, 94874, 739, 2172, 89925, 364, 1090, 39567, 306, 3919, 2024, 55825, 350, 2172, 6, 284, 7155, 26, 13556, 12310, 26, 1313, 34, 26, 843, 14062, 19834, 10583, 1727, 64, 62, 16, 364, 129, 248, 839, 39518, 118955, 10036, 550, 9345, 6, 284, 4642, 198, 4238, 739, 88, 4697, 669, 94523, 918, 333, 87, 364, 51, 4010, 818, 83008, 10574, 23762, 24890, 24665, 6, 284, 1022, 26, 23449, 98335, 24409, 320, 8953, 11, 220, 10674, 11, 220, 1627, 1237, 3264, 14062, 7551, 15892, 31339, 73, 58021, 12162, 599, 306, 1928, 4697, 111627, 364, 2170, 599, 306, 66789, 437, 37975, 31948, 584, 73, 18980, 84, 6, 284, 28747, 26, 12821, 28, 15, 13, 17, 198, 1039, 54762, 8733, 364, 57, 373, 74, 1377, 89, 17043, 656, 829, 359, 88, 6, 284, 16447, 198, 3773, 4025, 89127, 37975, 364, 57, 89127, 24665, 1377, 89, 17043, 342, 5169, 6077, 24665, 6, 284, 8036, 198, 3773, 739, 4010, 818, 20404, 2201, 364, 39972, 1169, 88, 4415, 259, 4010, 818, 5509, 2201, 6, 284, 1825, 198, 4238, 10583, 89, 21151, 1311, 56042, 52789, 34033, 364, 3617, 16284, 11044, 83008, 10574, 23762, 312, 56042, 96653, 818, 6, 284, 389, 26, 485, 7992, 320, 2075, 11, 220, 15, 11, 220, 5894, 1237, 5547, 128009, 128006, 882, 128007, 271, 48872, 50272, 11044, 83008, 10574, 23762, 312, 56042, 96653, 818, 13599, 42910, 78512, 3273, 12088, 336, 36900, 37050, 111806, 36900, 78161, 648, 30, 128009, 128006, 78191, 128007, 271, 652, 16284, 11044, 83008, 10574, 23762, 312, 56042, 96653, 818, 13599, 42910, 78512, 3273, 78161, 648, 220, 5547, 13, 128009, 128006, 78191, 128007, 271], [128000, 128006, 9125, 128007, 271, 41, 18223, 7545, 15179, 14804, 78, 9520, 95811, 49221, 12585, 336, 15592, 11, 42942, 97134, 34967, 4433, 77781, 36188, 289, 118447, 13, 24100, 40637, 1662, 281, 21446, 6077, 82, 3059, 71748, 19700, 1167, 40811, 11044, 1167, 9680, 3178, 66, 37937, 28445, 72086, 89, 4415, 281, 21446, 6077, 82, 3059, 95771, 19700, 11, 33054, 55081, 96351, 16999, 45500, 89594, 1167, 7661, 45553, 6179, 34478, 13, 9220, 5697, 72086, 40611, 7019, 258, 3458, 60605, 61852, 39234, 3458, 602, 1167, 8805, 3059, 1167, 42930, 53444, 7886, 12951, 11018, 15179, 3513, 752, 12, 1255, 454, 113068, 92040, 940, 3458, 10087, 89, 2259, 602, 828, 311, 220, 1313, 25, 1544, 289, 11752, 6217, 11, 220, 1032, 274, 1291, 20553, 689, 220, 2366, 17, 198, 3642, 4697, 51640, 25, 3504, 4757, 39187, 1535, 3504, 5949, 39187, 1535, 3504, 13530, 39187, 1535, 3504, 24582, 1535, 10182, 995, 766, 276, 7450, 968, 276, 7450, 705, 10182, 995, 1552, 75026, 3283, 75026, 705, 10182, 995, 1552, 54803, 7450, 1535, 10182, 995, 623, 9915, 7450, 1535, 10182, 995, 54625, 7, 35658, 705, 10182, 24582, 1535, 10182, 46734, 13965, 1535, 10182, 46734, 4570, 1535, 8571, 2337, 20542, 17374, 1535, 8571, 1896, 20542, 17374, 1535, 8571, 24582, 1535, 8571, 46734, 13965, 1535, 8571, 46734, 4570, 1535, 3504, 4757, 39187, 1535, 3504, 5949, 39187, 1535, 3504, 13530, 39187, 1535, 3504, 24582, 1535, 3177, 24582, 1535, 3177, 46734, 13965, 1535, 3177, 46734, 4570, 67423, 6855, 8568, 37730, 705, 5409, 21679, 1535, 5409, 48281, 1535, 3772, 15892, 26142, 11507, 29645, 1535, 3772, 15892, 26142, 61089, 1535, 3772, 15892, 26142, 22932, 1535, 3772, 15892, 26142, 22932, 61089, 1535, 3772, 15892, 26142, 53652, 29645, 1535, 3772, 15892, 26142, 19577, 1535, 3772, 15892, 24582, 1535, 3772, 15892, 46734, 13965, 1535, 3772, 15892, 46734, 4570, 1535, 3772, 15892, 41584, 14325, 1535, 3772, 15892, 41584, 722, 1088, 1535, 3772, 15892, 41584, 8401, 1535, 3480, 24582, 1535, 3480, 46734, 13965, 1535, 3480, 46734, 4570, 1535, 12067, 1388, 5752, 5508, 340, 55615, 77781, 36188, 512, 7551, 15892, 83605, 437, 64014, 364, 1090, 39567, 306, 3919, 5222, 2308, 12103, 437, 6, 284, 1022, 198, 1039, 558, 347, 86, 60477, 9509, 35045, 40952, 364, 57, 373, 74, 656, 10244, 1289, 7500, 24409, 61482, 6, 284, 36718, 198, 17825, 98116, 37422, 805, 58554, 45553, 766, 9672, 12543, 364, 44717, 98502, 73, 55174, 19699, 13672, 14088, 66105, 4632, 78380, 6, 284, 220, 24, 198, 31278, 32663, 19700, 62, 16, 364, 42987, 23762, 7545, 22212, 93562, 6780, 6, 284, 389, 198, 17790, 14778, 7545, 86, 3978, 2963, 648, 67708, 96377, 364, 81450, 55489, 297, 7545, 86, 3978, 2963, 27960, 12460, 96377, 6, 284, 389, 198, 4238, 558, 5408, 67914, 2868, 722, 33367, 364, 129, 248, 86, 10574, 23762, 289, 6160, 324, 3059, 4415, 1370, 466, 3059, 6, 284, 1022, 198, 4238, 1276, 64, 821, 1286, 4803, 3394, 722, 33367, 364, 129, 248, 86, 10574, 23762, 597, 2259, 4415, 656, 273, 6, 284, 1022, 26, 1544, 14062, 17790, 34430, 90179, 3039, 918, 2866, 364, 38, 4697, 42413, 88, 289, 45500, 14088, 22212, 83008, 6629, 4697, 289, 274, 1100, 532, 7907, 6, 284, 1022, 198, 4238, 94281, 36934, 86437, 646, 103844, 10830, 364, 46, 7545, 86, 3978, 2963, 648, 10244, 2642, 88, 7661, 5817, 347, 10830, 6, 284, 389, 26, 13223, 11912, 320, 6550, 11, 220, 8259, 11, 220, 6849, 1237, 508, 14062, 94874, 870, 2596, 9336, 94437, 2265, 530, 21, 364, 39, 2596, 9336, 16333, 2265, 350, 21, 1322, 6, 284, 1022, 26, 13556, 5234, 26, 972, 34, 26, 1272, 14062, 1039, 38038, 89, 17043, 47022, 73, 18980, 24665, 364, 57, 373, 74, 1377, 89, 17043, 50272, 77, 718, 6, 284, 16447, 198, 3773, 13, 3043, 64, 89925, 364, 1090, 39567, 306, 818, 938, 1169, 88, 1167, 99734, 72, 6, 284, 8036, 198, 4238, 1276, 64, 1928, 1832, 55174, 890, 69713, 354, 53413, 6551, 343, 33131, 364, 129, 248, 86, 10574, 23762, 44615, 354, 79853, 4415, 9115, 5267, 376, 3059, 6, 284, 389, 26, 11490, 398, 6798, 320, 12338, 11, 220, 6393, 11, 220, 8929, 1237, 2618, 14062, 4238, 1190, 51588, 21557, 364, 129, 248, 86, 10574, 23762, 289, 503, 51588, 7907, 6, 284, 389, 198, 3773, 1444, 1832, 64123, 1928, 8255, 37975, 364, 38, 5169, 6077, 289, 1832, 41796, 12543, 19665, 47011, 6, 284, 1825, 198, 4238, 10583, 89, 21151, 47232, 26053, 34033, 364, 3617, 16284, 11044, 297, 7545, 86, 3978, 2963, 648, 281, 28342, 8783, 8122, 86, 450, 88, 73, 53199, 6, 284, 389, 26, 1399, 14062, 31278, 739, 88, 2312, 88, 1928, 276, 1247, 364, 54, 306, 4010, 859, 4415, 13892, 2312, 1631, 342, 1201, 84, 6, 284, 1022, 198, 4238, 25244, 5267, 33542, 5407, 89, 6910, 41796, 1286, 6551, 31498, 364, 129, 248, 86, 10574, 23762, 289, 21951, 6910, 41796, 44906, 4415, 105747, 55174, 6, 284, 1022, 26, 717, 14062, 3773, 13, 7545, 86, 3978, 7792, 364, 129, 119, 278, 5308, 3841, 83008, 3978, 7792, 24665, 6, 284, 8036, 198, 4238, 10583, 89, 21151, 80704, 722, 33367, 364, 129, 248, 86, 10574, 23762, 289, 6160, 324, 3059, 6, 284, 1022, 54847, 324, 59770, 320, 2075, 11, 220, 13860, 11, 220, 11247, 340, 1039, 516, 1100, 532, 21557, 62, 18, 364, 57, 373, 74, 289, 490, 89, 762, 57647, 274, 1100, 532, 7907, 6, 284, 36718, 198, 4238, 739, 88, 4697, 1704, 1003, 3039, 918, 333, 87, 364, 51, 4010, 818, 83008, 10574, 23762, 311, 109149, 88, 6, 284, 389, 26, 51755, 13553, 320, 1591, 11, 220, 10350, 11, 220, 1958, 340, 4238, 5314, 1412, 21557, 669, 1885, 96934, 269, 1215, 669, 1786, 364, 42, 1412, 18314, 597, 5985, 66, 1609, 597, 675, 37975, 435, 1662, 818, 83008, 10574, 23762, 6, 284, 1022, 26, 6083, 14062, 31278, 516, 44711, 623, 28342, 8783, 33691, 10952, 364, 44503, 3919, 4024, 4010, 859, 656, 40828, 84, 6, 284, 1022, 198, 7551, 15892, 23451, 47232, 1284, 875, 364, 17111, 6007, 47561, 6, 284, 1022, 198, 4238, 558, 347, 86, 60477, 9509, 364, 129, 248, 86, 10574, 23762, 7661, 86, 1832, 48372, 24665, 6, 284, 1022, 26, 5833, 4, 128009, 128006, 882, 128007, 271, 40173, 11044, 7545, 6940, 2693, 5267, 656, 220, 1419, 3009, 7907, 128009, 128006, 78191, 128007, 271, 79, 602, 281, 1174, 256, 281, 297, 294, 342, 436, 1167, 384, 503, 256, 259, 384, 296, 281, 384, 436, 264, 259, 577, 436, 220, 5267, 1174, 256, 577, 274, 259, 264, 289, 602, 264, 503, 220, 5985, 272, 256, 259, 384, 296, 281, 384, 436, 264, 259, 577, 436, 220, 5267, 256, 308, 264, 256, 366, 259, 384, 296, 281, 721, 272, 871, 256, 274, 259, 297, 281, 308, 602, 1174, 256, 293, 577, 296, 16853, 74694, 5227, 78191, 198, 5018, 8095, 794, 330, 94874, 995, 54625, 498, 330, 5775, 9385, 794, 330, 94874, 870, 2596, 9336, 94437, 2265, 530, 21, 498, 330, 35658, 794, 220, 1419, 534, 74694, 128009, 128006, 78191, 128007, 271]]
warning! example had no assistant response in it!
...
```

The script is printing the warning along with the tokenized version of the example that is being trained on. This is just a helpful thing I did to help determine what the correct prefix_ids and suffix_ids.

The idea is that the training script need to build a mask (array) that is True for all of the tokens that the assistant would respond with, and False for all of the tokens that the user inputted or was included in the system prompt. We don't want to train the model to reproduce those tokens because it is a waste of computation power and can also confuse the model. This is the biggest difference between pre-training an LLM and performing Supervised Fine Tuning. In pre-training you train the model on the entire example (mask is True for all tokens).

The training script here attempts to auto-detect which tokens are for the assistant, but that is not trivial, and sometimes you need to manually provide the tokens that start an assistant response, and the tokens that end an assistant response.

For a model like TinyLlama that uses Zephyr format, the prefix is <|assistant|>\n and the suffix is </s>. That ends up equating to [29966, 29989, 465, 22137, 29989, 29958, 13] and [2] as the prefix and suffix tokens respectively (the suffix token is actually just the end of sentence/eos token in this case but is not always true for all chat models)

The other issue is that tokenizers perform differently based on if a token is preceded by white-space or if it is adjacent to the token that came before it. For example, check out https://gpt-tokenizer.dev/ to mess around with the GPT tokenizers. Try tokenizing the word Computerwiz. You will see that it returns 2 tokens: [50411, 146049] split up with Computer and wiz. Now if you split the word up with a space as Computer wiz, you would expect there to be 3 tokens now, the same 2 tokens from before separated by the "space" token. Instead you get back 2 tokens [50411, 121731]. The first token is the same, but the second token has "consumed" the space we inserted and is now a totally different token. This means that figuring out the exact prefix and suffix IDs can be a bit hard to do without the full prompt assembled and all of the spaces, newlines, and tabs that are part of the full chat template.

I made a script to show this and potentially assist in determining the correct prefix and suffix tokens for your model:
https://github.com/acon96/home-llm/blob/develop/find_split.py

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
```

after that you shuld add extra parameters to you script for properly set those params. This is Supervised Fine Tuning params for correctly hide requests for the model. Few examples you can find below:

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
