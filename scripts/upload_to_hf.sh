#!/bin/bash

MODEL_NAME=$1

pushd models/
huggingface-cli upload $MODEL_NAME \
    --repo-type model \
    --commit-message "Upload model" \
    --include "*.gguf" "README.md"

# huggingface-cli upload $MODEL_NAME \
#     --repo-type model \
#     --commit-message "Upload safetensors" \
#     --include "*.safetensors" "config.json" "special_tokens_map.json" "tokenizer_config.json" "tokenizer.json" "tokenizer.model" "generation_config.json"