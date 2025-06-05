#!/bin/bash
set -e

MODEL_NAME=$1

huggingface-cli upload $MODEL_NAME models/$MODEL_NAME \
    --repo-type model --commit-message "Upload model" \
    --exclude "runs/" "training_args.bin" "gguf_overrides.json"