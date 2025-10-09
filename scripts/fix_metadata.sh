#!/bin/bash

LLAMA_CPP=../llama.cpp
MODEL_NAME=$1
OUTPUT_NAME=$2
PRE_TOKENIZER=${3:-stablelm2}
CHAT_TEMPLATE=${4:-zephyr_legacy}

python3 ${LLAMA_CPP}/gguf-py/gguf/scripts/gguf_new_metadata.py $MODEL_NAME $OUTPUT_NAME --pre-tokenizer $PRE_TOKENIZER --chat-template "$(cat $CHAT_TEMPLATE.txt)"