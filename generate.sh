#!/bin/bash

LLAMA_CPP=../llama.cpp
MODEL_NAME=$1
PROMPT_SRC=${2:-./data/test_prompts/ha_demo.txt}
QUANT_TYPE=${3:-f16}

if [[ ! -d "./models/$MODEL_NAME" ]]; then
    echo "Unknown model $MODEL_NAME"
    exit -1
fi

dos2unix $PROMPT_SRC
PROMPT=$(cat $PROMPT_SRC)
$LLAMA_CPP/build/bin/main --model "./models/$MODEL_NAME/$MODEL_NAME.$QUANT_TYPE.gguf" --temp 0.1 --ctx-size 2048 --prompt "$PROMPT" --grammar-file ./custom_components/llama_conversation/output.gbnf
