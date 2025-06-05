#!/bin/bash
set -e

LLAMA_CPP=../llama.cpp
MODEL_NAME=$1

if [[ ! -d "./models/$MODEL_NAME" ]]; then
    echo "Unknown model $MODEL_NAME"
    exit -1
fi

if [ -f "./models/$MODEL_NAME/gguf_overrides.json" ]; then
    OVERRIDES="--metadata ./models/$MODEL_NAME/gguf_overrides.json"
    echo "Using metadata from ./models/$MODEL_NAME/gguf_overrides.json"
else
    OVERRIDES=""
fi

echo "Converting to GGUF..."
if [ ! -f "./models/$MODEL_NAME/$MODEL_NAME.f16.gguf" ]; then
    $LLAMA_CPP/convert_hf_to_gguf.py --outfile ./models/$MODEL_NAME/$MODEL_NAME.f16.gguf --outtype f16 ./models/$MODEL_NAME/ $OVERRIDES
else
    echo "Converted model for already exists. Skipping..."
fi


DESIRED_QUANTS=("Q8_0" "Q5_K_M" "Q4_0" "Q4_1" "Q4_K_M")
for QUANT in "${DESIRED_QUANTS[@]}"
do
    QUANT_LOWER=$(echo "$QUANT" | awk '{print tolower($0)}')
    if [ ! -f "./models/$MODEL_NAME/$MODEL_NAME.$QUANT_LOWER.gguf" ]; then
        $LLAMA_CPP/build/bin/llama-quantize ./models/$MODEL_NAME/$MODEL_NAME.f16.gguf ./models/$MODEL_NAME/$MODEL_NAME.$QUANT_LOWER.gguf $QUANT
    else
        echo "Quantized model for '$QUANT' already exists. Skipping..."
    fi
done