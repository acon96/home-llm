#!/bin/bash
set -e

LLAMA_CPP=../llama.cpp
MODEL_NAME=$1

if [[ ! -d "./models/$MODEL_NAME" ]]; then
    echo "Unknown model $MODEL_NAME"
    exit -1
fi

echo "Converting to GGUF..."
if [ ! -f "./models/$MODEL_NAME/$MODEL_NAME.f16.gguf" ]; then
    $LLAMA_CPP/convert.py --outfile ./models/$MODEL_NAME/$MODEL_NAME.f16.gguf --outtype f16 ./models/$MODEL_NAME/
    # $LLAMA_CPP/convert-hf-to-gguf.py --outfile ./models/$MODEL_NAME/$MODEL_NAME.f16.gguf --outtype f16 ./models/$MODEL_NAME/
else
    echo "Converted model for already exists. Skipping..."
fi


DESIRED_QUANTS=("Q8_0" "Q5_K_M" "Q4_K_M" "Q3_K_M" "Q2_K")
for QUANT in "${DESIRED_QUANTS[@]}"
do
    QUANT_LOWER=$(echo "$QUANT" | awk '{print tolower($0)}')
    if [ ! -f "./models/$MODEL_NAME/$MODEL_NAME.$QUANT_LOWER.gguf" ]; then
        $LLAMA_CPP/build/bin/quantize ./models/$MODEL_NAME/$MODEL_NAME.f16.gguf ./models/$MODEL_NAME/$MODEL_NAME.$QUANT_LOWER.gguf $QUANT
    else
        echo "Quantized model for '$QUANT' already exists. Skipping..."
    fi
done