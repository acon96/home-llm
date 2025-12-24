
#!/bin/bash
set -e

MODEL_NAME=$1
OUT_TYPE=${2:-"f16"}
MODELS_DIR=${3:-"./models"}
LLAMA_CPP=${4:-"./llama.cpp"}

if [[ ! -d "$MODELS_DIR/$MODEL_NAME" ]]; then
    echo "Unknown model $MODEL_NAME"
    exit -1
fi

if [ -f "$MODELS_DIR/$MODEL_NAME/gguf_overrides.json" ]; then
    OVERRIDES="--metadata $MODELS_DIR/$MODEL_NAME/gguf_overrides.json"
    echo "Using metadata from $MODELS_DIR/$MODEL_NAME/gguf_overrides.json"
else
    OVERRIDES=""
fi

echo "Converting to GGUF..."
if [ ! -f "$MODELS_DIR/$MODEL_NAME/$MODEL_NAME.$OUT_TYPE.gguf" ]; then
    $LLAMA_CPP/convert_hf_to_gguf.py --outfile $MODELS_DIR/$MODEL_NAME/$MODEL_NAME.$OUT_TYPE.gguf --outtype $OUT_TYPE $MODELS_DIR/$MODEL_NAME/ $OVERRIDES
else
    echo "Converted model for already exists. Skipping..."
fi

echo "Generate imatrix for model..."
if [ ! -f "groups_merged.txt" ]; then
    echo "Downloading groups_merged.txt..."
    wget https://huggingface.co/datasets/froggeric/imatrix/resolve/main/groups_merged.txt
fi

if [ ! -f "$MODELS_DIR/$MODEL_NAME/$MODEL_NAME.imatrix.gguf" ]; then
    $LLAMA_CPP/build/bin/llama-imatrix -m $MODELS_DIR/$MODEL_NAME/$MODEL_NAME.$OUT_TYPE.gguf -ngl 999 -c 512 -f groups_merged.txt -o $MODELS_DIR/$MODEL_NAME/$MODEL_NAME.imatrix.gguf
else
    echo "Imatrix model already exists. Skipping..."
fi

DESIRED_QUANTS=("Q8_0" "Q6_K" "Q5_K_M" "Q4_0" "Q4_1" "Q3_K_M" "IQ4_NL" "IQ4_XS")
for QUANT in "${DESIRED_QUANTS[@]}"
do
    echo "Quantizing to $QUANT..."
    QUANT_LOWER=$(echo "$QUANT" | awk '{print tolower($0)}')
    if [ ! -f "$MODELS_DIR/$MODEL_NAME/$MODEL_NAME.$QUANT_LOWER.gguf" ]; then
        $LLAMA_CPP/build/bin/llama-quantize --imatrix $MODELS_DIR/$MODEL_NAME/$MODEL_NAME.imatrix.gguf $MODELS_DIR/$MODEL_NAME/$MODEL_NAME.$OUT_TYPE.gguf $MODELS_DIR/$MODEL_NAME/$MODEL_NAME.$QUANT_LOWER.gguf $QUANT
    else
        echo "Quantized model for '$QUANT' already exists. Skipping..."
    fi
done

echo "All done!"