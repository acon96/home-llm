# Usage: docker exec -it ollama bash -c "/scripts/import_ollama_model.sh /models/Home-3B-v3.q4_k_m.gguf Home-3B-v3:q4_k_m"
LLAMA_CPP=../llama.cpp
GGUF_FILE=$1
MODEL_NAME=$2

echo "FROM $GGUF_FILE" > $GGUF_FILE.Modelfile
ollama create $MODEL_NAME -f $GGUF_FILE.Modelfile
rm -f $GGUF_FILE.Modelfile