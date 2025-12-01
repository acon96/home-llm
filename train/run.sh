docker run -d --rm \
    --gpus all \
    -p 8888:8888 \
    -v /mnt/data/training-runs:/workspace/data/axolotl-artifacts \
    -v /mnt/data/training-data:/workspace/data/datasets \
    -v /mnt/data/training-configs:/workspace/configs \
    -v /mnt/data/hf-cache:/workspace/data/huggingface-cache \
    axolotlai/axolotl-cloud:main-py3.11-cu128-2.8.0