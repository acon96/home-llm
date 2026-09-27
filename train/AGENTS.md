# Training and evaluation guidance

- Read [train/README.md](README.md) for Axolotl/Docker setup, supported training approaches, dataset mounts, and config details; the generator is documented in [data/README.md](../data/README.md).
- Keep `configs/` dataset paths, `chat_templates/` tool-call formatting, and generated JSONL structure aligned. Verify actual mounted dataset paths before launching training; example configs expect `/workspace/data/datasets/sample.jsonl`, not necessarily `data/output/`.
- `train.sh` and `training-job.yml` define remote Kubernetes training; `evaluate.py` measures tool selection and argument accuracy. When changing tool schemas or chat formatting, check both training and evaluation paths.