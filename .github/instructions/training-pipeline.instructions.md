---
applyTo: "data/**,train/**,scripts/**"
description: "Use when working on synthetic data generation, model training, evaluation, or the data/train/scripts directories."
---

# Training & Data Pipeline

Synthetic data generation → Axolotl fine-tuning → GGUF quantization → deployment via HA integration backends.

## Data Generation

### Quick Start

```bash
pip install -r data/requirements.txt

# Generate a small sample dataset
python3 data/generate_data.py --sample --language english

# Generate full training set
python3 data/generate_data.py --train --large --language english

# Generate test set
python3 data/generate_data.py --test --language english
```

Output goes to `data/output/` as JSONL files (e.g., `home_assistant_train_english.jsonl`).

### How It Works

`data/generate_data.py` assembles training examples from modular CSV "piles" in `data/piles/{language}/`:

| Pile | Purpose |
|------|---------|
| `pile_of_device_names.csv` | Entity IDs and friendly names |
| `pile_of_templated_actions.csv` | Parameterized action templates (`<device_name>`, `<brightness>`, `<temp_f>`, `<duration>`) |
| `pile_of_specific_actions.csv` | Exact device + action pairs |
| `pile_of_responses.csv` | Two-part responses: `response_starting` (in progress) + `response_confirmed` (completed) |
| `pile_of_status_requests.csv` | Queries about device state |
| `pile_of_system_prompts.csv` | Varied personas (assistant, pirate, robot) |
| `pile_of_failed_tool_calls.csv` | Error recovery scenarios |
| `pile_of_refusals.csv` | Model should decline (device unavailable, already in desired state) |

Each generated example includes a system prompt with 50-128 random device entities for context robustness.

### Supported Devices

Defined in `data/devices.py`: light, switch, fan, garage_door, blinds, lock, climate, media_player, vacuum, timer, todo. Each maps to HA tools in `data/tools.py` (20+ tool definitions: `HassTurnOn`, `HassLightSet`, `HassClimateSetTemperature`, etc.).

### Data Format

```json
{
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "You are 'Al'... Devices:\n[ALL STATES]"}]},
    {"role": "user", "content": "lower the kitchen blinds"},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tools": [...]
}
```

Two assistant turns per example: acknowledging action → confirming completion.

### Synthetic Augmentation

`data/synthesize.py` generates new pile rows via an LLM API:

```bash
python3 data/synthesize.py --failed-tool-calls 50 --refusals 50 --actions 100
```

Uses concurrent aiohttp calls. Results go to language-specific pile CSVs.

### Translation

```bash
python3 data/translate_data.py --language german
```

5 supported languages: English, German, French, Spanish, Polish. Tool calling format stays the same across languages.

## Training

### Framework & Requirements

- **Framework**: Axolotl (LoRA/QLoRA/full fine-tuning)
- **GPU**: 24GB VRAM minimum (2×12GB, 1×24GB, or 2×16GB)
- **Docker**: `axolotlai/axolotl-cloud:main-py3.11-cu128-2.8.0`

### Configs

Training configs in `train/configs/`:
- `gemma3-270m.yml` — 4096 seq length, sample packing, bf16, adamw_bnb_8bit
- `functiongemma-270m.yml` — Structured tool definition format variant

Chat templates in `train/chat_templates/`:
- `gemma3_withtools.j2` — Gemma3 format with `<tool_call>` / `<tool_result>` tokens
- `chatml_template.j2`, `zephyr_legacy.j2` — Alternative formats

### Running Training

```bash
# Local with Docker
docker run --gpus all -v $(pwd):/workspace axolotlai/axolotl-cloud:main-py3.11-cu128-2.8.0 \
  axolotl train /workspace/train/configs/gemma3-270m.yml

# Remote via Kubernetes
train/train.sh {MODEL_NAME} {REMOTE_SERVER}
```

`train/train.sh` creates a Kubernetes Job with preprocess init container + training container + TensorBoard sidecar.

## Evaluation

```bash
python3 train/evaluate.py --model /path/to/model --test-data data/output/home_assistant_test.jsonl
```

Measures tool call accuracy (correct tool + arguments vs expected). Supports ICL examples from the component's CSV files. Color matching has tolerance for RGB name mapping differences.

## Model Artifacts

### Naming Convention

- `Home-FunctionGemma-270m`: Newer fine-tuned models (e.g. based on FunctionGemma-270m)
- `Home-{size}-{version}`: Older models (e.g., `Home-3B-v3`, `Home-1B-rev4`)
- `tinyhome-revN`: experiment results
- Quantization suffix: `.q4_k_m` (4-bit), `.q8_0` (8-bit), `.f16` (full precision)

### Quantization & Upload

```bash
scripts/convert_and_quantize.sh    # Convert to GGUF + quantize
scripts/upload_to_hf.sh            # Push to Hugging Face Hub
scripts/import_ollama_model.sh     # Import GGUF into Ollama
```

## Conventions

- **Two-turn responses**: First turn acknowledges action, second confirms completion
- **Random device context**: Each example includes 50-128 random devices so the model learns to find the right one
- **Parameterized templates**: Use `<device_name>`, `<brightness>`, etc. — substituted at generation time
- **Multi-device actions**: "Close blinds AND turn on light" — joined with varied "and" words from pile
- **Multilingual by default**: Generate + translate piles for all 5 languages
- **Tool format is language-invariant**: JSON tool calls are the same regardless of language
