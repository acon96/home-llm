# Synthetic dataset guidance

- Read [data/README.md](README.md) for generation flags, CSV pile formats, personas, and synthesis workflows. Install dependencies from `data/requirements.txt` before running generators.
- `generate_data.py` assembles multilingual JSONL from `piles/{language}/`; keep entity names, actions, status requests, prompts, refusals, and response piles compatible. Device types and tool definitions live in `devices.py` and `tools.py`.
- Preserve two-turn examples (acknowledge the action, then confirm its result), randomized device context, and placeholder substitution such as `<device_name>` and `<brightness>`. Tool-call JSON stays language-invariant across English, German, French, Spanish, and Polish.
- `synthesize.py` appends generated pile rows; `translate_data.py` handles translated piles. Check generated samples and relevant languages before changing generation formats consumed by [train configs](../train/README.md).