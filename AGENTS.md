# Home LLM agent guidance

Home Assistant integration `llama_conversation` plus synthetic data, training, and evaluation tooling. Read [README.md](README.md) and [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) before proposing user-facing changes; contributions require an issue and predominantly AI-generated contributions are not accepted upstream.

## General Guidance

1. Never use special characters in your code, documentation, or dataset files. You should always utilize standard ASCII characters. This is to ensure compatibility across different systems and avoid encoding issues. If you need to use a symbol then you should use the corresponding ASCII representation such as '->' for an arrow or '...' for an ellipsis.
2. Do not unnecessarily wrap lines in code or documentation. It is standard to have larger lines in code and documentation should only use newlines for paragraph boundaries. 
3. When responding after finishing a turn: be concise and do not list out every individual change from that turn.

## Integration layout and conventions

- Work in `custom_components/llama_conversation/`: `__init__.py` owns entry setup, migrations, and `BACKEND_TO_CLS`; `config_flow.py` handles connection/model configuration; `entity.py` defines `LocalLLMClient`; `conversation.py` and `ai_task.py` expose the two platforms; `utils.py` handles parsing and prompt utilities.
- Import config keys/defaults from `const.py`, not string literals. Backend subclasses implement the relevant generation/streaming methods, connection validation, and `get_name()`; register new backends in `BACKEND_TO_CLS`, config flow, and translations. Follow an existing backend as an example.
- A backend connection is a config entry; models/tasks are subentries. Merge options with subentry data for entity settings (`{**entry.options, **subentry.data}`). Subentry data is immutable: update through `hass.config_entries.async_update_subentry()`.
- For config structure changes, update `VERSION`/`MINOR_VERSION` in `config_flow.py`, migrate old entries in `__init__.py` without discarding existing settings, and add coverage in `tests/llama_conversation/test_migrations.py`. Check current values in code rather than copying version numbers from prose.
- Preserve both structured tool calls and legacy `<tool_call>` parsing. Streaming, vision, and token limits depend on the backend; check its capability methods and honor configured limits. Optional ICL examples are loaded from `in_context_examples*.csv` (columns `type`, `request`, `tool`, `response`); guard against missing examples. System prompts can be regenerated each turn, so avoid extra expensive state collection.
- Legacy tool-call errors are returned to the model for correction; keep malformed-call handling and `<think>` stripping intact. Entity aliases create additional prompt entries, so test prompt-size changes with many exposed entities. `config_flow.py` uses `internal_step` for multi-step progress; verify resets when changing flow transitions.
- Keep UI strings consistent across `custom_components/llama_conversation/translations/` and the five supported languages (English, German, French, Spanish, Polish).

## Testing

- Install development dependencies with `pip install -r custom_components/requirements-dev.txt`; run `pytest tests/` or a focused file under `tests/llama_conversation/`.
- Tests use pytest, pytest-asyncio (`asyncio_mode = auto`), and pytest-homeassistant-custom-component. Mock backend generation (`_generate()` / `_generate_stream()`); do not make live LLM requests.
- `tests/conftest.py` installs compatibility shims for newer Home Assistant versions before collection; preserve older-HA compatibility when adjusting dependencies or schema conversion. Prefer existing dependencies over adding unnecessary dev-only packages.

## References

- [Setup](docs/Setup.md), [Backend Configuration](docs/Backend%20Configuration.md), [Model Prompting](docs/Model%20Prompting.md), [AI Tasks](docs/AI%20Tasks.md), [Performance](docs/Performance.md).
- For dataset and training changes, follow the nearest `AGENTS.md` under `data/`, `train/`, or `scripts/`.