# Home Assistant LLM — Project Guidelines

Local LLM integration for Home Assistant that enables AI-powered voice/chat control of smart home devices. Domain: `llama_conversation`, version 0.4.6.

## Architecture

```
custom_components/llama_conversation/
├── __init__.py          # Entry setup, migrations, LLM API registration, backend mapping
├── config_flow.py       # Multi-step config flow (pick_backend → connection → model_select)
├── const.py             # All config keys and defaults — never hardcode config strings
├── conversation.py      # LocalLLMAgent — conversation entity with tool call loop
├── ai_task.py           # LocalLLMTaskEntity — structured data extraction for automations
├── entity.py            # LocalLLMEntity base class, LocalLLMClient backend interface
├── utils.py             # Tool call parsing, prompt utilities
├── backends/
│   ├── llamacpp.py      # Local GGUF inference via llama-cpp-python
│   ├── ollama.py        # Ollama REST API
│   ├── generic_openai.py        # OpenAI-compatible APIs (LM Studio, vLLM, LocalAI)
│   ├── tailored_openai.py       # OpenAI Responses API format
│   └── anthropic.py     # Claude API with vision + prompt caching
└── translations/        # Multi-language UI strings
```

### Key Abstractions

- **`LocalLLMClient`** (entity.py): Abstract base for all backends. Subclasses implement `_generate()` / `_generate_stream()`, `async_validate_connection()`, and optionally `_async_load_model()` / `_async_unload_model()`.
- **Config Entry + Subentry pattern**: One config entry per backend connection, multiple subentries per model/task. Access merged config via `{**entry.options, **subentry.data}`.
- **`BACKEND_TO_CLS`** (__init__.py): Maps backend type strings to client classes. Add new backends here.
- **Tool calling**: Two modes — legacy (regex-extracted `<tool_call>` blocks) and modern (LLM API structured tools). Controlled by `CONF_ENABLE_LEGACY_TOOL_CALLING`.
- **System prompts**: Jinja2 templates rendered per-turn with live entity states. Model-specific overrides in `option_overrides()`.

### Conversation Flow

1. `LocalLLMAgent.async_process()` receives user input
2. System prompt regenerated with current entity states (if `CONF_REFRESH_SYSTEM_PROMPT`)
3. Tool call loop runs up to `max_tool_call_iterations` times:
   - Generate response → extract tool calls → execute → feed results back
4. Message history trimmed by `CONF_REMEMBER_NUM_INTERACTIONS` and `CONF_REMEMBER_CONVERSATION_TIME_MINUTES`

## Build and Test

```bash
# Install dev dependencies
pip install -r custom_components/requirements-dev.txt

# Run tests
pytest tests/

# Run a specific test file
pytest tests/llama_conversation/test_conversation_agent.py -v
```

- **Framework**: pytest + pytest-asyncio + pytest-homeassistant-custom-component
- **Config**: `asyncio_mode = auto` in pytest.ini
- **Pattern**: Mock `_generate()` / `_generate_stream()` — never make real LLM calls in tests

### Local Backend Services

```bash
docker compose up ollama        # Port 11434
docker compose up llamacpp      # Port 8000
docker compose up textgenwebui  # Port 7860
docker compose up localai       # Port 8080
```

## Conventions

### Config Keys

All configuration constants live in `const.py`. Always import from there — never use raw strings for config keys.

### Adding a New Backend

1. Create `backends/{name}.py` with a class extending `LocalLLMClient`
2. Implement required methods: `_generate()`, `async_validate_connection()`, `get_name()`
3. Add to `BACKEND_TO_CLS` mapping in `__init__.py`
4. Add config flow step in `config_flow.py` with schema + validation
5. Add translation strings in `translations/`

### Config Flow Migrations

Migrations run in `async_migrate_entry()` in `__init__.py`. Current version: 3.2. When changing config structure:
- Bump `VERSION` or `MINOR_VERSION` in const.py
- Add migration logic — must be non-destructive and handle partial upgrades
- Add test coverage in `tests/llama_conversation/test_migrations.py`

### Tool Call Parsing

- Legacy format: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` — configurable prefix/suffix
- Thinking blocks (`<think>...</think>`) are stripped from final responses
- Malformed tool calls produce `MalformedToolCallException` — the error is fed back to the model for self-correction
- Streaming parser uses a 5-token sliding window for prefix/suffix detection

### In-Context Learning (ICL)

CSV files (`in_context_examples*.csv`) with columns: `type`, `request`, `tool`, `response`. Placeholders (`<area>`, `<name>`, `<brightness>`, `<color>`) are dynamically substituted with random values from exposed entities. ICL is optional — always guard with `if self.in_context_examples`.

### Multi-Language Support

5 languages: English, German, French, Spanish, Polish. Language selection via `CONF_SELECTED_LANGUAGE` affects prompt templates and UI labels. ICL example files are per-language (e.g., `in_context_examples_de.csv`).

## Pitfalls

- **Subentry data is immutable** — use `async_update_subentry()`, not direct mutation
- **System prompt is regenerated every turn** — expensive with many entities, not memoized
- **Entity aliases create separate prompt entries** — "Kitchen light" + alias "Main light" = two entries
- **Token limits differ per backend** — always respect `CONF_MAX_TOKENS` and `CONF_CONTEXT_LENGTH`
- **Config flow internal state** — `self.internal_step` tracks progress; watch for reset issues
- **llama-cpp-python wheel install** is async in executor — can block on slow networks
- **Vision support is backend-specific** — check `_supports_vision()` before sending image data

## Documentation

Detailed docs live in `docs/` — link rather than duplicate:
- [docs/Setup.md](docs/Setup.md) — Installation and deployment paths
- [docs/Backend Configuration.md](docs/Backend%20Configuration.md) — Per-backend setup details
- [docs/Model Prompting.md](docs/Model%20Prompting.md) — System prompt design and ICL
- [docs/AI Tasks.md](docs/AI%20Tasks.md) — AI Task entity integration
- [docs/Performance.md](docs/Performance.md) — Throughput benchmarks (RPi4/RPi5)

## Training & Data Pipeline

See `.github/instructions/training-pipeline.instructions.md` for the synthetic data generation, model training, and evaluation workflow.
