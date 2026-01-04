# Home LLM

Control your Home Assistant smart home with a **completely local** Large Language Model. No cloud services and no subscriptions needed. Just privacy-focused AI running entirely on your own hardware.

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?category=Integration&repository=home-llm&owner=acon96)

## What is Home LLM?

Home LLM is a complete solution for adding AI-powered voice and chat control to Home Assistant. It consists of two parts:

1. **Local LLM Integration** – A Home Assistant custom component that connects local language models to your smart home
2. **Home Models** – Small, efficient AI models fine-tuned specifically for smart home control

### Key Features

- 🏠 **Fully Local** – Everything runs on your hardware. Your data never leaves your control (unless you want to!)
- 🗣️ **Voice & Chat Control** – Use as a conversation agent with voice assistants or chat interfaces
- 🤖 **AI Task Automation** – Generate dynamic content and structured data for automations
- 🌍 **Multi-Language Support** – Built-in support for English, German, French, Spanish, and Polish (better translations are welcome!)
- ⚡ **Runs on Low-Power Devices** – Models work on Raspberry Pi and other modest hardware -- no GPU required!
- 🔌 **Flexible Backends** – Run models locally as part of Home Assistant **or** connect to external model providers

## Quick Start

See the [Setup Guide](./docs/Setup.md) for detailed installation instructions.

**Requirements:** Home Assistant 2025.7.0 or newer

---

## Local LLM Integration

The integration connects language models to Home Assistant, enabling them to understand your requests and control your smart devices.

### Supported Backends

Choose how and where you want to run your models:

| Backend                                                                                             | Best For                                                                      |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| **Llama.cpp** (built-in)                                                                            | Running models directly in Home Assistant                                     |
| **[Ollama](https://ollama.com/)**                                                                   | Easy setup on a separate GPU machine                                          |
| **[Generic OpenAI API](https://platform.openai.com/docs/api-reference/conversations/create)**       | LM Studio, LocalAI, vLLM, and other OpenAI-compatible servers                 |
| **[llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)**              | Heterogeneous (non-uniform) GPU compute setups, including CPU + GPU inference |
| **[OpenAI 'Responses' Style API](https://platform.openai.com/docs/api-reference/responses/create)** | Cloud services supporting the 'responses' style API                           |
| **[Anthropic 'Messages' Style API](https://platform.claude.com/docs/en/api/messages)**              | Cloud services supporting the 'messages' style API                            |
| **[text-generation-webui](https://github.com/oobabooga/text-generation-webui)**                     | Advanced users with existing setups                                           |

> NOTE: When utilizing **external** APIs or model providers, your data will be transmitted over the internet and shared with the respective service providers. Ensure you understand the privacy implications of using these third-party services, since they will be able to see the status of all exposed entities in your Home Assistant instance, which can potentially include your current location.

### Supported Device Types

The integration can control: **lights, switches, fans, covers, locks, climate, media players, vacuums, buttons, timers, todo lists, and scripts**

### Using the Integration

**As a Conversation Agent:**
- Chat with your assistant through the Home Assistant UI
- Connect to voice pipelines with Speech-to-Text and Text-to-Speech
- Supports voice streaming for faster responses

**As an AI Task Handler:**
- Create automations that use AI to process data and generate structured responses
- Perfect for dynamic content generation, data extraction, and intelligent decision making
- See [AI Tasks documentation](./docs/AI%20Tasks.md) for examples

---

## Home LLM Models

The "Home" models are small language models (under 5B parameters) fine-tuned specifically for smart home control. They understand natural language commands and translate them into Home Assistant service calls.

### Latest Models

| Model Family  | Size | Link                                                                                    |
|---------------|------|-----------------------------------------------------------------------------------------|
| **Llama 3.2** | 3B   | [acon96/Home-Llama-3.2-3B](https://huggingface.co/acon96/Home-Llama-3.2-3B)             |
| **Gemma**     | 270M | [acon96/Home-FunctionGemma-270m](https://huggingface.co/acon96/Home-FunctionGemma-270m) |

<details>
<summary>Previous Model Versions</summary>

**Stable Models:**
- 3B v3 (StableLM-Zephyr-3B): [acon96/Home-3B-v3-GGUF](https://huggingface.co/acon96/Home-3B-v3-GGUF)
- 1B v3 (TinyLlama-1.1B): [acon96/Home-1B-v3-GGUF](https://huggingface.co/acon96/Home-1B-v3-GGUF)
- 3B v2 (Phi-2): [acon96/Home-3B-v2-GGUF](https://huggingface.co/acon96/Home-3B-v2-GGUF)
- 1B v2 (Phi-1.5): [acon96/Home-1B-v2-GGUF](https://huggingface.co/acon96/Home-1B-v2-GGUF)
- 1B v1 (Phi-1.5): [acon96/Home-1B-v1-GGUF](https://huggingface.co/acon96/Home-1B-v1-GGUF)

**Multilingual Experiments:**
- German, French, & Spanish (3B): [acon96/stablehome-multilingual-experimental](https://huggingface.co/acon96/stablehome-multilingual-experimental)
- Polish (1B): [acon96/tinyhome-polish-experimental](https://huggingface.co/acon96/tinyhome-polish-experimental)

> **Note:** Models v1 (3B) and earlier are only compatible with integration version 0.2.17 and older.

</details>

### Using Other Models

Don't have dedicated hardware? You can use any instruction-tuned model with **in-context learning (ICL)**. The integration provides examples that teach general-purpose models (like Qwen3, Llama 3, Mistral) how to control your smart home. See the [Setup Guide](./docs/Setup.md) for configuration details.

### Training Your Own

The fine-tuning dataset and training scripts are included in this repository:
- **Dataset:** [Home-Assistant-Requests-V2](https://huggingface.co/datasets/acon96/Home-Assistant-Requests-V2) on HuggingFace
- **Source:** [data/](./data) directory
- **Training:** See [train/README.md](./train/README.md)

---

## Documentation

- [Setup Guide](./docs/Setup.md) – Installation and configuration
- [Backend Configuration](./docs/Backend%20Configuration.md) – Detailed backend options
- [Model Prompting](./docs/Model%20Prompting.md) – Customize system prompts
- [AI Tasks](./docs/AI%20Tasks.md) – Using AI in automations

---

## Version History

| Version    | Highlights                                                                                 |
|------------|--------------------------------------------------------------------------------------------|
| **v0.4.6** | Anthropic API support, on-disk caching for Llama.cpp, new tool calling dataset             |
| **v0.4.5** | AI Task entities, multiple LLM APIs at once, official Ollama package                       |
| **v0.4**   | Tool calling rewrite, voice streaming, agentic tool use loop, multiple configs per backend |
| **v0.3**   | Home Assistant LLM API support, improved prompting, HuggingFace GGUF auto-detection        |

<details>
<summary>Full Version History</summary>

| Version | Description                                                                                                                                             |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| v0.4.6  | New dataset supporting proper tool calling, Add Anthropic "messages" style API support, Add on-disk caching for Llama.cpp backend                       |
| v0.4.5  | Add support for AI Task entities, Replace custom Ollama API implementation with the official `ollama-python` package, Support multiple LLM APIs at once |
| v0.4.4  | Fix issue with OpenAI backends appending `/v1` to all URLs                                                                                              |
| v0.4.3  | Fix model config creation during setup                                                                                                                  |
| v0.4.2  | Fix default model settings, numeric config fields, finish_reason handling                                                                               |
| v0.4.1  | Fix Llama.cpp models downloaded from HuggingFace                                                                                                        |
| v0.4    | Rewrite for tool calling models, agentic tool use loop, voice streaming, multiple config sub-entries                                                    |
| v0.3.11 | Bug-fixes and llama.cpp version update                                                                                                                  |
| v0.3.10 | OpenAI "Responses" API support, HA 2025.7.0 compatibility                                                                                               |
| v0.3.9  | Fix conversation history                                                                                                                                |
| v0.3.8  | Thinking model support, HA 2025.4 compatibility                                                                                                         |
| v0.3.7  | German ICL examples, multi-turn fixes                                                                                                                   |
| v0.3.6  | Small llama.cpp backend fixes                                                                                                                           |
| v0.3.5  | Polish ICL examples                                                                                                                                     |
| v0.3.4  | Full Polish translation, improved language support                                                                                                      |
| v0.3.3  | Generic OpenAI improvements, area handling                                                                                                              |
| v0.3.2  | Script entity fixes                                                                                                                                     |
| v0.3.1  | Basic area support in prompting                                                                                                                         |
| v0.3    | Home Assistant LLM API support, improved prompting                                                                                                      |
| v0.2.x  | Ollama support, in-context learning, flash attention, prompt caching                                                                                    |
| v0.1    | Initial Release                                                                                                                                         |

</details>

