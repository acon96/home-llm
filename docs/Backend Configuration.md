# Backend Configuration

There are multiple backends to choose for running the model that the Home Assistant integration uses. Here is a description of all the options for each backend

# Common Options
These options are available for all backends and control model inference behavior, conversation memory, and integration-specific settings.

## Tool Calling Modes

This integration supports two distinct tool calling modes. Choosing the wrong one will cause problems:

- **Agentic mode** (default): The model runs in a multi-turn loop — it can call tools, receive their results, and then decide to call more tools or produce a final response. Use this for any modern instruction-tuned model. Set **Max Tool Call Iterations** to a value greater than 0 (e.g. 3).
- **Legacy/1-shot mode**: The model generates a response *and* any tool calls in a single turn. Results are not fed back for a follow-up turn. Required for older Home-LLM models (v1–v3) that do not know how to produce a final summary response after tool execution. Enable **Legacy Tool Calling** and set **Max Tool Call Iterations** to 0.

> **Warning:** These two settings are mutually exclusive. Enabling Legacy Tool Calling with Max Tool Call Iterations > 0 will be rejected when saving — the model will enter an infinite loop because it cannot produce the agentic summary turn.

## Options Table

| Option Name                                    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Suggested Value         |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| Selected Language                              | The language to use for prompts and responses. Affects system prompt templates and examples.                                                                                                                                                                                                                                                                                                                                                                | en                      |
| LLM API                                        | The API(s) to use for tool execution. Options are dynamically populated from installed Home Assistant LLM integrations. Select "Assist" for device control via the built-in Assist API, or leave empty to disable tool execution.                                                                                                                                                                                                                           | Assist                  |
| System Prompt                                  | [see here](./Model%20Prompting.md)                                                                                                                                                                                                                                                                                                                                                                                                                          |                         |
| Additional attributes to expose in the context | Extra attributes that will be exposed to the model via the `{{ devices }}` template variable (e.g., rgb_color, brightness, temperature, humidity, fan_mode, volume_level)                                                                                                                                                                                                                                                                                   | See suggestions         |
| Enable in context learning (ICL) examples      | If enabled, will load examples from the specified file and expose them as the `{{ response_examples }}` variable in the system prompt template                                                                                                                                                                                                                                                                                                              | Enabled                 |
| In context learning examples CSV filename      | The file to load in context learning examples from. Must be located in the same directory as the custom component                                                                                                                                                                                                                                                                                                                                           | in_context_examples.csv |
| Number of ICL examples to generate             | The number of examples to select when expanding the `{{ in_context_examples }}` template in the prompt                                                                                                                                                                                                                                                                                                                                                      | 4                       |
| Max tool call iterations                       | **Agentic mode:** set to 3 or more to allow the model to call tools, receive results, and loop. **Legacy/1-shot mode:** set to 0 — the model produces its response and tool calls in one turn with no follow-up. Must be 0 when Legacy Tool Calling is enabled.                                                                                                                                                                                              | 3                       |

### Prompt format options

The options that tell Home LLM how a model writes its thinking and its tool calls - `thinking_prefix`,
`thinking_suffix`, `tool_call_prefix`, `tool_call_suffix`, `tool_response_as_string` and
`enable_legacy_tool_calling` - are chosen for you by the model family and are no longer offered in the UI. Home
Assistant's own models and the widely used instructions-tuned ones are covered there, and they stay available as
options for anything else.

# Llama.cpp
For what the sampling parameters mean, see the documentation of the server hosting the model.

## Connection & Model Selection
| Option Name        | Description                                                                                                | Suggested Value        |
| ------------------ | ---------------------------------------------------------------------------------------------------------- | ---------------------- |
| Chat Model         | The Hugging Face model repository or local model filename to use for inference                             | acon96/Home-3B-v3-GGUF |
| Model Quantization | The quantization level to download for the selected model from Hugging Face                                | Q4_K_M                 |
| Model File Path    | The full path to a local GGUF model file. If not specified, the model will be downloaded from Hugging Face |                        |

## Sampling & Output

> A sampling parameter left empty is left to the server, which knows the model it loaded. Fill one in only to override it for this model.

| Option Name                                 | Description                                                                                                                               | Suggested Value |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| Temperature                                 | Sampling parameter; see above note                                                                                                        | 0.1             |
| Top K                                       | Sampling parameter; see above note                                                                                                        | 40              |
| Top P                                       | Sampling parameter; see above note                                                                                                        | 1.0             |
| Min P                                       | Sampling parameter; see above note                                                                                                        | 0.0             |
| Typical P                                   | Sampling parameter; see above note                                                                                                        | 1.0             |
| Maximum tokens to return in response        | Limits the number of tokens that can be produced by each model response                                                                   | 512             |
| Context Length                                  | Tokens the model has for the conversation and its answer together. Home LLM drops the oldest messages that no longer fit, so a long conversation stays inside the limit instead of failing | 2048          |

## Performance Optimization
| Option Name            | Description                                                                                                          | Suggested Value                |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Batch Size             | Number of tokens to process in each batch. Higher values increase speed but consume more memory                      | 512                            |
| Thread Count           | Number of CPU threads to use for inference                                                                           | (number of physical CPU cores) |
| Batch Thread Count     | Number of threads to use for batch processing                                                                        | (number of physical CPU cores) |
| Enable Flash Attention | Use Flash Attention optimization if supported by the model. Can significantly improve performance on compatible GPUs | Disabled                       |

## Advanced Features
| Option Name             | Description                                                                                                                     | Suggested Value                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| GBNF Grammar | Restricts the output of the model to a grammar written in GBNF; eliminates function calling syntax errors. Leave it empty to let the model answer freely | |
| Enable Prompt Caching   | Cache the system prompt to avoid recomputing it on every turn (requires refresh_system_prompt to be disabled)                   | Disabled                                                           |
| Prompt Caching Interval | Number of seconds between prompt cache refreshes (if caching is enabled)                                                        | 30                                                                 |

## Wheels
The wheels for `llama-cpp-python` can be built or downloaded manually for installation/re-installation.

Take the appropriate wheel and copy it to the `custom_components/llama_conversation/` directory.

After the wheel file has been copied to the correct folder, attempt the wheel installation step of the integration setup. The local wheel file should be detected and installed.

## Pre-built
Pre-built wheel files (`*.whl`) are sourced from the [acon96/llama-cpp-python releases](https://github.com/acon96/llama-cpp-python/releases).

llama-cpp-python uses generic `py3-none` wheels that work across all Python 3.x versions. Wheels are provided per C library and architecture — the integration detects the platform and picks the right one automatically:
- **Home Assistant OS and the Home Assistant container** (musl-based):
    - **ARM devices** (e.g., Raspberry Pi 4/5): `llama_cpp_python-{version}-py3-none-musllinux_1_2_aarch64.whl`
    - **x86_64 devices** (e.g., Intel/AMD desktops): `llama_cpp_python-{version}-py3-none-musllinux_1_2_x86_64.whl`
- **glibc-based installations**:
    - **ARM devices**: `llama_cpp_python-{version}-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl`
    - **x86_64 devices**: `llama_cpp_python-{version}-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`

> **Note:** A single wheel works for all Python 3.x versions on the same platform. You no longer need to match your specific Python version (e.g., 3.12, 3.13, 3.14).

## Build your own

1. Clone the repository on the target machine that will be running Home Assistant
2. Ensure `docker` is installed
2. Run the `scripts/run_docker_to_make_wheels.sh` script
3. The compatible wheel files will be placed in the folder you executed the script from


# Llama.cpp Server
Llama.cpp Server backend is used when running inference via a separate `llama-cpp-python` HTTP server.

## Connection
| Option Name | Description                                               | Suggested Value |
| ----------- | --------------------------------------------------------- | --------------- |
| Host        | The hostname or IP address of the llama-cpp-python server |                 |
| Port        | The port number the server is listening on                | 8000            |
| SSL         | Whether to use HTTPS for the connection                   | false           |

## Sampling & Output
| Option Name                                 | Description                                                                                                                                                      | Suggested Value                                 |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Top K                                       | Sampling parameter; see your server's documentation | 40                                              |
| Top P                                       | Sampling parameter; see above note                                                                                                                               | 1.0                                             |
| Maximum tokens to return in response        | Limits the number of tokens that can be produced by each model response                                                                                          | 512                                             |
| Request Timeout                             | The maximum time in seconds that the integration will wait for a response from the remote server                                                                 | 90 (higher if running on low resource hardware) |

## Advanced Features
| Option Name           | Description                                                                                                 | Suggested Value |
| --------------------- | ----------------------------------------------------------------------------------------------------------- | --------------- |
| GBNF Grammar | Restricts the output of the model to a grammar written in GBNF; eliminates function calling syntax errors. Leave it empty to let the model answer freely | |


# Ollama
For details about Ollama's sampling parameters, see: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

## Connection
| Option Name | Description                                     | Suggested Value |
| ----------- | ----------------------------------------------- | --------------- |
| Host        | The hostname or IP address of the Ollama server |                 |
| Port        | The port number the server is listening on      | 11434           |
| SSL         | Whether to use HTTPS for the connection         | false           |

## Sampling & Output
| Option Name                                 | Description                                                                                                   | Suggested Value                                 |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Top K                                       | Sampling parameter; see above note                                                                            | 40                                              |
| Top P                                       | Sampling parameter; see above note                                                                            | 1.0                                             |
| Typical P                                   | Sampling parameter; see above note                                                                            | 1.0                                             |
| Maximum tokens to return in response        | Limits the number of tokens that can be produced by each model response                                       | 512                                             |
| Context Length                                  | Tokens the model has for the conversation and its answer together. Home LLM drops the oldest messages that no longer fit, so a long conversation stays inside the limit instead of failing | 2048          |
| Request Timeout                             | The maximum time in seconds that the integration will wait for a response from the remote server              | 90 (higher if running on low resource hardware) |

## Advanced Features
| Option Name                   | Description                                                                                                                     | Suggested Value |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| JSON Mode                     | Restricts the model to only output valid JSON objects. Enable this if you are using ICL and are getting invalid JSON responses. | True            |
| Keep Alive/Inactivity Timeout | The duration in minutes to keep the model loaded after each request. Set to a negative value to keep loaded forever             | 30 (minutes)    |

# Generic OpenAI API (Chat Completions)
For what the sampling parameters mean, see the documentation of the server hosting the model.

## Connection
| Option Name | Description                                                                  | Suggested Value |
| ----------- | ---------------------------------------------------------------------------- | --------------- |
| Host        | The hostname or IP address of the OpenAI-compatible API server               |                 |
| Port        | The port number the server is listening on (leave empty for default)         |                 |
| SSL         | Whether to use HTTPS for the connection                                      | false           |
| API Key     | The API key for authentication (if required by your server)                  |                 |
| API Path    | The path prefix for API requests (e.g., `/v1` for OpenAI-compatible servers) | v1              |

## Sampling & Output
| Option Name                                 | Description                                                                                                   | Suggested Value                                 |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Top P                                       | Sampling parameter; see above note                                                                            | 1.0                                             |
| Request Timeout                             | The maximum time in seconds that the integration will wait for a response from the remote server              | 90 (higher if running on low resource hardware) |

# Generic OpenAI Responses
The Responses backend keeps the conversation on the server it talks to, so Home LLM does not resend it: while a
conversation keeps going, only the message being answered is sent, together with the id of the response it follows.
When the conversation is not the one that id belongs to - a new conversation, or a second model sharing the same
connection - the whole exchange is sent again, system prompt included.

## Connection
| Option Name | Description                                                          | Suggested Value |
| ----------- | -------------------------------------------------------------------- | --------------- |
| Host        | The hostname or IP address of the OpenAI-compatible API server       |                 |
| Port        | The port number the server is listening on (leave empty for default) |                 |
| SSL         | Whether to use HTTPS for the connection                              | false           |
| API Key     | The API key for authentication (if required by your server)          |                 |
| API Path    | The path prefix for API requests                                     | v1              |

## Sampling & Output
| Option Name                                 | Description                                                                                                   | Suggested Value                                 |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Temperature                                 | Sampling parameter; see above note                                                                            | 0.1                                             |
| Top P                                       | Sampling parameter; see above note                                                                            | 1.0                                             |
| Request Timeout                             | The maximum time in seconds that the integration will wait for a response from the remote server              | 90 (higher if running on low resource hardware) |

| Option Name                          | Description                                                                                              | Suggested Value |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------- | --------------- |

# Anthropic API
Anthropic backend uses the [Anthropic Messages API](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) for tool execution. Supports vision (image attachments) and streaming. Works with any Anthropic-compatible API (including Claude, and third-party providers like AWS Bedrock, Azure, or self-hosted solutions using the Anthropic API format).

## Connection
| Option Name | Description                                                                                                 | Suggested Value |
| ----------- | ----------------------------------------------------------------------------------------------------------- | --------------- |
| Base URL    | The full base URL of the Anthropic-compatible API (e.g., `https://api.anthropic.com` or a compatible proxy) |                 |
| API Key     | The API key for authentication                                                                              |                 |

> **Note:** Unlike other backends, Anthropic uses a single `base_url` field instead of separate Host/Port/SSL fields. The API key is passed via headers.

## Sampling & Output
| Option Name                                 | Description                                                                                                   | Suggested Value                                 |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Temperature                                 | Sampling parameter; controls randomness in responses                                                          | 0.1                                             |
| Top K                                       | Sampling parameter; limits token selection to the top K candidates                                            | 40                                              |
| Top P                                       | Sampling parameter; nucleus sampling threshold                                                                | 1.0                                             |
| Maximum tokens to return in response        | Limits the number of tokens that can be produced by each model response                                       | 512                                             |
| Request Timeout                             | The maximum time in seconds that the integration will wait for a response from the remote server              | 90 (higher if running on low resource hardware) |

## Features
| Option Name    | Description                                                                   | Suggested Value     |
| -------------- | ----------------------------------------------------------------------------- | ------------------- |
| Vision Support | Anthropic models natively support image attachments — no configuration needed | Enabled (automatic) |
