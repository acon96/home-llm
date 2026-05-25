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
| Refresh System Prompt Every Turn               | Flag to update the system prompt with updated device states on every chat turn. Disabling can significantly improve agent response times when using a backend that supports prefix caching (Llama.cpp)                                                                                                                                                                                                                                                      | Enabled                 |
| Remember conversation                          | Flag to remember the conversation history (excluding system prompt) in the model context.                                                                                                                                                                                                                                                                                                                                                                   | Enabled                 |
| Number of past interactions to remember        | If `Remember conversation` is enabled, number of user-assistant interaction pairs to keep in history. Not used by Generic OpenAI Responses backend.                                                                                                                                                                                                                                                                                                         |                         |
| Enable in context learning (ICL) examples      | If enabled, will load examples from the specified file and expose them as the `{{ response_examples }}` variable in the system prompt template                                                                                                                                                                                                                                                                                                              | Enabled                 |
| In context learning examples CSV filename      | The file to load in context learning examples from. Must be located in the same directory as the custom component                                                                                                                                                                                                                                                                                                                                           | in_context_examples.csv |
| Number of ICL examples to generate             | The number of examples to select when expanding the `{{ in_context_examples }}` template in the prompt                                                                                                                                                                                                                                                                                                                                                      | 4                       |
| Thinking prefix                                | String prefix to mark the start of internal model reasoning (used when the model supports explicit thinking)                                                                                                                                                                                                                                                                                                                                                | `<think>`               |
| Thinking suffix                                | String suffix to mark the end of internal model reasoning                                                                                                                                                                                                                                                                                                                                                                                                   | `</think>`              |
| Tool call prefix                               | String prefix to mark the start of a function call in the model response                                                                                                                                                                                                                                                                                                                                                                                    | `<tool_call>`           |
| Tool call suffix                               | String suffix to mark the end of a function call in the model response                                                                                                                                                                                                                                                                                                                                                                                      | `</tool_call>`          |
| Enable legacy tool calling                     | **Legacy/1-shot mode.** If enabled, uses the legacy `` ```homeassistant `` tool calling format instead of the newer prefix/suffix format. Required for older Home-LLM models (v1–v3). Must be combined with Max Tool Call Iterations = 0. See [Tool Calling Modes](#tool-calling-modes) above.                                                                                                                                                             | Disabled                |
| Max tool call iterations                       | **Agentic mode:** set to 3 or more to allow the model to call tools, receive results, and loop. **Legacy/1-shot mode:** set to 0 — the model produces its response and tool calls in one turn with no follow-up. Must be 0 when Legacy Tool Calling is enabled.                                                                                                                                                                                              | 3                       |
| Use server defaults for sampling parameters    | **Recommended for most users.** When enabled, temperature, top_p, top_k, min_p, and typical_p are omitted from requests so the inference engine uses the values it knows are best for the loaded model. Inference engines and model providers have gotten much better at shipping well-tuned defaults — only disable this if you're seeing odd output behavior or intentionally want a specific temperature (e.g., lower for more deterministic responses). | Enabled                 |

# Llama.cpp
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description

## Connection & Model Selection
| Option Name        | Description                                                                                                | Suggested Value        |
| ------------------ | ---------------------------------------------------------------------------------------------------------- | ---------------------- |
| Chat Model         | The Hugging Face model repository or local model filename to use for inference                             | acon96/Home-3B-v3-GGUF |
| Model Quantization | The quantization level to download for the selected model from Hugging Face                                | Q4_K_M                 |
| Model File Path    | The full path to a local GGUF model file. If not specified, the model will be downloaded from Hugging Face |                        |

## Sampling & Output

> **Note:** Most users should leave **Use server defaults for sampling parameters** enabled (the default). Modern inference engines ship with well-tuned sampling values for each model — only override these if you are experiencing unexpected behavior or intentionally targeting a specific temperature.

| Option Name                                 | Description                                                                                                                               | Suggested Value |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| Use server defaults for sampling parameters | Omit temperature, top_p, top_k, min_p, and typical_p from requests so the server uses its own model defaults. Recommended for most users. | Enabled         |
| Temperature                                 | Sampling parameter; see above link                                                                                                        | 0.1             |
| Top K                                       | Sampling parameter; see above link                                                                                                        | 40              |
| Top P                                       | Sampling parameter; see above link                                                                                                        | 1.0             |
| Min P                                       | Sampling parameter; see above link                                                                                                        | 0.0             |
| Typical P                                   | Sampling parameter; see above link                                                                                                        | 1.0             |
| Maximum tokens to return in response        | Limits the number of tokens that can be produced by each model response                                                                   | 512             |
| Context Length                              | Maximum number of tokens the model can consider in its context window                                                                     | 2048            |

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
| Enable GBNF Grammar     | Restricts the output of the model to follow a pre-defined syntax; eliminates function calling syntax errors on quantized models | Enabled                                                            |
| GBNF Grammar Filename   | The file to load as the GBNF grammar. Must be located in the same directory as the custom component.                            | `output.gbnf` for Home LLM and `json.gbnf` for any model using ICL |
| Enable Prompt Caching   | Cache the system prompt to avoid recomputing it on every turn (requires refresh_system_prompt to be disabled)                   | Disabled                                                           |
| Prompt Caching Interval | Number of seconds between prompt cache refreshes (if caching is enabled)                                                        | 30                                                                 |

## Wheels
The wheels for `llama-cpp-python` can be built or downloaded manually for installation/re-installation.

Take the appropriate wheel and copy it to the `custom_components/llama_conversation/` directory.

After the wheel file has been copied to the correct folder, attempt the wheel installation step of the integration setup. The local wheel file should be detected and installed.

## Pre-built
Pre-built wheel files (`*.whl`) are built as part of a fork of llama-cpp-python and are available on the [GitHub releases](https://github.com/acon96/llama-cpp-python/releases/latest) page for the fork.

As of version 0.3.20, llama-cpp-python uses generic `py3-none` wheels that work across all Python 3.x versions. Select the correct `.whl` file for your hardware's architecture:
- **ARM devices** (e.g., Raspberry Pi 4/5):
    - Example filename:
        - `llama_cpp_python-{version}-py3-none-linux_aarch64.whl`
- **x86_64 devices** (e.g., Intel/AMD desktops):
    - Example filename:
        - `llama_cpp_python-{version}-py3-none-linux_x86_64.whl`

> **Note:** A single wheel now works for all Python 3.x versions on the same platform. You no longer need to match your specific Python version (e.g., 3.12, 3.13, 3.14).

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
| Use server defaults for sampling parameters | Omit sampling parameters from requests so the server uses its own model defaults. Recommended for most users.                                                    | Enabled                                         |
| Top K                                       | Sampling parameter; see [text-generation-webui wiki](https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description) | 40                                              |
| Top P                                       | Sampling parameter; see above link                                                                                                                               | 1.0                                             |
| Maximum tokens to return in response        | Limits the number of tokens that can be produced by each model response                                                                                          | 512                                             |
| Request Timeout                             | The maximum time in seconds that the integration will wait for a response from the remote server                                                                 | 90 (higher if running on low resource hardware) |

## Advanced Features
| Option Name           | Description                                                                                                 | Suggested Value |
| --------------------- | ----------------------------------------------------------------------------------------------------------- | --------------- |
| Enable GBNF Grammar   | Restricts the output of the model to follow a pre-defined syntax; eliminates function calling syntax errors | Enabled         |
| GBNF Grammar Filename | The file to load as the GBNF grammar. Must be located in the same directory as the custom component.        | `output.gbnf`   |


# text-generation-webui
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description

## Connection
| Option Name | Description                                                                           | Suggested Value |
| ----------- | ------------------------------------------------------------------------------------- | --------------- |
| Host        | The hostname or IP address of the text-generation-webui server                        |                 |
| Port        | The port number the server is listening on                                            | 5000            |
| SSL         | Whether to use HTTPS for the connection                                               | false           |
| Admin Key   | The admin key for the text-generation-webui server (if configured for authentication) |                 |

> **Note:** The default-cpu Docker image exposes port 7860 (Web UI). If using the provided `docker-compose.yml`, set the port to **7860**. Port 5000 is the API-only mode port.

## Sampling & Output
| Option Name                                 | Description                                                                                                   | Suggested Value                                 |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Use server defaults for sampling parameters | Omit sampling parameters from requests so the server uses its own model defaults. Recommended for most users. | Enabled                                         |
| Temperature                                 | Sampling parameter; see above link                                                                            | 0.1                                             |
| Top K                                       | Sampling parameter; see above link                                                                            | 40                                              |
| Top P                                       | Sampling parameter; see above link                                                                            | 1.0                                             |
| Min P                                       | Sampling parameter; see above link                                                                            | 0.0                                             |
| Typical P                                   | Sampling parameter; see above link                                                                            | 1.0                                             |
| Context Length                              | Maximum number of tokens the model can consider in its context window                                         | 2048                                            |
| Request Timeout                             | The maximum time in seconds that the integration will wait for a response from the remote server              | 90 (higher if running on low resource hardware) |

## UI Configuration
| Option Name                      | Description                                                                                                                                      | Suggested Value |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- |
| Generation Preset/Character Name | The preset or character name to pass to the backend. If none is provided then the settings that are currently selected in the UI will be applied |                 |
| Chat Mode                        | [see here](https://github.com/oobabooga/text-generation-webui/wiki/01-%E2%80%90-Chat-Tab#mode)                                                   | Instruct        |

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
| Use server defaults for sampling parameters | Omit sampling parameters from requests so the server uses its own model defaults. Recommended for most users. | Enabled                                         |
| Top K                                       | Sampling parameter; see above link                                                                            | 40                                              |
| Top P                                       | Sampling parameter; see above link                                                                            | 1.0                                             |
| Typical P                                   | Sampling parameter; see above link                                                                            | 1.0                                             |
| Maximum tokens to return in response        | Limits the number of tokens that can be produced by each model response                                       | 512                                             |
| Context Length                              | Maximum number of tokens the model can consider in its context window                                         | 2048                                            |
| Request Timeout                             | The maximum time in seconds that the integration will wait for a response from the remote server              | 90 (higher if running on low resource hardware) |

## Advanced Features
| Option Name                   | Description                                                                                                                     | Suggested Value |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| JSON Mode                     | Restricts the model to only output valid JSON objects. Enable this if you are using ICL and are getting invalid JSON responses. | True            |
| Keep Alive/Inactivity Timeout | The duration in minutes to keep the model loaded after each request. Set to a negative value to keep loaded forever             | 30 (minutes)    |

# Generic OpenAI API (Chat Completions)
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description

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
| Use server defaults for sampling parameters | Omit sampling parameters from requests so the server uses its own model defaults. Recommended for most users. | Enabled                                         |
| Top P                                       | Sampling parameter; see above link                                                                            | 1.0                                             |
| Request Timeout                             | The maximum time in seconds that the integration will wait for a response from the remote server              | 90 (higher if running on low resource hardware) |

# Generic OpenAI Responses
Generic OpenAI Responses backend uses time-based conversation memory instead of interaction counts and is compatible with specialized response APIs.

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
| Use server defaults for sampling parameters | Omit sampling parameters from requests so the server uses its own model defaults. Recommended for most users. | Enabled                                         |
| Temperature                                 | Sampling parameter; see above link                                                                            | 0.1                                             |
| Top P                                       | Sampling parameter; see above link                                                                            | 1.0                                             |
| Request Timeout                             | The maximum time in seconds that the integration will wait for a response from the remote server              | 90 (higher if running on low resource hardware) |

## Memory & Conversation
| Option Name                          | Description                                                                                              | Suggested Value |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------- | --------------- |
| Remember conversation time (minutes) | Number of minutes to remember conversation history. Uses time-based memory instead of interaction count. | 2 (minutes)     |

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
| Use server defaults for sampling parameters | Omit sampling parameters from requests so the server uses its own model defaults. Recommended for most users. | Enabled                                         |
| Temperature                                 | Sampling parameter; controls randomness in responses                                                          | 0.1                                             |
| Top K                                       | Sampling parameter; limits token selection to the top K candidates                                            | 40                                              |
| Top P                                       | Sampling parameter; nucleus sampling threshold                                                                | 1.0                                             |
| Maximum tokens to return in response        | Limits the number of tokens that can be produced by each model response                                       | 512                                             |
| Request Timeout                             | The maximum time in seconds that the integration will wait for a response from the remote server              | 90 (higher if running on low resource hardware) |

## Features
| Option Name    | Description                                                                   | Suggested Value     |
| -------------- | ----------------------------------------------------------------------------- | ------------------- |
| Vision Support | Anthropic models natively support image attachments — no configuration needed | Enabled (automatic) |
