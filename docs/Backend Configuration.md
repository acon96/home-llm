# Backend Configuration

There are multiple backends to choose for running the model that the Home Assistant integration uses. Here is a description of all the options for each backend

# Common Options
These options are available for all backends and control model inference behavior, conversation memory, and integration-specific settings.

| Option Name                                   | Description                                                                                                                                                                                            | Suggested Value         |
|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| Selected Language                             | The language to use for prompts and responses. Affects system prompt templates and examples.                                                                                                           | en                      |
| LLM API                                       | The API to use for tool execution. Select "Assist" for the built-in Home Assistant API, or "No control" to disable tool execution. Other options are specialized APIs like Home-LLM v1/v2/v3.          | Assist                  |
| System Prompt                                 | [see here](./Model%20Prompting.md)                                                                                                                                                                     |                         |
| Additional attributes to expose in the context | Extra attributes that will be exposed to the model via the `{{ devices }}` template variable (e.g., rgb_color, brightness, temperature, humidity, fan_mode, volume_level)                             | See suggestions         |
| Refresh System Prompt Every Turn              | Flag to update the system prompt with updated device states on every chat turn. Disabling can significantly improve agent response times when using a backend that supports prefix caching (Llama.cpp) | Enabled                 |
| Remember conversation                         | Flag to remember the conversation history (excluding system prompt) in the model context.                                                                                                              | Enabled                 |
| Number of past interactions to remember       | If `Remember conversation` is enabled, number of user-assistant interaction pairs to keep in history. Not used by Generic OpenAI Responses backend.                                                    |                         |
| Enable in context learning (ICL) examples     | If enabled, will load examples from the specified file and expose them as the `{{ response_examples }}` variable in the system prompt template                                                         | Enabled                 |
| In context learning examples CSV filename     | The file to load in context learning examples from. Must be located in the same directory as the custom component                                                                                      | in_context_examples.csv |
| Number of ICL examples to generate            | The number of examples to select when expanding the `{{ in_context_examples }}` template in the prompt                                                                                                 | 4                       |
| Thinking prefix                               | String prefix to mark the start of internal model reasoning (used when the model supports explicit thinking)                                                                                           | `<think>`               |
| Thinking suffix                               | String suffix to mark the end of internal model reasoning                                                                                                                                              | `</think>`              |
| Tool call prefix                              | String prefix to mark the start of a function call in the model response                                                                                                                               | `<tool_call>`           |
| Tool call suffix                              | String suffix to mark the end of a function call in the model response                                                                                                                                 | `</tool_call>`          |
| Enable legacy tool calling                    | If enabled, uses the legacy `\`\`\`homeassistant` tool calling format instead of the newer prefix/suffix format. Required for some older Home-LLM models.                                              | Disabled                |
| Max tool call iterations                      | Maximum number of times the model can make tool calls in sequence before the conversation is terminated                                                                                                | 3                       |

# Llama.cpp
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description

## Connection & Model Selection
| Option Name           | Description                                                                                                                    | Suggested Value        |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------|------------------------|
| Chat Model            | The Hugging Face model repository or local model filename to use for inference                                                 | acon96/Home-3B-v3-GGUF |
| Model Quantization    | The quantization level to download for the selected model from Hugging Face                                                    | Q4_K_M                 |
| Model File Path       | The full path to a local GGUF model file. If not specified, the model will be downloaded from Hugging Face                     |                        |

## Sampling & Output
| Option Name           | Description                                                                                                                     | Suggested Value |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|-----------------|
| Temperature           | Sampling parameter; see above link                                                                                              | 0.1             |
| Top K                 | Sampling parameter; see above link                                                                                              | 40              |
| Top P                 | Sampling parameter; see above link                                                                                              | 1.0             |
| Min P                 | Sampling parameter; see above link                                                                                              | 0.0             |
| Typical P             | Sampling parameter; see above link                                                                                              | 1.0             |
| Maximum tokens to return in response | Limits the number of tokens that can be produced by each model response                                          | 512             |
| Context Length        | Maximum number of tokens the model can consider in its context window                                                           | 2048            |

## Performance Optimization
| Option Name           | Description                                                                                                                     | Suggested Value                |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| Batch Size            | Number of tokens to process in each batch. Higher values increase speed but consume more memory                                 | 512                            |
| Thread Count          | Number of CPU threads to use for inference                                                                                      | (number of physical CPU cores) |
| Batch Thread Count    | Number of threads to use for batch processing                                                                                   | (number of physical CPU cores) |
| Enable Flash Attention | Use Flash Attention optimization if supported by the model. Can significantly improve performance on compatible GPUs           | Disabled                       |

## Advanced Features
| Option Name           | Description                                                                                                                     | Suggested Value                                                    |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Enable GBNF Grammar   | Restricts the output of the model to follow a pre-defined syntax; eliminates function calling syntax errors on quantized models | Enabled                                                            |
| GBNF Grammar Filename | The file to load as the GBNF grammar. Must be located in the same directory as the custom component.                            | `output.gbnf` for Home LLM and `json.gbnf` for any model using ICL |
| Enable Prompt Caching | Cache the system prompt to avoid recomputing it on every turn (requires refresh_system_prompt to be disabled)                   | Disabled                                                           |
| Prompt Caching Interval | Number of seconds between prompt cache refreshes (if caching is enabled)                                                      | 30                                                                 |

## Wheels
The wheels for `llama-cpp-python` can be built or downloaded manually for installation/re-installation.

Take the appropriate wheel and copy it to the `custom_components/llama_conversation/` directory.

After the wheel file has been copied to the correct folder, attempt the wheel installation step of the integration setup. The local wheel file should be detected and installed.

## Pre-built
Pre-built wheel files (`*.whl`) are built as part of a fork of llama-cpp-python and are available on the [GitHub releases](https://github.com/acon96/llama-cpp-python/releases/latest) page for the fork.

To ensure compatibility with your Home Assistant and Python versions, select the correct `.whl` file for your hardware's architecture:
- For Home Assistant `2024.2.0` and newer, use the Python 3.12 wheels (`cp312`)
- **ARM devices** (e.g., Raspberry Pi 4/5):
    - Example filename:
        - `llama_cpp_python-{version}-cp312-cp312-musllinux_1_2_aarch64.whl`
- **x86_64 devices** (e.g., Intel/AMD desktops):
    - Example filename:
        - `llama_cpp_python-{version}-cp312-cp312-musllinux_1_2_x86_64.whl`

## Build your own

1. Clone the repository on the target machine that will be running Home Assistant
2. Ensure `docker` is installed
2. Run the `scripts/run_docker_to_make_wheels.sh` script
3. The compatible wheel files will be placed in the folder you executed the script from


# Llama.cpp Server
Llama.cpp Server backend is used when running inference via a separate `llama-cpp-python` HTTP server.

## Connection
| Option Name           | Description                                                                                                                     | Suggested Value                                                    |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Host                  | The hostname or IP address of the llama-cpp-python server                                                                       |                                                                    |
| Port                  | The port number the server is listening on                                                                                      | 8000                                                               |
| SSL                   | Whether to use HTTPS for the connection                                                                                         | false                                                              |

## Sampling & Output
| Option Name           | Description                                                                                                                     | Suggested Value                                                    |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Top K                 | Sampling parameter; see [text-generation-webui wiki](https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description)                                                                                              | 40                                                                 |
| Top P                 | Sampling parameter; see above link                                                                                              | 1.0                                                                |
| Maximum tokens to return in response | Limits the number of tokens that can be produced by each model response                                                                                | 512                                                                |
| Request Timeout       | The maximum time in seconds that the integration will wait for a response from the remote server                                | 90 (higher if running on low resource hardware)                   |

## Advanced Features
| Option Name           | Description                                                                                                                     | Suggested Value                                                    |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Enable GBNF Grammar   | Restricts the output of the model to follow a pre-defined syntax; eliminates function calling syntax errors                    | Enabled                                                            |
| GBNF Grammar Filename | The file to load as the GBNF grammar. Must be located in the same directory as the custom component.                            | `output.gbnf`                                                      |


# text-generation-webui
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description

## Connection
| Option Name           | Description                                                                                                                     | Suggested Value                                                    |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Host                  | The hostname or IP address of the text-generation-webui server                                                                  |                                                                    |
| Port                  | The port number the server is listening on                                                                                      | 5000                                                               |
| SSL                   | Whether to use HTTPS for the connection                                                                                         | false                                                              |
| Admin Key             | The admin key for the text-generation-webui server (if configured for authentication)                                           |                                                                    |

## Sampling & Output
| Option Name                      | Description                                                                                                                      | Suggested Value                                 |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Temperature                      | Sampling parameter; see above link                                                                                              | 0.1                                             |
| Top K                            | Sampling parameter; see above link                                                                                               | 40                                              |
| Top P                            | Sampling parameter; see above link                                                                                               | 1.0                                             |
| Min P                            | Sampling parameter; see above link                                                                                               | 0.0                                             |
| Typical P                        | Sampling parameter; see above link                                                                                               | 1.0                                             |
| Context Length                   | Maximum number of tokens the model can consider in its context window                                                             | 2048                                            |
| Request Timeout                  | The maximum time in seconds that the integration will wait for a response from the remote server                                 | 90 (higher if running on low resource hardware) |

## UI Configuration
| Option Name                      | Description                                                                                                                      | Suggested Value                                 |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Generation Preset/Character Name | The preset or character name to pass to the backend. If none is provided then the settings that are currently selected in the UI will be applied |                                                 |
| Chat Mode                        | [see here](https://github.com/oobabooga/text-generation-webui/wiki/01-%E2%80%90-Chat-Tab#mode)                                   | Instruct                                        |

# Ollama
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description

## Connection
| Option Name           | Description                                                                                                                     | Suggested Value                                                    |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Host                  | The hostname or IP address of the Ollama server                                                                                 |                                                                    |
| Port                  | The port number the server is listening on                                                                                      | 11434                                                              |
| SSL                   | Whether to use HTTPS for the connection                                                                                         | false                                                              |

## Sampling & Output
| Option Name                   | Description                                                                                                                    | Suggested Value                                 |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Top K                         | Sampling parameter; see above link                                                                                             | 40                                              |
| Top P                         | Sampling parameter; see above link                                                                                             | 1.0                                             |
| Typical P                     | Sampling parameter; see above link                                                                                             | 1.0                                             |
| Maximum tokens to return in response | Limits the number of tokens that can be produced by each model response                                                 | 512                                             |
| Context Length                | Maximum number of tokens the model can consider in its context window                                                            | 2048                                            |
| Request Timeout               | The maximum time in seconds that the integration will wait for a response from the remote server                               | 90 (higher if running on low resource hardware) |

## Advanced Features
| Option Name                   | Description                                                                                                                    | Suggested Value                                 |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| JSON Mode                     | Restricts the model to only output valid JSON objects. Enable this if you are using ICL and are getting invalid JSON responses. | True                                            |
| Keep Alive/Inactivity Timeout | The duration in minutes to keep the model loaded after each request. Set to a negative value to keep loaded forever            | 30 (minutes)                                    |

# Generic OpenAI API (Chat Completions)
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description

## Connection
| Option Name           | Description                                                                                                                     | Suggested Value                                                    |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Host                  | The hostname or IP address of the OpenAI-compatible API server                                                                  |                                                                    |
| Port                  | The port number the server is listening on (leave empty for default)                                                            |                                                                    |
| SSL                   | Whether to use HTTPS for the connection                                                                                         | false                                                              |
| API Key               | The API key for authentication (if required by your server)                                                                     |                                                                    |
| API Path              | The path prefix for API requests (e.g., `/v1` for OpenAI-compatible servers)                                                   | v1                                                                |

## Sampling & Output
| Option Name           | Description                                                                                                                     | Suggested Value                                 |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Top P                 | Sampling parameter; see above link                                                                                               | 1.0                                             |
| Request Timeout       | The maximum time in seconds that the integration will wait for a response from the remote server                                | 90 (higher if running on low resource hardware) |

# Generic OpenAI Responses
Generic OpenAI Responses backend uses time-based conversation memory instead of interaction counts and is compatible with specialized response APIs.

## Connection
| Option Name           | Description                                                                                                                     | Suggested Value                                                    |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Host                  | The hostname or IP address of the OpenAI-compatible API server                                                                  |                                                                    |
| Port                  | The port number the server is listening on (leave empty for default)                                                            |                                                                    |
| SSL                   | Whether to use HTTPS for the connection                                                                                         | false                                                              |
| API Key               | The API key for authentication (if required by your server)                                                                     |                                                                    |
| API Path              | The path prefix for API requests                                                                                                | v1                                                                |

## Sampling & Output
| Option Name                      | Description                                                                                                                     | Suggested Value                                 |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Temperature                      | Sampling parameter; see above link                                                                                              | 0.1                                             |
| Top P                            | Sampling parameter; see above link                                                                                               | 1.0                                             |
| Request Timeout                  | The maximum time in seconds that the integration will wait for a response from the remote server                                 | 90 (higher if running on low resource hardware) |

## Memory & Conversation
| Option Name                           | Description                                                                                                                     | Suggested Value |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-----------------|
| Remember conversation time (minutes) | Number of minutes to remember conversation history. Uses time-based memory instead of interaction count.                       | 2 (minutes)     |
