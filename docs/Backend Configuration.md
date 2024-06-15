# Backend Configuration

There are multiple backends to choose for running the model that the Home Assistant integration uses. Here is a description of all the options for each backend

# Common Options
| Option Name                                   | Description                                                                                                                                                                                            | Suggested Value |
|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| LLM API                                       | This is the set of tools that are provided to the LLM. Use Assist for the built-in API. If you are using Home-LLM v1, v2, or v3, then select the dedicated API                                         |                 |
| System Prompt                                 | [see here](./Model%20Prompting.md)                                                                                                                                                                     |                 |
| Prompt Format                                 | The format for the context of the model                                                                                                                                                                |                 |
| Tool Format                                   | The format of the tools that are provided to the model. Full, Reduced, or Minimal                                                                                                                      |                 |
| Multi-Turn Tool Use                           | Enable this if the model you are using expects to receive the result from the tool call before responding to the user                                                                                  |                 |
| Maximum tokens to return in response          | Limits the number of tokens that can be produced by each model response                                                                                                                                | 512             |
| Additional attribute to expose in the context | Extra attributes that will be exposed to the model via the `{{ devices }}` template variable                                                                                                           |                 |
| Arguments allowed to be pass to service calls | Any arguments not listed here will be filtered out of service calls. Used to restrict the model from modifying certain parts of your home.                                                             |                 |
| Service Call Regex                            | The regular expression used to extract service calls from the model response; should contain 1 repeated capture group                                                                                  |                 |
| Refresh System Prompt Every Turn              | Flag to update the system prompt with updated device states on every chat turn. Disabling can significantly improve agent response times when using a backend that supports prefix caching (Llama.cpp) | Enabled         |
| Remember conversation                         | Flag to remember the conversation history (excluding system prompt) in the model context.                                                                                                              | Enabled         |
| Number of past interactions to remember       | If `Remember conversation` is enabled, number of user-assistant interaction pairs to keep in history.                                                                                                  |                 |
| Enable in context learning (ICL) examples     | If enabled, will load examples from the specified file and expose them as the `{{ response_examples }}` variable in the system prompt template                                                         |                 |
| In context learning examples CSV filename     | The file to load in context learning examples from. Must be located in the same directory as the custom component                                                                                      |                 |
| Number of ICL examples to generate            | The number of examples to select when expanding the `{{ in_context_examples }}` template in the prompt                                                                                                 |                 |

# Llama.cpp
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description
| Option Name           | Description                                                                                                                     | Suggested Value                                                    |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Top K                 | Sampling parameter; see above link                                                                                              | 40                                                                 |
| Top P                 | Sampling parameter; see above link                                                                                              | 1.0                                                                |
| Temperature           | Sampling parameter; see above link                                                                                              | 0.1                                                                |
| Min P                 | Sampling parameter; see above link                                                                                              | 0.1                                                                |
| Typical P             | Sampling parameter; see above link                                                                                              | 0.95                                                               |
| Enable GBNF Grammar   | Restricts the output of the model to follow a pre-defined syntax; eliminates function calling syntax errors on quantized models | Enabled                                                            |
| GBNF Grammar Filename | The file to load as the GBNF grammar. Must be located in the same directory as the custom component.                            | `output.gbnf` for Home LLM and `json.gbnf` for any model using ICL |

## Wheels
The wheels for `llama-cpp-python` can be built or downloaded manually for installation.

Take the appropriate wheel and copy it to the `custom_components/llama_conversation/` directory.

After the wheel file has been copied to the correct folder, attempt the wheel installation step of the integration setup. The local wheel file should be detected and installed.

## Pre-built
Pre-built wheel files (`*.whl`) are provided as part of the [GitHub release](https://github.com/acon96/home-llm/releases/latest) for the integration.

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


# text-generation-webui
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description
| Option Name                      | Description                                                                                                                                      | Suggested Value                                 |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Request Timeout                  | The maximum time in seconds that the integration will wait for a response from the remote server                                                 | 90 (higher if running on low resource hardware) |
| Use chat completions endpoint    | If set, tells text-generation-webui to format the prompt instead of this extension. Prompt Format set here will not apply if this is enabled     |                                                 |
| Generation Preset/Character Name | The preset or character name to pass to the backend. If none is provided then the settings that are currently selected in the UI will be applied |                                                 |
| Chat Mode                        | [see here](https://github.com/oobabooga/text-generation-webui/wiki/01-%E2%80%90-Chat-Tab#mode)                                                   | Instruct                                        |
| Top K                            | Sampling parameter; see above link                                                                                                               | 40                                              |
| Top P                            | Sampling parameter; see above link                                                                                                               | 1.0                                             |
| Temperature                      | Sampling parameter; see above link                                                                                                               | 0.1                                             |
| Min P                            | Sampling parameter; see above link                                                                                                               | 0.1                                             |
| Typical P                        | Sampling parameter; see above link                                                                                                               | 0.95                                            |

# Ollama
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description
| Option Name                   | Description                                                                                                                    | Suggested Value                                 |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Request Timeout               | The maximum time in seconds that the integration will wait for a response from the remote server                               | 90 (higher if running on low resource hardware) |
| Keep Alive/Inactivity Timeout | The duration in minutes to keep the model loaded after each request. Set to a negative value to keep loaded forever            | 30m                                             |
| Use chat completions endpoint | If set, tells Ollama to format the prompt instead of this extension. Prompt Format set here will not apply if this is enabled  |                                                 |
| JSON Mode                     | Restricts the model to only ouput valid JSON objects. Enable this if you are using ICL and are getting invalid JSON responses. | True                                            |
| Top K                         | Sampling parameter; see above link                                                                                             | 40                                              |
| Top P                         | Sampling parameter; see above link                                                                                             | 1.0                                             |
| Temperature                   | Sampling parameter; see above link                                                                                             | 0.1                                             |
| Typical P                     | Sampling parameter; see above link                                                                                             | 0.95                                            |

# Generic OpenAI API Compatible
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description
| Option Name                   | Description                                                                                      | Suggested Value                                 |
|-------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Request Timeout               | The maximum time in seconds that the integration will wait for a response from the remote server | 90 (higher if running on low resource hardware) |
| Use chat completions endpoint | Flag to use `/v1/chat/completions` as the remote endpoint instead of `/v1/completions`           | Backend Dependent                               |
| Top P                         | Sampling parameter; see above link                                                               | 1.0                                             |
| Temperature                   | Sampling parameter; see above link                                                               | 0.1                                             |
