# Backend Configuration

There are multiple backends to choose for running the model that the Home Assistant integration uses. Here is a description of all the options for each backend

# Common Options
| Option Name | Description | Suggested Value |
| ------------ | --------- | ------------ |
| System Prompt | [see here](./Model%20Prompting.md) |  |
| Prompt Format | The format for the context of the model |  |
| Maximum tokens to return in response | Limits the number of tokens that can be produced by each model response | 512 |
| Additional attribute to expose in the context | Extra attributes that will be exposed to the model via the `{{ devices }}` template variable |  |
| Service Call Regex | The regular expression used to extract service calls from the model response; should contain 1 repeated capture group |  |
| Refresh System Prompt Every Turn | Flag to update the system prompt with updated device states on every chat turn. Disabling can significantly improve agent response times when using a backend that supports prefix caching (Llama.cpp) | Enabled |
| Remember conversation | Flag to remember the conversation history (excluding system prompt) in the model context. | Enabled |
| Number of past interactions to remember | If `Remember conversation` is enabled, number of user-assistant interaction pairs to keep in history. |  |

# Llama.cpp
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description
| Option Name | Description | Suggested Value |
| ------------ | --------- | ------------ |
| Top K | Sampling parameter; see above link | 40 |
| Top P | Sampling parameter; see above link | 1.0 |
| Temperature | Sampling parameter; see above link | 0.1 |
| Enable GBNF Grammar | Restricts the output of the model to follow a pre-defined syntax; eliminates function calling syntax errors on quantized models | Enabled |

# text-generation-webui
| Option Name | Description | Suggested Value |
| ------------ | --------- | ------------ |
| Request Timeout | The maximum time in seconds that the integration will wait for a response from the remote server | 90 (higher if running on low resource hardware) |
| Use chat completions endpoint | Flag to use `/v1/chat/completions` as the remote endpoint instead of `/v1/completions` |  |
| Generation Preset/Character Name | The preset or character name to pass to the backend. If none is provided then the settings that are currently selected in the UI will be applied |  |
| Chat Mode | [see here](https://github.com/oobabooga/text-generation-webui/wiki/01-%E2%80%90-Chat-Tab#mode) | Instruct |

# Generic OpenAI API Compatible
For details about the sampling parameters, see here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description
| Option Name | Description | Suggested Value |
| ------------ | --------- | ------------ |
| Request Timeout | The maximum time in seconds that the integration will wait for a response from the remote server | 90 (higher if running on low resource hardware) |
| Use chat completions endpoint | Flag to use `/v1/chat/completions` as the remote endpoint instead of `/v1/completions` | Backend Dependent |
| Top P | Sampling parameter; see above link | 1.0 |
| Temperature | Sampling parameter; see above link | 0.1 |
