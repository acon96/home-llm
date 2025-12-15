# Using AI Tasks
The AI Tasks feature allows you to define structured tasks that your local LLM can perform. These tasks can be integrated into Home Assistant automations and scripts, enabling you to generate dynamic content based on specific prompts and instructions.

## Setting up an AI Task Handler
Setting up a task handler is similar to setting up a conversation agent. You can choose to run the model directly within Home Assistant using `llama-cpp-python`, or you can use an external backend like Ollama. See the [Setup Guide](./docs/Setup.md) for detailed instructions on configuring your AI Task handler.

The specific configuration options for AI Tasks are:
| Option Name                       | Description                                                                                                                         |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Structured Data Extraction Method | Choose how the AI Task should extract structured data from the model's output. Options include `structured_output` and `tool`.      |
| Data Extraction Retry Count       | The number of times to retry data extraction if the initial attempt fails. Useful when models can produce incorrect tool responses. |

If no structured data extraction method is specified, then the task entity will always return raw text.

## Using an AI Task in a Script or Automation
To use an AI Task in a Home Assistant script or automation, you can utilize the `ai_task.generate_data` action. This action allows you to specify the task name, instructions, and the structure of the expected output. Below is an example of a script that generates a joke about a smart device in your home.

**Device Joke Script:**
```yaml
sequence:
  - action: ai_task.generate_data
    data:
      task_name: Device Joke Generation
      instructions: |
        Write a funny joke about one of the smart devices in my home.
        Here are all of the smart devices I have: 
        {% for device in states | rejectattr('domain', 'in', ['update', 'event']) -%}
        - {{ device.name }} ({{device.domain}})
        {% endfor %}
      # You MUST set this to your own LLM entity ID if you do not set a default one in HA Settings
      # entity_id: ai_task.unsloth_qwen3_0_6b_gguf_unsloth_qwen3_0_6b_gguf
      structure:
        joke_setup:
          description: The beginning of a joke about a smart device in the home
          required: true
          selector:
            text: null
        joke_punchline:
          description: The punchline of the same joke about the smart device
          required: true
          selector:
            text: null
    response_variable: joke_output
  - action: notify.persistent_notification
    data:
      message: |-
        {{ joke_output.data.joke_setup }}
        ...
        {{ joke_output.data.joke_punchline }}
alias: Device Joke
description: "Generates a funny joke about one of the smart devices in the home."
```
