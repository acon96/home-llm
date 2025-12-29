# Template Error Fix: LoggingUndefined Not JSON Serializable

## Error Message
```
Sorry, I had a problem with my template: TypeError: Type is not JSON serializable: LoggingUndefined
```

## Root Cause

The error occurs in the template rendering logic in `custom_components/llama_conversation/entity.py:662`. When your prompt template references a variable that doesn't exist, Jinja2 returns a special `LoggingUndefined` object. When Home Assistant tries to process this undefined value (during string formatting or JSON serialization), it fails because `LoggingUndefined` is not JSON serializable.

## The Problem Location

### In Your Template
The problematic line in your prompt template is:

```jinja2
Tools: {{ tools | to_json }}
```

### Why It Fails

The `tools` variable is **only available** when **"Enable Legacy Tool Calling"** is enabled in your configuration.

Looking at the code (`entity.py:645-655`), `tools` is only added to the template variables when `enable_legacy_tool_calling` is `True`:

```python
if enable_legacy_tool_calling:
    if llm_api:
        # ... tools are added here
        render_variables["tools"] = tools
```

When this setting is disabled (which is the default), the template tries to access `{{ tools }}`, gets a `LoggingUndefined` object, and then `to_json` fails trying to serialize it.

## Available Template Variables

When rendering your prompt template, only these variables are available:

- `devices` - List of exposed devices
- `formatted_devices` - String representation of devices
- `response_examples` - ICL examples (if enabled)
- `tool_call_prefix` / `tool_call_suffix` - Tool call markers
- `tools` / `formatted_tools` - Available tools (⚠️ **only if legacy calling enabled**)

## Solutions

### Option 1: Enable Legacy Tool Calling (Quick Fix)

1. Go to your LLaMA Conversation integration settings
2. Find the "Enable Legacy Tool Calling" option
3. Enable it
4. Save and restart

### Option 2: Fix the Template (Recommended)

Replace the problematic line:

```jinja2
Tools: {{ tools | to_json }}
```

With one of these conditional versions:

#### Using `is defined` check:
```jinja2
{% if tools is defined %}Tools: {{ tools | to_json }}{% endif %}
```

#### Using `default()` filter:
```jinja2
Tools: {{ tools | default("No tools configured") | to_json }}
```

#### Using `default()` with empty list:
```jinja2
Tools: {{ tools | default([]) | to_json }}
```

## Complete Fixed Template

```jinja2
You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.
The current time and date is {{ (as_timestamp(now()) | timestamp_custom("%I:%M %p on %A %B %d, %Y", True, "")) }}
{% if tools is defined %}Tools: {{ tools | to_json }}{% endif %}
Devices:
{% for device in devices | selectattr('area_id', 'none'): %}
{{ device.entity_id }} '{{ device.name }}' = {{ device.state }}{{ ([""] + device.attributes) | join(";") }}
{% endfor %}
{% for area in devices | rejectattr('area_id', 'none') | groupby('area_name') %}
## Area: {{ area.grouper }}
{% for device in area.list %}
{{ device.entity_id }} '{{ device.name }}' = {{ device.state }};{{ device.attributes | join(";") }}
{% endfor %}
{% endfor %}
{% for item in response_examples %}
{{ item.request }}
{{ item.response }}
<functioncall> {{ item.tool | to_json }}
{% endfor %}
```

## Prevention

To avoid this error in the future:

1. **Always check if optional variables are defined** before using them
2. **Use the `default()` filter** for variables that might not exist
3. **Stick to documented variables** listed in the integration documentation
4. **Test templates** after making changes

## Common Template Mistakes

❌ **Wrong:**
```jinja2
{{ some_typo }}              # Undefined variable
{{ devises }}                # Typo - should be "devices"
{{ device.missing_attr }}    # Attribute doesn't exist
{{ tools | to_json }}        # Not always available
```

✅ **Correct:**
```jinja2
{{ devices }}                           # Defined variable
{{ tools | default([]) }}               # With fallback
{% if tools is defined %}{{ tools }}{% endif %}  # With check
```

## Related Files

- `custom_components/llama_conversation/entity.py:662` - Template rendering
- `custom_components/llama_conversation/entity.py:637-660` - Available variables
- `custom_components/llama_conversation/entity.py:645-655` - Conditional `tools` variable

## Additional Notes

The `tools` variable is only populated when:
1. Legacy tool calling is enabled (`CONF_ENABLE_LEGACY_TOOL_CALLING = True`)
2. An LLM API is configured (`llm_api` is not `None`)

If you need tools in your template, ensure these conditions are met or use conditional checks as shown above.
