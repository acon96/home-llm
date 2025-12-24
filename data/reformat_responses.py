#!/usr/bin/env python3
"""
Script to reformat CSV responses by splitting the 'response' column into
'response_starting' (action in progress) and 'response_confirmed' (action completed)
using llama.cpp's native chat completion endpoint with concurrent aiohttp calls.

pip3 install aiohttp pydantic tqdm
"""
import argparse
import asyncio
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
from pydantic import BaseModel, Field
from tqdm import tqdm


class ResponseFormat(BaseModel):
    response_starting: str = Field(description="Response indicating the action is in progress")
    response_confirmed: str = Field(description="Response indicating the action has been completed")


TASK_DESCRIPTION = """
You are reformatting assistant responses for a smart home system.

Given an original response that describes an action, generate TWO variations:
1. response_starting: A response indicating the action is IN PROGRESS (starting, initiating, working on it)
2. response_confirmed: A response indicating the action has been SUCCESSFULLY COMPLETED (done, completed, finished)

Both responses should:
- Maintain the same tone and persona as the original
- Keep any device names or parameters (like <device_name>, <temp_f>, etc.) exactly as they appear
- Be natural and conversational
- Be concise (similar length to the original)
- Avoid overly formal language
- Preserve the language of the original data even if it is not English

Example:
Original: "Opening the blinds for you."
{
  "response_starting": "Opening the blinds now."
  "response_confirmed": "The blinds are now open."
}

Original: "Setting temperature to <temp_f> degrees."
{
  "response_starting": "Setting temperature to <temp_f> degrees."
  "response_confirmed": "Temperature has been set to <temp_f> degrees."
}

Respond ONLY with a JSON object in this exact format:
{
    "response_starting": "your starting response here",
    "response_confirmed": "your confirmed response here"
}
"""

JSON_SCHEMA = ResponseFormat.model_json_schema()


def load_system_prompts(system_prompts_path: Path) -> Dict[str, str]:
    """Load system prompts from CSV file."""
    prompts: Dict[str, str] = {}
    with open(system_prompts_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts[row['persona']] = row['prompt']
    return prompts


def _extract_message_content(payload: Dict) -> str:
    """Extract assistant message content from diverse llama.cpp response shapes."""
    if not isinstance(payload, dict):
        return ''

    if 'choices' in payload and payload['choices']:
        message = payload['choices'][0].get('message', {})
        content = message.get('content')
    else:
        content = payload.get('content')

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get('text', ''))
            elif isinstance(item, str):
                parts.append(item)
        return ''.join(parts)
    if isinstance(content, str):
        return content
    return ''


async def generate_reformatted_responses(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    system_prompt: str,
    original_response: str,
    service: str,
    semaphore: asyncio.Semaphore,
    max_attempts: int,
) -> Dict[str, str]:
    """Use llama.cpp chat endpoint to generate structured responses with retries."""

    user_message = f"""Service: {service}
Original response: {original_response}

Generate the two response variations as specified."""

    conversation = [
        {
            "role": "system",
            "content": "You are a helpful synthetic data generation assistant. Your task is to respond only using JSON format as specified."
        },
        {
            "role": "user",
            "content": f"{TASK_DESCRIPTION}\n\nCurrent Persona System Prompt: {system_prompt}\n{user_message}"
        },
    ]

    schema_payload = {
        "type": "json_schema",
        "json_schema": {
            "name": "response_format",
            "schema": JSON_SCHEMA,
        },
    }

    attempts_remaining = max_attempts
    last_error: Optional[Exception] = None

    content = None

    while attempts_remaining > 0:
        attempts_remaining -= 1
        payload = {
            "model": model,
            "messages": conversation,
            "seed": random.randint(1, 1_000_000),
            "response_format": schema_payload,
            "stream": False,
        }

        try:
            async with semaphore:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()

            content = _extract_message_content(data)
            result = json.loads(content)

            if 'response_starting' not in result or 'response_confirmed' not in result:
                raise ValueError(f"Invalid response format: {result}")

            return {
                'response_starting': result['response_starting'],
                'response_confirmed': result['response_confirmed'],
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"\nError processing response '{original_response}': {exc} - new response '{content}'")
            conversation.append({
                "role": "user",
                "content": "The previous response was invalid. Respond ONLY with JSON matching the schema: " + str(JSON_SCHEMA),
            })

    print(
        f"Failed to reformat response after multiple attempts. Using original response of '{original_response}'."
    )
    if last_error:
        print(f"Last error: {last_error}")
    return {
        'response_starting': original_response,
        'response_confirmed': original_response,
    }


async def process_csv(
    input_path: Path,
    output_path: Path,
    system_prompts_path: Path,
    base_endpoint: str,
    route: str,
    api_key: str | None,
    model: str,
    max_concurrency: int,
    max_attempts: int,
    request_timeout: float,
):
    """Process the CSV file and generate reformatted responses concurrently."""

    print("Loading system prompts...")
    system_prompts = load_system_prompts(system_prompts_path)
    print(f"Loaded {len(system_prompts)} personas: {', '.join(system_prompts.keys())}")

    print(f"\nReading input file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
    total_rows = len(rows)
    print(f"Processing {total_rows} rows with concurrency={max_concurrency}...")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = aiohttp.ClientTimeout(total=request_timeout)
    endpoint = base_endpoint.rstrip('/') + '/' + route.lstrip('/')

    semaphore = asyncio.Semaphore(max_concurrency)

    output_file = open(output_path, 'w', encoding='utf-8', newline='')
    fieldnames = ['service', 'response_starting', 'response_confirmed', 'persona', 'short']
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        tasks = []
        for idx, row in enumerate(rows):
            tasks.append(
                asyncio.create_task(
                    process_row(
                        idx,
                        row,
                        session,
                        endpoint,
                        model,
                        system_prompts,
                        semaphore,
                        max_attempts,
                    )
                )
            )

        progress = tqdm(total=total_rows, desc="Generating responses")
        try:
            for coro in asyncio.as_completed(tasks):
                idx, output_row = await coro
                writer.writerow(output_row)
                output_file.flush()
                progress.update(1)
        finally:
            progress.close()

    output_file.close()
        
    print(f"✓ Successfully processed {total_rows} rows")
    print(f"✓ Output saved to: {output_path}")


async def process_row(
    idx: int,
    row: Dict[str, str],
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    system_prompts: Dict[str, str],
    semaphore: asyncio.Semaphore,
    max_attempts: int,
) -> Tuple[int, Dict[str, str]]:
    """Process a single CSV row asynchronously."""

    service = row['service']
    original_response = row['response']
    persona = row['persona']
    short = row['short']

    system_prompt = system_prompts.get(persona, system_prompts.get('assistant', ''))

    reformatted = await generate_reformatted_responses(
        session=session,
        url=endpoint,
        model=model,
        system_prompt=system_prompt,
        original_response=original_response,
        service=service,
        semaphore=semaphore,
        max_attempts=max_attempts,
    )

    output_row = {
        'service': service,
        'response_starting': reformatted['response_starting'],
        'response_confirmed': reformatted['response_confirmed'],
        'persona': persona,
        'short': short,
    }

    return idx, output_row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reformat CSV responses using llama.cpp chat completions with concurrent aiohttp calls",
    )
    parser.add_argument('input_csv', type=Path, help='Input CSV file path (e.g., pile_of_responses.csv)')
    parser.add_argument('output_csv', type=Path, help='Output CSV file path')
    parser.add_argument(
        '--system-prompts',
        type=Path,
        help='Path to system prompts CSV file (default: same directory as input, pile_of_system_prompts.csv)',
    )
    parser.add_argument(
        '--endpoint',
        type=str,
        default='https://ai.cloud.alexoconnell.net/v1',
        help='Base URL for the llama.cpp server (default: https://ai.cloud.alexoconnell.net)',
    )
    parser.add_argument(
        '--route',
        type=str,
        default='/chat/completions',
        help='Route for the llama.cpp chat completion endpoint (default: /chat/completions)',
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Optional API key for the llama.cpp server (default: use OPENAI_API_KEY env var if set)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-oss-120b',
        help='Model to use (default: gpt-oss-120b)',
    )
    parser.add_argument(
        '--max-concurrency',
        type=int,
        default=4,
        help='Maximum number of concurrent generations (default: 4)',
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=3,
        help='Maximum number of retries per row (default: 3)',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Total request timeout in seconds (default: 60.0)',
    )

    args = parser.parse_args()

    if not args.input_csv.exists():
        print(f"Error: Input file not found: {args.input_csv}")
        return 1

    if args.system_prompts:
        system_prompts_path = args.system_prompts
    else:
        system_prompts_path = args.input_csv.parent / 'pile_of_system_prompts.csv'

    if not system_prompts_path.exists():
        print(f"Error: System prompts file not found: {system_prompts_path}")
        return 1

    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(
            process_csv(
                input_path=args.input_csv,
                output_path=args.output_csv,
                system_prompts_path=system_prompts_path,
                base_endpoint=args.endpoint,
                route=args.route,
                api_key=api_key,
                model=args.model,
                max_concurrency=args.max_concurrency,
                max_attempts=args.max_attempts,
                request_timeout=args.timeout,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
