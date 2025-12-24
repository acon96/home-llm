import argparse
import asyncio
import csv
import json
import random
import string
import aiohttp
from tqdm import tqdm
import os

from utils import get_dataset_piles
from devices import random_device_list

LLM_ENDPOINT = "https://ai.cloud.alexoconnell.net/v1/chat/completions"
cwd = os.path.dirname(os.path.abspath(__file__))

def get_hass_match_error_message(bad_device_name: str) -> str:
    return f"<MatchFailedError result=MatchTargetsResult(is_match=False, no_match_reason=<MatchFailedReason.NAME: 1>, states=[], no_match_name=None, areas=[], floors=[]), constraints=MatchTargetsConstraints(name='{bad_device_name}', area_name=None, floor_name=None, domains=None, device_classes=None, features=None, states=None, assistant='conversation', allow_duplicate_names=False, single_target=False), preferences=MatchTargetsPreferences(area_id=None, floor_id=None)>"

class SyntheticDataGenerator:
    def __init__(self, model_name: str, language: str, concurrency: int):
        self.language = language
        self.concurrency = concurrency
        self.model_name = model_name
        self.piles = get_dataset_piles(language)
        self.synthetic_devices = {} # device_type -> list of {device_name, description}
        self.failed_tool_fieldnames = [
            "service_name",
            "correct_device_name",
            "correct_friendly_name",
            "bad_device_name",
            "phrase",
            "error_result",
            "retry_prompt"
        ]
        self.refusal_fieldnames = [
            "reason_type",
            "service_name",
            "device_name",
            "friendly_name",
            "desired_state",
            "phrase",
            "response"
        ]

    async def _chat_completion(self, session, system_prompt: str, user_prompt: str, *, temperature: float = 0.8, max_tokens: int = 300, structured_response: dict = {}):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens if not structured_response else None, # don't limit if structured (causes failed generations)
            "response_format": structured_response if structured_response else None,
        }

        try:
            async with session.post(LLM_ENDPOINT, json=payload) as response:
                if response.status < 400:
                    data = await response.json()
                    return data['choices'][0]['message']['content'].strip()
                print(f"LLM request failed with status {response.status}")
        except Exception as exc:
            print(f"Completion request failed: {exc}")
        return None

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        if not content:
            return ""
        text = content.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            return "\n".join(lines).strip()
        return text

    def _parse_json_object(self, content: str):
        if not content:
            return None
        cleaned = self._strip_code_fence(content)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            return None
        snippet = cleaned[start:end+1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _ensure_placeholder(text: str, friendly_name: str, placeholder: str) -> str:
        if placeholder in text:
            return text
        lower_text = text.lower()
        lower_name = friendly_name.lower()
        idx = lower_text.find(lower_name)
        if idx == -1:
            return text
        return text[:idx] + placeholder + text[idx + len(friendly_name):]

    @staticmethod
    def _describe_service(service_name: str, service_data: dict) -> str:
        domain, action = service_name.split(".", 1)
        description = f"{action.replace('_', ' ')} the {domain}"
        if service_data:
            description += f" with arguments {json.dumps(service_data, ensure_ascii=False)}"
        return description

    @staticmethod
    def _desired_state_for_service(service_name: str) -> str:
        action = service_name.split(".", 1)[1]
        action_mapping = {
            "turn_on": "on",
            "turn_off": "off",
            "open_cover": "open",
            "close_cover": "closed",
            "lock": "locked",
            "unlock": "unlocked",
            "media_play_pause": "playing",
        }
        if action in action_mapping:
            return action_mapping[action]
        service_mapping = {
            "media_player.turn_on": "on",
            "media_player.turn_off": "off",
            "vacuum.start": "cleaning",
            "vacuum.return_to_base": "docked",
            "garage_door.open_cover": "open",
            "garage_door.close_cover": "closed",
            "blinds.open_cover": "open",
            "blinds.close_cover": "closed",
        }
        return service_mapping.get(service_name, "")

    @staticmethod
    def _append_rows_to_csv(path: str, fieldnames, rows):
        if not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_exists = os.path.exists(path)
        if not file_exists:
            with open(path, "w", newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        with open(path, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row in rows:
                writer.writerow(row)
        
    async def generate_device_names(self, session, device_type, count=10):
        """
        Generates a list of new device names for a given type.
        """
        system_prompt = "You are a creative assistant that generates realistic smart home device names."
        user_prompt = f"Generate {count} realistic and diverse friendly names for a smart home device of type '{device_type}' (e.g. 'Kitchen Light', 'Porch Fan', 'Master Bedroom Blinds').\n" \
                      f"Output ONLY the names, one per line. Do not number them. Do not include the device type if it's not part of the natural name."

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 1.2,
            "max_tokens": 200,
        }

        try:
            async with session.post(LLM_ENDPOINT, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content'].strip()
                    names = [line.strip() for line in content.split('\n') if line.strip()]
                    
                    # Add to synthetic devices
                    if device_type not in self.synthetic_devices:
                        self.synthetic_devices[device_type] = []
                    
                    new_devices = []
                    for name in names:
                        # Create a fake entity ID
                        slug = name.lower().replace(" ", "_").replace("'", "")
                        entity_id = f"{device_type}.{slug}"
                        device_entry = {
                            "device_name": entity_id,
                            "description": name
                        }
                        self.synthetic_devices[device_type].append(device_entry)
                        new_devices.append(device_entry)

                    print(f"Generated {len(names)} new names for {device_type}")
                    return new_devices
                else:
                    print(f"Failed to generate device names: {response.status}")
                    return []
        except Exception as e:
            print(f"Device generation failed: {e}")
            return []

    async def generate_phrase(self, session, context):
        """
        Generates a user phrase for a given context (device, service, args).
        """
        task_type = context.get("type", "action")
        device_name = context["device_name"]
        friendly_name = context["friendly_name"]
        
        system_prompt = "You are a helpful assistant that generates synthetic training data for a smart home voice assistant. " \
                        "Your goal is to generate diverse, natural, and realistic user commands based on a specific action. " \
                        "The commands should vary in complexity and phrasing."
        
        if task_type == "action":
            service_name = context["service_name"]
            service_args = context["service_data"]
            
            user_prompt = f"""
            Task: Generate a natural language voice command in {self.language} that a user would say to perform the following action.
            
            Target Device: {friendly_name} (ID: {device_name})
            Action: {service_name}
            Arguments: {json.dumps(service_args)}
            
            Instructions:
            1. The command must be in {self.language}.
            2. The command should be natural and conversational.
            3. Do not include the device ID (e.g., {device_name}) in the command, only refer to it by name or context.
            4. Include the necessary information to imply the arguments (e.g., if brightness is 50%, mention "50%" or "half brightness").
            5. Provide ONLY the command text. Do not add quotes or explanations.
            """
        elif task_type == "status":
            attribute = context["attribute"]
            user_prompt = f"""
            Task: Generate a natural language question in {self.language} that a user would ask to check the status of a device.
            
            Target Device: {friendly_name} (ID: {device_name})
            Attribute to check: {attribute}
            
            Instructions:
            1. The question must be in {self.language}.
            2. The question should be natural and conversational.
            3. Do not include the device ID.
            4. Provide ONLY the question text. Do not add quotes or explanations.
            """
        else:
            # Fallback for unknown task types
            user_prompt = "Generate a random smart home command."
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 1.0, # High temperature for diversity
            "max_tokens": 60,
            "stream": False
        }
        
        try:
            async with session.post(LLM_ENDPOINT, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content'].strip()
                    # Cleanup: remove leading/trailing quotes if present
                    if content.startswith('"') and content.endswith('"'):
                        content = content[1:-1]
                    return content
                else:
                    # print(f"Error from LLM: {response.status}")
                    return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    async def generate_failed_tool_call_entry(self, session, context):
        service_name = context["service_name"]
        friendly_name = context["friendly_name"]
        correct_device = context["device_name"]
        action_description = self._describe_service(service_name, context.get("service_data", {}))
        device_list, device_types, extra_exposed_attributes = random_device_list(max_devices=10, avoid_device_names=[], language=self.language)

        extra_phrase_instructions = ""
        if service_name == "climate.set_temperature":
            extra_phrase_instructions = "Always reference the target temperature using the literal placeholder <temp_f>."

        system_prompt = "You create high-quality multilingual training data for a smart home assistant."
        user_prompt = f"""
Generate content for a failed tool call in {self.language}.
Service: {service_name}
Action description: {action_description}
Device name: {correct_device} (friendly name: {friendly_name})
The assistant first misidentifies the entity using the "bad_device" name, receives an error, then retries with {correct_device}.
Output a single JSON object with the keys "phrase", "bad_device", "error_result", and "retry_prompt".
- phrase: A natural voice command template the user would say. {extra_phrase_instructions}
- bad_device: The incorrect version of '{correct_device}'. Ensure it is a truncation, mutation, transposition, or inclusion of an extra suffix/prefix. Avoid simple typos in words.
- retry_prompt: A short acknowledgement from the assistant that it will try the correct device ({correct_device}).

Here are potential device names to help generate a bad device name: {', '.join(device_list)}
Keep the tone conversational and stay entirely in {self.language}. Do not add explanations or code fences.
"""

        raw = await self._chat_completion(
            session,
            system_prompt,
            user_prompt,
            temperature=0.7,
            max_tokens=512,
            structured_response={ "type": "json_object" }
        )
        parsed = self._parse_json_object(raw)
        if not parsed:
            print(f"Failed to parse JSON for failed tool call generation: {raw}")
            return None

        phrase = parsed.get("phrase", "").strip()
        bad_device = parsed.get("bad_device", "").strip()
        retry_prompt = parsed.get("retry_prompt", "").strip()
        if not phrase or not bad_device or not retry_prompt:
            return None

        return {
            "service_name": service_name,
            "correct_device_name": correct_device,
            "correct_friendly_name": friendly_name,
            "bad_device_name": bad_device,
            "phrase": phrase,
            "error_result": get_hass_match_error_message(bad_device),
            "retry_prompt": retry_prompt
        }

    async def generate_refusal_entry(self, session, context, reason_type: str, desired_state: str):
        service_name = context["service_name"]
        friendly_name = context["friendly_name"]
        action_description = self._describe_service(service_name, context.get("service_data", {}))
        device_suffix = context["device_name"].split(".", 1)[1]

        if reason_type == "not_available":
            reason_detail = "Explain that the assistant cannot locate that device in the home."
        else:
            reason_detail = f"Explain that the device is already {desired_state} and no change is needed."

        system_prompt = "You write concise refusal-style responses for a multilingual smart home assistant."
        user_prompt = f"""
Create a refusal example in {self.language} for the following request.
Service: {service_name}
Action description: {action_description}
Reason: {reason_detail}
Output a JSON object with:
- "phrase": the user's natural command template. Use the literal placeholder <device_name> anywhere the user mentions the device.
- "response": the assistant's refusal message in {self.language} describing the reason.
Keep both fields brief, conversational, and free of extra narration or code fences.
"""

        raw = await self._chat_completion(
            session,
            system_prompt,
            user_prompt,
            temperature=0.7,
            max_tokens=512,
            structured_response={ "type": "json_object" }
        )
        parsed = self._parse_json_object(raw)
        if not parsed:
            print("Failed to parse JSON for refusal generation")
            return None

        phrase = parsed.get("phrase", "").strip()
        response = parsed.get("response", "").strip()
        if not phrase or not response:
            return None

        phrase = self._ensure_placeholder(phrase, friendly_name, "<device_name>")
        if "<device_name>" not in phrase:
            return None

        return {
            "reason_type": reason_type,
            "service_name": service_name,
            "device_name": device_suffix,
            "friendly_name": friendly_name,
            "desired_state": desired_state if reason_type == "already_state" else "",
            "phrase": phrase,
            "response": response
        }

    def sample_context(self, request_type: str):
        """
        Creates a random scenario: device, service, and arguments.
        """
        # 1. Pick a device from the loaded piles OR synthetic devices
        device_types = list(self.piles.stacks_of_device_names.keys())
        if not device_types:
            return None
            
        dt = random.choice(device_types)
        
        # Mix real and synthetic devices
        devices = self.piles.stacks_of_device_names[dt]
        if dt in self.synthetic_devices:
            devices = devices + self.synthetic_devices[dt]
            
        if not devices:
            return None
            
        device = random.choice(devices)
        device_name = device["device_name"]
        friendly_name = device["description"]
        
        # Decide between Action and Status
        if request_type == "status":
            # Status Request
            # Determine available attributes based on domain
            domain = device_name.split(".")[0]
            attributes = ["state"] # Default
            if domain == "light":
                attributes.extend(["brightness", "color"])
            elif domain == "climate":
                attributes.extend(["temperature", "humidity", "hvac_mode"])
            elif domain == "media_player":
                attributes.extend(["volume", "media_title", "state"])
            elif domain == "cover":
                attributes.extend(["position", "state"])
            elif domain == "fan":
                attributes.extend(["speed", "state"])
            
            attribute = random.choice(attributes)
            return {
                "type": "status",
                "device_name": device_name,
                "friendly_name": friendly_name,
                "attribute": attribute
            }

        elif request_type == "action":
            # Action
            # 2. Pick a service compatible with this device type
            domain = device_name.split(".")[0]
            
            services = []
            if domain == "light":
                services = ["light.turn_on", "light.turn_off", "light.toggle"]
            elif domain == "switch":
                services = ["switch.turn_on", "switch.turn_off", "switch.toggle"]
            elif domain == "cover":
                services = ["cover.open_cover", "cover.close_cover", "cover.stop_cover", "cover.toggle"]
            elif domain == "blinds":
                services = ["blinds.open_cover", "blinds.close_cover", "blinds.stop_cover", "blinds.toggle"]
            elif domain == "garage_door":
                services = ["garage_door.open_cover", "garage_door.close_cover", "garage_door.stop_cover", "garage_door.toggle"]
            elif domain == "fan":
                services = ["fan.turn_on", "fan.turn_off", "fan.toggle", "fan.increase_speed", "fan.decrease_speed"]
            elif domain == "climate":
                services = ["climate.turn_on", "climate.turn_off", "climate.set_temperature"]
            elif domain == "media_player":
                services = ["media_player.turn_on", "media_player.turn_off", "media_player.media_play_pause", "media_player.volume_up", "media_player.volume_down"]
            elif domain == "lock":
                services = ["lock.lock", "lock.unlock"]
            elif domain == "vacuum":
                services = ["vacuum.start", "vacuum.return_to_base", "vacuum.stop"]
            
            if not services:
                return None
                
            service_name = random.choice(services)
            
            # 3. Generate Arguments
            service_data = {}
            if service_name == "light.turn_on":
                if random.random() < 0.3:
                    service_data["brightness_pct"] = random.randint(10, 100)
                if random.random() < 0.3:
                    # Simple colors
                    colors = ["red", "blue", "green", "yellow", "purple", "white", "warm white", "cool white"]
                    service_data["color_name"] = random.choice(colors)
            elif service_name == "climate.set_temperature":
                service_data["temperature"] = random.randint(18, 28)
            
            return {
                "type": "action",
                "device_name": device_name,
                "friendly_name": friendly_name,
                "service_name": service_name,
                "service_data": service_data
            }
        raise ValueError(f"Unknown request type {request_type}")

    async def run(self, num_actions: int, num_status_requests: int, num_devices: int,
                  persona_name=None, persona_description=None, num_failed_tool_calls: int = 0,
                  num_refusals: int = 0):
        print(f"Starting generation...")
        print(f"Language: {self.language}")
        
        async with aiohttp.ClientSession() as session:
            
            if num_devices > 0:
                print("Generating synthetic device names...")
                device_types = list(self.piles.stacks_of_device_names.keys())
                gen_tasks = []
                for dt in device_types:
                    gen_tasks.append(self.generate_device_names(session, dt, count=num_devices))
                
                generated_lists = await asyncio.gather(*gen_tasks)
                
                # Flatten list and write to CSV
                all_new_devices = [item for sublist in generated_lists if sublist for item in sublist]
                
                if all_new_devices:
                    csv_path = f"{cwd}/piles/{self.language}/pile_of_device_names.csv"
                    try:
                        with open(csv_path, "a", newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=["device_name", "description"])
                            for device in all_new_devices:
                                writer.writerow(device)
                        print(f"Appended {len(all_new_devices)} new devices to {csv_path}")
                    except Exception as e:
                        print(f"Failed to write new devices to CSV: {e}")
            
            if num_actions > 0 or num_status_requests > 0:
                print(f"Generating {num_actions} actions and {num_status_requests} status requests...")
                tasks = {}
                results = []
                
                pbar = tqdm(total=num_actions + num_status_requests, desc="Generating phrases")
                
                while len(results) < num_actions + num_status_requests:
                    # Fill up the task queue
                    while len(tasks) < self.concurrency and (len(results) + len(tasks)) < num_actions + num_status_requests:
                        context = self.sample_context("action" if len(results) < num_actions else "status")
                        if not context:
                            continue
                            
                        task = asyncio.create_task(self.generate_phrase(session, context))
                        tasks[task] = context
                    
                    if not tasks:
                        break
                    
                    # Wait for completed tasks
                    done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
                    
                    for task in done:
                        context = tasks.pop(task)
                        try:
                            phrase = await task
                            if phrase:
                                entry = context.copy()
                                entry["phrase"] = phrase
                                
                                if entry["type"] == "action":
                                    # Write to pile_of_specific_actions.csv
                                    csv_path = f"{cwd}/piles/{self.language}/pile_of_specific_actions.csv"
                                    
                                    # Prepare row
                                    # device_name in CSV is the suffix (e.g. 'kitchen' from 'light.kitchen')
                                    # But wait, generate_data.py expects device_name to be the suffix ONLY if the domain matches the service domain?
                                    # Actually generate_data.py does: target_device = f"{device_type}.{action['device_name']}"
                                    # where device_type = service_name.split(".")[0]
                                    # So if service is light.turn_on, device_type is light.
                                    # If device is light.kitchen, action['device_name'] should be 'kitchen'.
                                    
                                    full_device_name = entry["device_name"]
                                    service_name = entry["service_name"]
                                    service_domain = service_name.split(".")[0]
                                    device_domain = full_device_name.split(".")[0]
                                    
                                    if service_domain != device_domain:
                                        # This might happen if we use a service from a different domain (e.g. homeassistant.turn_on)
                                        # But our sample_context ensures domain match (mostly).
                                        # For blinds/garage_door, we use blinds.open_cover etc.
                                        # So service_domain is blinds. device_domain is blinds.
                                        pass
                                    
                                    device_suffix = full_device_name.split(".", 1)[1]
                                    
                                    row = {
                                        "service_name": service_name,
                                        "device_name": device_suffix,
                                        "phrase": phrase,
                                        "arguments": json.dumps(entry["service_data"]) if entry["service_data"] else ""
                                    }
                                    
                                    # Check if header needs update (only once)
                                    if not hasattr(self, "_action_header_updated"):
                                        self._action_header_updated = True
                                        # Read header
                                        with open(csv_path, "r", encoding='utf-8') as f:
                                            reader = csv.DictReader(f)
                                            all_rows = list(reader)
                                            current_fieldnames = reader.fieldnames if reader.fieldnames else []
                                        
                                        fieldnames = list(current_fieldnames) + ["arguments"]
                                        with open(csv_path, "w", newline='', encoding='utf-8') as f:
                                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                                            writer.writeheader()
                                            writer.writerows(all_rows)
                                            
                                    with open(csv_path, "a", newline='', encoding='utf-8') as f:
                                        # We need to know fieldnames.
                                        # We can read them from file or assume standard + arguments.
                                        # Let's read them.
                                        with open(csv_path, "r", encoding='utf-8') as fr:
                                            reader = csv.reader(fr)
                                            header = next(reader)
                                        
                                        writer = csv.DictWriter(f, fieldnames=header)
                                        writer.writerow(row)

                                elif entry["type"] == "status":
                                    # Write to pile_of_status_requests.csv
                                    # We need to templatize the phrase.
                                    # Replace friendly_name with <device_name>
                                    phrase_tmpl = phrase.replace(entry["friendly_name"], "<device_name>")
                                    # Also try case insensitive?
                                    phrase_tmpl = phrase_tmpl.replace(entry["friendly_name"].lower(), "<device_name>")
                                    
                                    # If friendly name not found, maybe skip?
                                    if "<device_name>" not in phrase_tmpl:
                                        # Try to find partial match?
                                        # For now, just skip if we can't templatize.
                                        pass
                                    else:
                                        csv_path = f"{cwd}/piles/{self.language}/pile_of_status_requests.csv"
                                        # Columns: device_type,state,phrase,assistant_response
                                        # We don't have assistant_response.
                                        # We can generate a generic one?
                                        # Or ask LLM to generate it?
                                        # For now, let's skip status requests writing as we lack assistant_response.
                                        pass
                                    
                                results.append(entry)
                                pbar.update(1)
                        except Exception as e:
                            print(f"Task error: {e}")
                
                pbar.close()
            
            if num_failed_tool_calls > 0:
                print(f"Generating {num_failed_tool_calls} failed tool call scenarios...")
                failed_entries = []
                tasks = {}
                pbar_failed = tqdm(total=num_failed_tool_calls, desc="Failed tool calls")

                while len(failed_entries) < num_failed_tool_calls:
                    while len(tasks) < self.concurrency and (len(failed_entries) + len(tasks)) < num_failed_tool_calls:
                        context = self.sample_context("action")
                        if not context:
                            continue
                        task = asyncio.create_task(self.generate_failed_tool_call_entry(session, context))
                        tasks[task] = None

                    if not tasks:
                        break

                    done, _ = await asyncio.wait(list(tasks.keys()), return_when=asyncio.FIRST_COMPLETED)

                    for task in done:
                        tasks.pop(task, None)
                        try:
                            entry = await task
                        except Exception as exc:
                            print(f"Failed tool call task error: {exc}")
                            entry = None
                        if entry:
                            failed_entries.append(entry)
                            pbar_failed.update(1)

                pbar_failed.close()
                failed_path = f"{cwd}/piles/{self.language}/pile_of_failed_tool_calls.csv"
                self._append_rows_to_csv(failed_path, self.failed_tool_fieldnames, failed_entries)

            if num_refusals > 0:
                print(f"Generating {num_refusals} refusal scenarios...")
                refusal_entries = []
                tasks = {}
                pbar_refusals = tqdm(total=num_refusals, desc="Refusals")

                while len(refusal_entries) < num_refusals:
                    while len(tasks) < self.concurrency and (len(refusal_entries) + len(tasks)) < num_refusals:
                        context = self.sample_context("action")
                        if not context:
                            continue
                        reason_type = random.choice(["not_available", "already_state"])
                        desired_state = ""
                        if reason_type == "already_state":
                            desired_state = self._desired_state_for_service(context["service_name"])
                            if not desired_state:
                                reason_type = "not_available"
                        task = asyncio.create_task(self.generate_refusal_entry(
                            session,
                            context,
                            reason_type,
                            desired_state
                        ))
                        tasks[task] = None

                    if not tasks:
                        break

                    done, _ = await asyncio.wait(list(tasks.keys()), return_when=asyncio.FIRST_COMPLETED)

                    for task in done:
                        tasks.pop(task, None)
                        try:
                            entry = await task
                        except Exception as exc:
                            print(f"Refusal generation error: {exc}")
                            entry = None
                        if entry:
                            refusal_entries.append(entry)
                            pbar_refusals.update(1)

                pbar_refusals.close()
                refusal_path = f"{cwd}/piles/{self.language}/pile_of_refusals.csv"
                self._append_rows_to_csv(refusal_path, self.refusal_fieldnames, refusal_entries)
            
            if persona_name and persona_description:
                await self.generate_persona(session, persona_name, persona_description)
                
            print("Generation complete.")

    async def generate_persona(self, session, persona_name, persona_description):
        print(f"Generating new persona: {persona_name}...")
        
        # 1. Generate System Prompt
        sys_prompt_instruction = (
            f"Generate a system prompt for an AI assistant named '{persona_name}' "
            f"who has the following personality: {persona_description}. "
            "The prompt should define the persona's character and instructions. "
            "It should start with 'You are ...'. "
            "Keep it under 50 words."
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert at creating AI system prompts."},
                {"role": "user", "content": sys_prompt_instruction}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
        }
        
        system_prompt_text = ""
        try:
            async with session.post(LLM_ENDPOINT, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    system_prompt_text = data['choices'][0]['message']['content'].strip()
                    if system_prompt_text.startswith('"') and system_prompt_text.endswith('"'):
                        system_prompt_text = system_prompt_text[1:-1]
                    print(f"Generated system prompt: {system_prompt_text}")
                else:
                    print(f"Failed to generate system prompt: {response.status}")
                    return
        except Exception as e:
            print(f"System prompt generation failed: {e}")
            return

        # 2. Get list of services to generate responses for
        responses_csv_path = f"{cwd}/piles/{self.language}/pile_of_responses.csv"
        services = set()
        try:
            with open(responses_csv_path, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    services.add(row["service"])
        except Exception as e:
            print(f"Failed to read responses CSV: {e}")
            return
            
        print(f"Found {len(services)} unique services to generate responses for.")
        
        # 3. Generate responses for each service
        new_responses = []
        
        async def generate_service_responses(svc):
            # We want normal and short responses
            prompt = (
                f"You are acting as '{persona_name}', described as: {persona_description}.\n"
                f"Generate 3 diverse responses confirming that you are performing the action: '{svc}'.\n"
                "Then generate 3 SHORT/CONCISE responses for the same action.\n"
                "Format the output as follows:\n"
                "NORMAL: <response 1>\n"
                "NORMAL: <response 2>\n"
                "NORMAL: <response 3>\n"
                "SHORT: <short response 1>\n"
                "SHORT: <short response 2>\n"
                "SHORT: <short response 3>\n"
                "Do not include any other text."
            )
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": f"You are {persona_name}. {persona_description}"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.8,
                "max_tokens": 300,
            }
            
            try:
                async with session.post(LLM_ENDPOINT, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content'].strip()
                        lines = content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line.startswith("NORMAL:"):
                                text = line.replace("NORMAL:", "").strip()
                                if text:
                                    new_responses.append({
                                        "service": svc,
                                        "response": text,
                                        "persona": persona_name,
                                        "short": 0
                                    })
                            elif line.startswith("SHORT:"):
                                text = line.replace("SHORT:", "").strip()
                                if text:
                                    new_responses.append({
                                        "service": svc,
                                        "response": text,
                                        "persona": persona_name,
                                        "short": 1
                                    })
            except Exception as e:
                print(f"Failed to generate responses for {svc}: {e}")

        # Run in batches
        tasks = []
        for svc in services:
            tasks.append(generate_service_responses(svc))
            if len(tasks) >= self.concurrency:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)
            
        print(f"Generated {len(new_responses)} responses.")
        
        # 4. Write to files
        # Append system prompt
        sys_prompts_path = f"{cwd}/piles/{self.language}/pile_of_system_prompts.csv"
        try:
            with open(sys_prompts_path, "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Check if we need to add a newline if file doesn't end with one? 
                # csv module handles newlines usually.
                writer.writerow([persona_name, system_prompt_text])
            print(f"Appended system prompt to {sys_prompts_path}")
        except Exception as e:
            print(f"Failed to write system prompt: {e}")

        # Append responses
        try:
            with open(responses_csv_path, "a", newline='', encoding='utf-8') as f:
                fieldnames = ["service", "response", "persona", "short"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                for resp in new_responses:
                    writer.writerow(resp)
            print(f"Appended responses to {responses_csv_path}")
        except Exception as e:
            print(f"Failed to write responses: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data using LLM")
    parser.add_argument("--actions", type=int, default=0, help="Number of actions to generate")
    parser.add_argument("--status", type=int, default=0, help="Number of status requests to generate")
    parser.add_argument("--devices", type=int, default=0, help="Number of new devices to generate")
    parser.add_argument("--failed-tool-calls", type=int, default=0, help="Number of failed tool call scenarios to generate")
    parser.add_argument("--refusals", type=int, default=0, help="Number of refusal scenarios to generate")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent requests")
    parser.add_argument("--languages", type=str, nargs='+', default=["english"], help="Languages to generate data for")
    parser.add_argument("--model", type=str, default="gpt-oss-120b", help="LLM model to use")
    parser.add_argument("--persona-name", type=str, help="Name of the new persona to generate")
    parser.add_argument("--persona-description", type=str, help="Description of the new persona")
    
    args = parser.parse_args()
    
    for language in args.languages:
        print(f"=== Generating data for language: {language} ===")
        generator = SyntheticDataGenerator(model_name=args.model, language=language, concurrency=args.concurrency)
        asyncio.run(generator.run(
            num_actions=args.actions,
            num_status_requests=args.status,
            num_devices=args.devices,
            persona_name=args.persona_name,
            persona_description=args.persona_description,
            num_failed_tool_calls=args.failed_tool_calls,
            num_refusals=args.refusals
        ))
