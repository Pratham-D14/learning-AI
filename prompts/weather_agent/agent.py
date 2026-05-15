# Chain of Thought Prompting
import json
import re
import requests
from pydantic import BaseModel, Field
from typing import Optional

url = "http://localhost:11434/api/chat"


def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The Weather in {city} is {response.text}"


available_tools = {"get_weather": get_weather}


SYSTEM_PROMPT = """You are an AI assistant that solves problems using step-by-step chain-of-thought reasoning.

==== CRITICAL: JSON OUTPUT ONLY ====
- You MUST respond with ONLY a single valid JSON object
- NO explanations outside the JSON
- NO markdown (no ```)
- NO multiple JSON objects
- ONE step per response

==== OUTPUT FORMAT ====
Every response must be exactly ONE of these JSON objects:

For planning steps:
{"step": "PLAN", "content": "your reasoning here"}

For tool calls:
{"step": "TOOL", "content": "why calling tool", "tool": "tool_name", "input": "tool_input"}

For final answer:
{"step": "OUTPUT", "content": "final answer to user"}

==== AVAILABLE TOOLS ====
- get_weather: Input is a city name (string). Returns weather information for that city.

==== WORKFLOW ====
1. Start with PLAN steps to think through the problem
2. If you need a tool, use TOOL step (you'll receive tool output in next message)
3. Continue with more PLAN steps if needed
4. End with OUTPUT step when ready to answer the user

==== EXAMPLES ====

Example 1 - Math Problem:
User: "What is 2 + 3 * 5?"

Your response 1:
{"step": "PLAN", "content": "User wants to solve 2 + 3 * 5. I need to use order of operations (PEMDAS)."}

Your response 2:
{"step": "PLAN", "content": "First, multiply: 3 * 5 = 15"}

Your response 3:
{"step": "PLAN", "content": "Then add: 2 + 15 = 17"}

Your response 4:
{"step": "OUTPUT", "content": "The answer is 17."}

---

Example 2 - Weather with Tool:
User: "What's the weather in Mumbai?"

Your response 1:
{"step": "PLAN", "content": "User wants weather info for Mumbai. I should use the get_weather tool."}

Your response 2:
{"step": "TOOL", "content": "Calling weather tool for Mumbai", "tool": "get_weather", "input": "Mumbai"}

System provides tool result:
{"role": "system", "content": "Tool result: Temperature is 28°C, cloudy skies"}

Your response 3:
{"step": "PLAN", "content": "Received weather data: 28°C and cloudy in Mumbai."}

Your response 4:
{"step": "OUTPUT", "content": "The current weather in Mumbai is 28°C with cloudy skies."}

==== REMEMBER ====
- ONE JSON object per response
- NO extra text
- Follow the exact format shown above
"""

USER_PROMPT = "Hey write a code to add n numbers in javascript"

message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]


# Structure Output:
class MyOutputFormat(BaseModel):
    step: str = Field(
        ..., description="The ID of the step. Example: PLAN, OUTPUT, TOOL, etc"
    )
    content: Optional[str] = Field(
        None, description="The optional string content for the step"
    )
    tool: Optional[str] = Field(None, description="The ID of the tool to call.")
    input: Optional[str] = Field(None, description="The input params for the tool")


# Use MyOutputFormat in response format

user_query = input("👉 ")
message_history.append({"role": "user", "content": user_query})

while True:
    payload = {
        "model": "llama3.2:3b",
        "response_format": "json",
        "messages": message_history,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Lower temperature for more consistent JSON
            "num_predict": 150,  # Limit response length
        },
    }

    response = requests.post(url, json=payload)
    raw_result = response.json()["message"]["content"]

    # print("Raw response:", raw_result)
    # print("-" * 50)

    try:
        # Use regex to find all JSON objects
        # This pattern matches { ... } where ... can contain nested braces in strings
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, raw_result)

        if not matches:
            print("❌ No valid JSON objects found in response")
            print(f"Raw result: {raw_result}")
            break

        json_objects = []
        for match in matches:
            try:
                json_obj = json.loads(match)
                json_objects.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse JSON object: {match}")
                print(f"Error: {e}")
                continue

        if not json_objects:
            print("❌ No valid JSON objects could be parsed")
            break

        # Process each JSON object
        for json_obj in json_objects:
            # Add each step to message history
            message_history.append(
                {"role": "assistant", "content": json.dumps(json_obj)}
            )

            if json_obj["step"] == "START":
                print("🔥", json_obj["content"])

            elif json_obj["step"] == "PLAN":
                print("🧠", json_obj["content"])

            elif json_obj["step"] == "TOOL":
                print(f"⚒️: {json_obj["tool"]}")
                tool_response = available_tools[json_obj["tool"]](json_obj["input"])
                print(f"⚒️: {json_obj['tool']} ({json_obj['input']}) = {tool_response}")
                message_history.append(
                    {
                        "role": "developer",
                        "content": json.dumps(
                            {
                                "step": "OBSERVE",
                                "tool": json_obj["tool"],
                                "output": tool_response,
                            }
                        ),
                    }
                )
                continue

            elif json_obj["step"] == "OUTPUT":
                print("🤖", json_obj["content"])
                # Exit the while loop when OUTPUT is reached
                break

        # If we processed an OUTPUT step, break the outer loop
        if any(obj.get("step") == "OUTPUT" for obj in json_objects):
            break

    except Exception as e:
        print("❌ Unexpected error occurred")
        print(f"Error: {e}")
        print(f"Raw result: {raw_result}")
        break
