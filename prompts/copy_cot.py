# Chain of Thought Prompting
import json
import re
import requests
from pydantic import BaseModel, Field
from typing import Optional

url = "http://localhost:11434/api/chat"


SYSTEM_PROMPT = """
    You're an expert AI Assistant in resolving user queries using chain of thought.
    You work on START, PLAN and OUTPUT steps.
    You need to first PLAN what needs to be done. The PLAN can be multiple steps.
    Once you think enough PLAN has been done, finally you can give an OUTPUT.
    
    
    Rules: 
    - Strictly Follow the given JSON output format
    - Only run one step at a time.
    - The sequence of steps is START (where user gives an input), PLAN (That can be multiple times) and finally OUTPUT (Which is going to the displayed to the user).
    
    Output format:
    You are a JSON-only response engine.

    You MUST reply with valid JSON.
    You MUST NOT include any text outside JSON.
    You MUST NOT include explanations, markdown, or comments.

    Allowed JSON schema:
        {
        "step": "START" | "PLAN" | "OUTPUT",
        "content": "string"
        }

    If you cannot comply, still return valid JSON using:
        {
        "step": "OUTPUT",
        "content": "ERROR"
        }

    
    Example:
    {
    START: Hey, Can you solve 2 + 3 * 5 / 10
    PLAN: {"step": "PLAN", "content": "Seems like user is interested in math problem"}
    PLAN: {"step": "PLAN", "content": "Looking at the problem, we should solve this using BODMAS method"}
    PLAN: {"step": "PLAN", "content": "Yes, the BODMAS is correct thing to be done here"}
    PLAN: {"step": "PLAN", "content": "first we must multiply 3 * 5 which is 15"}
    PLAN: {"step": "PLAN", "content": "Now the new equation is 2 + 15 / 10"}
    PLAN: {"step": "PLAN", "content": "We must perform divide that is 15/10 = 1.5"}
    PLAN: {"step": "PLAN", "content": "Now the new equation is 2 + 1.5"}
    PLAN: {"step": "PLAN", "content": "Now finally lets perform the addition which is 2 + 1.5 = 3.5"}
    PLAN: {"step": "PLAN", "content": "Great! we have solved and left with 3.5 as answer"}
    OUTPUT: {"step": "OUTPUT", "content": "The addition of 2 + 1.5 is '3.5' "}
    }
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
