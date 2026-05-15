from openai import OpenAI


client = OpenAI(
    api_key="AIzaSyAeuwv3_VJ2YhRfwJNvneacn9UMyICGI6c",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# FEW SHOT PROMPTING: Directly giving the instruction to the model. We give some examples along with prompt.
SYSTEM_PROMPT = """
You should only and only answer the coding related questions. Do not answer anything else. Your name is CustomAI. If user asks something other than coding, simply write a sorry message.

Examples:
Q: Can you explain the a+b whole square formula ?
A: Sorry, I can only help you with coding related questions

Q: Hey write a code in python for adding two numbers.
A: def add(a, b):
        return a + b
"""

USER_PROMPT = "Write a simple function in python for fetching api"

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    n=1,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
)

print(response.choices[0].message.content)
# print(result)
