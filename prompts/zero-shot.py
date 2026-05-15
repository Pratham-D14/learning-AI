from openai import OpenAI

client = OpenAI(
    api_key="AIzaSyAeuwv3_VJ2YhRfwJNvneacn9UMyICGI6c",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# ZERO SHOT PROMPTING: Directly giving the instruction to the model.
# The model is given a direct question or task without prior examples.

SYSTEM_PROMPT = "You are expert in mathematics and you are only allowed to give answers related to mathematics and if someone asked you anything else simply write a sorry message."
USER_PROMPT = ""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
)

result = response.choices[0].message.content
print(result)
