from openai import OpenAI
client = OpenAI(
    api_key="AIzaSyAeuwv3_VJ2YhRfwJNvneacn9UMyICGI6c",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)


response = client.chat.completions.create(
    model="gemini-2.5-flash",
    n=1,
    messages=[
        {"role": "system", "content": "You are expert in mathematics and you are only allowed to give answers related to mathematic and if someone asked you anything else simply write a sorry message."},
        {
            "role": "user",
            "content": "give me the formula to calculate personal loan"
        }
    ]
)

print(response.choices[0].message.content)
