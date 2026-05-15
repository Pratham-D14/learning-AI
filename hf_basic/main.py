from transformers import pipeline

pipe = pipeline("image-text-to-text", model="google/medgemma-4b-it")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
            },
            {"type": "text", "text": "What animal is on the candy?"},
        ],
    },
]

pipe(text=messages)
print(messages)
