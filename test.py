from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {"role": "system", "content": ""},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://m.media-amazon.com/images/I/51IvjrbjBiL._SY879_.jpg"
                }
            },
            {
                "type": "text",
                "text": (
                    "what is this image"
                )
            }
        ]
    }
]

response = client.chat.completions.create(
    model="tencent/HunyuanOCR",
    messages=messages,
    temperature=0.0,
    extra_body={
        "top_k": 1,
        "repetition_penalty": 1.0
    },
)
print(f"Generated text: {response.choices[0].message.content}")