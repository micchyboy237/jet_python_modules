from openai import OpenAI

from jet.logger import logger

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")  # Dummy API key
messages = [
    {
        "role": "user",
        "content": "Why is the sky blue?",
    },
]
stream = client.chat.completions.create(
    model="Qwen_Qwen3-4B-Instruct-2507-Q4_K_M",
    messages=messages,
    stream=True,
)
for part in stream:
    if part.choices:
        logger.teal(part.choices[0].delta.content or "", flush=True)
