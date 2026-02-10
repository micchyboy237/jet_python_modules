import os

from jet.logger import logger
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLAMA_CPP_LLM_URL"), api_key="sk-1234"
)  # Dummy API key
messages = [
    {
        "role": "user",
        "content": "Write a 2 sentence short story",
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
