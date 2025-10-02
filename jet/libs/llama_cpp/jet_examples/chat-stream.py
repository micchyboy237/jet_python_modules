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
    model="ggml-org/gemma-3-4b-it-GGUF",
    messages=messages,
    stream=True,
)
for part in stream:
    logger.teal(part.choices[0].delta.content or "", flush=True)
