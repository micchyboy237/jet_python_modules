from openai import OpenAI

from jet.logger import logger

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")  # Dummy API key
stream = client.completions.create(
    model="qwen3-instruct-2507:4b",
    prompt="Why is the sky blue?",
    stream=True,
)
for part in stream:
    logger.teal(part.choices[0].text or "", end="", flush=True)