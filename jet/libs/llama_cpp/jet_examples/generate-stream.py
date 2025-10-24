from openai import OpenAI

from jet.logger import logger

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")  # Dummy API key
stream = client.completions.create(
    model="Qwen_Qwen3-4B-Instruct-2507-Q4_K_M",
    prompt="Why is the sky blue?",
    stream=True,
)
for part in stream:
    logger.teal(part.choices[0].text or "", end="", flush=True)