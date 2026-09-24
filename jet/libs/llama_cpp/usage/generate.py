import os

from jet.logger import logger
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1"),
    api_key="sk-1234",
)

response = client.completions.create(
    model=os.getenv("LLAMA_CPP_LLM_MODEL", "not-needed"),
    prompt="What is OpenTelemetry? Answer in exactly one short sentence. Be concise.\nAnswer:",
    max_tokens=128,
    temperature=0.7,
    top_p=0.95,
    presence_penalty=1.5,
)
logger.teal(response.choices[0].text)
