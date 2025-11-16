from openai import OpenAI
from jet.logger import logger

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")  # Dummy API key
response = client.completions.create(
    model="qwen3-instruct-2507:4b",
    prompt="Why is the sky blue?",
)
logger.teal(response.choices[0].text)