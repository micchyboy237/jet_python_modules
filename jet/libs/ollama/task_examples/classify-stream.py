from ollama import Client

from jet.logger import logger
from jet.transformers.formatters import format_json

client = Client(host='http://localhost:11435')

messages = [
  {
    'role': 'user',
    'content': 'Classify this review: "Great product, fast shipping!"'
  },
]

for part in client.chat(model='gemma3n:e2b-it-q4_K_M', messages=messages, stream=True):
  logger.teal(part['message']['content'], flush=True)

logger.gray("Result:")
logger.success(format_json(part))
