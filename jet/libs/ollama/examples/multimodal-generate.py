import random
import sys

import httpx

from ollama import Client

from jet.logger import logger
from jet.transformers.formatters import format_json

client = Client(host='http://localhost:11435')

latest = httpx.get('https://xkcd.com/info.0.json')
latest.raise_for_status()

num = int(sys.argv[1]) if len(sys.argv) > 1 else random.randint(1, latest.json().get('num'))

comic = httpx.get(f'https://xkcd.com/{num}/info.0.json')
comic.raise_for_status()

print(f'xkcd #{comic.json().get("num")}: {comic.json().get("alt")}')
print(f'link: https://xkcd.com/{num}')
print('---')

raw = httpx.get(comic.json().get('img'))
raw.raise_for_status()

for response in client.generate('qwen2.5vl:3b-q4_K_M', 'explain this comic:', images=[raw.content], stream=True):
  logger.teal(response['response'], flush=True)

logger.gray("Result:")
# Remove 'context' to reduce logged response
response_dict = response.__dict__
response_dict.pop('context')
logger.success(format_json(response_dict))
