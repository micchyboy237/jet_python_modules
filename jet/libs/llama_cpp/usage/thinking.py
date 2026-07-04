from typing import Iterator
from openai import OpenAI
from jet.logger import logger

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")

messages = [
    {
        'role': 'user',
        'content': 'Think step by step and show your reasoning clearly before giving the final answer.\n\nWhat is 10 + 23?',
    },
]

stream: Iterator = client.chat.completions.create(
    model='deepseek-r1',
    messages=messages,
    stream=True,
    temperature=0,
)

thinking = ''
content = ''
final_printed = False

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        text = delta.content
        content += text
        logger.success(text, flush=True)

        # Detect transition to final answer
        if not final_printed and any(phrase in text.lower() for phrase in ["answer", "final", "so, 10 + 23"]):
            print('\n\n' + '=' * 10)
            print('Final result: ')
            final_printed = True

logger.debug('Thinking:\n========\n\n' + content)  # All output is "thinking" until final
logger.success('\nResponse:\n========\n\n' + content)