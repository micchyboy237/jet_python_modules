from typing import Iterator
from ollama import Client
from ollama._types import ChatResponse

from jet.logger import logger

messages = [
    {
        'role': 'user',
        'content': 'What is 10 + 23?',
    },
]

client = Client(
    # Ollama Turbo
    # host="https://ollama.com", headers={'Authorization': (os.getenv('OLLAMA_API_KEY'))}
    host='http://localhost:11435'
)

response_stream: Iterator[ChatResponse] = client.chat(
    model='granite3.3:2b', messages=messages, think=True, stream=True)
tool_calls = []
thinking = ''
content = ''
final = True

for chunk in response_stream:
    if chunk.message.tool_calls:
        tool_calls.extend(chunk.message.tool_calls)

    if chunk.message.content:
        if not (chunk.message.thinking or chunk.message.thinking == '') and final:
            print('\n\n' + '=' * 10)
            print('Final result: ')
            final = False
        logger.success(chunk.message.content, flush=True)

    if chunk.message.thinking:
        # accumulate thinking
        thinking += chunk.message.thinking
        logger.debug(chunk.message.thinking, flush=True)


logger.debug('Thinking:\n========\n\n' + thinking)
logger.success('\nResponse:\n========\n\n' + content)
