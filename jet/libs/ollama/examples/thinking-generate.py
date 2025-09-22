from ollama import Client

from jet.logger import logger
from jet.transformers.formatters import format_json

client = Client(host='http://localhost:11435')

response_stream = client.generate('deepseek-r1', 'why is the sky blue', think=True, stream=True)

thinking = ''
content = ''
final = True

for chunk in response_stream:
    if chunk.response:
        content += chunk.response
        if not (chunk.thinking or chunk.thinking == '') and final:
            print('\n\n' + '=' * 10)
            print('Final result: ')
            final = False
        logger.success(chunk.response, flush=True)

    if chunk.thinking:
        if not thinking:
            print('\n\n' + '=' * 10)
            print('Thinking: ')
        # accumulate thinking
        thinking += chunk.thinking
        logger.debug(chunk.thinking, flush=True)

logger.gray("\n\nOutput Info:")
logger.success(format_json(chunk))
