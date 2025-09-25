import asyncio
from typing import List, Dict, Any
from ollama import AsyncClient

async def chat_stream(model: str, messages: List[Dict[str, Any]]) -> None:
    """
    Stream chat responses from the specified model asynchronously.

    Args:
        model (str): The name of the model to use (e.g., 'gemma3').
        messages (List[Dict[str, Any]]): List of message dictionaries with role and content.
    """
    client = AsyncClient()
    async for chunk in await client.chat(model, messages=messages, stream=True):
        content = chunk.get('message', {}).get('content', '')
        if content:
            print(content, end='', flush=True)  # Print chunks without newline
    print()  # Newline at the end for clean output

async def main():
    messages = [
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        },
    ]
    await chat_stream('gemma3:4b-it-q4_K_M', messages)

if __name__ == '__main__':
    asyncio.run(main())
