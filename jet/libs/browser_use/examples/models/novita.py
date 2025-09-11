"""
Simple try of the agent.

@dev You need to add NOVITA_API_KEY to your environment variables.
"""

from jet.adapters.browser_use.ollama.chat import ChatOllama
from browser_use import Agent
from dotenv import load_dotenv
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


load_dotenv()


api_key = os.getenv('NOVITA_API_KEY', '')
if not api_key:
    raise ValueError('NOVITA_API_KEY is not set')


async def run_search():
    agent = Agent(
        task=(
            '1. Go to https://www.reddit.com/r/LocalLLaMA '
            "2. Search for 'browser use' in the search bar"
            '3. Click on first result'
            '4. Return the first comment'
        ),
        llm=ChatOllama(
            base_url='https://api.novita.ai/v3/openai',
            model='deepseek/deepseek-v3-0324',
            api_key=api_key,
        ),
        use_vision=False,
    )

    await agent.run()


if __name__ == '__main__':
    asyncio.run(run_search())
