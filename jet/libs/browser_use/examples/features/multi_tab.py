"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
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


# video: https://preview.screen.studio/share/clenCmS6
llm = ChatOllama(model='llama3.2')
agent = Agent(
    task='open 3 tabs with elon musk, sam altman, and steve jobs, then go back to the first and stop',
    llm=llm,
)


async def main():
    await agent.run()


asyncio.run(main())
