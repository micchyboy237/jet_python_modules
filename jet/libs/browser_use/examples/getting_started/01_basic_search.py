from jet.adapters.browser_use.ollama.chat import ChatOllama
from browser_use import Agent
from dotenv import load_dotenv
import asyncio
import os
import sys

# Add the parent directory to the path so we can import browser_use
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


load_dotenv()


async def main():
    llm = ChatOllama(model='llama3.2')
    task = "Search Google for 'what is browser automation' and tell me the top 3 results"
    agent = Agent(task=task, llm=llm)
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
