import asyncio

from browser_use import Agent
from jet.adapters.browser_use.ollama.chat import ChatOllama


async def main():
    task = 'Find the founders of browser-use'
    agent = Agent(task=task, llm=ChatOllama(model='llama3.2'))
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
