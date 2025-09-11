from jet.adapters.browser_use.ollama.chat import ChatOllama
from browser_use import Agent
from dotenv import load_dotenv
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


load_dotenv()


# This uses a bigger model for the planning
# And a smaller model for the page content extraction
# THink of it like a subagent which only task is to extract content from the current page
llm = ChatOllama(model='gpt-4.1')
small_llm = ChatOllama(model='llama3.2')
task = 'Find the founders of browser-use in ycombinator, extract all links and open the links one by one'
agent = Agent(task=task, llm=llm, page_extraction_llm=small_llm)


async def main():
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
