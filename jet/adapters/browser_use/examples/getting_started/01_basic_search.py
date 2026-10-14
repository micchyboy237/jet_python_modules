import asyncio
import os
import sys

# Add the parent directory to the path so we can import browser_use
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dotenv import load_dotenv

load_dotenv()

from browser_use import Agent
from jet.adapters.browser_use.chat_llama_cpp import ChatLlamaCpp
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL


async def main():
    llm = ChatLlamaCpp(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    task = (
        "Search Google for 'what is browser automation' and tell me the top 3 results"
    )
    agent = Agent(task=task, llm=llm)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
