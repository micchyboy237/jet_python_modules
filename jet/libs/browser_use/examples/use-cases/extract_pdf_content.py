#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["browser-use", "mistralai"]
# ///

from jet.adapters.browser_use.ollama.chat import ChatOllama
from browser_use import Agent
import logging
import asyncio
from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


load_dotenv()


logger = logging.getLogger(__name__)


async def main():
    agent = Agent(
        task="""
        Objective: Navigate to the following UR, what is on page 3?

        URL: https://docs.house.gov/meetings/GO/GO00/20220929/115171/HHRG-117-GO00-20220929-SD010.pdf
        """,
        llm=ChatOllama(model='llama3.2'),
    )
    result = await agent.run()
    logger.info(result)


if __name__ == '__main__':
    asyncio.run(main())
