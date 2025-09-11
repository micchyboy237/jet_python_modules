"""
Getting Started Example 3: Data Extraction

This example demonstrates how to:
- Navigate to a website with structured data
- Extract specific information from the page
- Process and organize the extracted data
- Return structured results

This builds on previous examples by showing how to get valuable data from websites.
"""

import asyncio
import os
import shutil
import sys

from dotenv import load_dotenv

from browser_use import Agent, BrowserProfile
from datetime import datetime

from jet.adapters.browser_use.custom_agent import CustomAgent
from jet.adapters.browser_use.ollama.chat import ChatOllama
from jet.logger import logger

# # Add the parent directory to the path so we can import browser_use
# sys.path.append(os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
log_file = os.path.join(
    OUTPUT_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger.basicConfig(filename=log_file)
logger.orange(f"Logs: {log_file}")


async def main():
    # Initialize the model
    llm = ChatOllama(model='llama3.2')

    # Define a data extraction task
    task = """
    Go to https://quotes.toscrape.com/ and extract the following information:
    - The first 5 quotes on the page
    - The author of each quote
    - The tags associated with each quote
    
    Present the information in a clear, structured format like:
    Quote 1: "[quote text]" - Author: [author name] - Tags: [tag1, tag2, ...]
    Quote 2: "[quote text]" - Author: [author name] - Tags: [tag1, tag2, ...]
    etc.
    """

    browser_profile = BrowserProfile(
        minimum_wait_page_load_time=0.1,
        wait_between_actions=0.1,
        headless=True,
    )
    # Create and run the agent
    agent = CustomAgent(task=task, llm=llm, browser_profile=browser_profile)
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
