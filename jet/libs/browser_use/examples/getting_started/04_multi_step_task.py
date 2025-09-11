"""
Getting Started Example 4: Multi-Step Task

This example demonstrates how to:
- Perform a complex workflow with multiple steps
- Navigate between different pages
- Combine search, form filling, and data extraction
- Handle a realistic end-to-end scenario

This is the most advanced getting started example, combining all previous concepts.
"""

import asyncio
import os
import shutil
import sys

from dotenv import load_dotenv
from datetime import datetime
from browser_use import Agent, BrowserProfile

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

    # Define a multi-step task
    task = """
    I want you to research Python web scraping libraries. Here's what I need:
    
    1. First, search Google for "best Python web scraping libraries 2024"
    2. Find a reputable article or blog post about this topic
    3. From that article, extract the top 3 recommended libraries
    4. For each library, visit its official website or GitHub page
    5. Extract key information about each library:
       - Name
       - Brief description
       - Main features or advantages
       - GitHub stars (if available)
    
    Present your findings in a summary format comparing the three libraries.
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
