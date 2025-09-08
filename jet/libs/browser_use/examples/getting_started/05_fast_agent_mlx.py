# jet_python_modules/jet/libs/browser_use/examples/getting_started/05_fast_agent.py
import shutil
from datetime import datetime
from jet.adapters.browser_use.mlx.chat import ChatMLX
from browser_use import Agent, BrowserProfile
from dotenv import load_dotenv
from jet.logger import logger
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
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

SPEED_OPTIMIZATION_PROMPT = """
Speed optimization instructions:
- Be extremely concise and direct in your responses
- Get to the goal as quickly as possible
- Use multi-action sequences whenever possible to reduce steps
"""


async def main():
    llm = ChatMLX()
    browser_profile = BrowserProfile(
        minimum_wait_page_load_time=0.1,
        wait_between_actions=0.1,
        headless=True,
    )
    task = """
	1. Go to reddit https://www.reddit.com/search/?q=browser+agent&type=communities 
	2. Click directly on the first 5 communities to open each in new tabs
    3. Find out what the latest post is about, and switch directly to the next tab
	4. Return the latest post summary for each page
	"""
    agent = Agent(
        task=task,
        llm=llm,
        flash_mode=True,
        browser_profile=browser_profile,
        extend_system_message=SPEED_OPTIMIZATION_PROMPT,
    )
    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
