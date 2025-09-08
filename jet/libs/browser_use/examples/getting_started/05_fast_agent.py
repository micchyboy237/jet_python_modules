# jet_python_modules/jet/libs/browser_use/examples/getting_started/05_fast_agent.py
import shutil
from datetime import datetime
from jet.adapters.browser_use.custom_agent import CustomAgent
from jet.adapters.browser_use.ollama.chat import ChatOllama
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
    llm = ChatOllama(
        model='llama3.2',
        ollama_options={
            "temperature": 0.0
        }
    )
    browser_profile = BrowserProfile(
        minimum_wait_page_load_time=0.1,
        wait_between_actions=0.1,
        headless=True,
    )
    task = """
    1. Go to https://anilist.co
    2. Search for upcoming isekai anime.
    3. Find the first 5 results from the search.
    4. For each anime, open its detail page in a new tab.
    5. For each, extract the title, release date, and a short synopsis.
    6. Return a summary table of these 5 upcoming isekai anime.
    """
    agent = CustomAgent(
        custom_screenshot_dir=OUTPUT_DIR,
        task=task,
        llm=llm,
        flash_mode=True,
        browser_profile=browser_profile,
        extend_system_message=SPEED_OPTIMIZATION_PROMPT,
    )
    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
