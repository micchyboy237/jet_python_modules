import asyncio

from browser_use import Agent
from jet.adapters.browser_use.ollama.chat import ChatOllama
from jet.logger import logger
from datetime import datetime
import os
import shutil

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
log_dir = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
filename_no_ext = os.path.splitext(os.path.basename(__file__))[0]
log_file = os.path.join(
    log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger.basicConfig(filename=log_file)
logger.orange(f"Logs: {log_file}")


async def main():
    task = 'Find the founders of browser-use'
    agent = Agent(task=task, llm=ChatOllama(model='llama3.2', log_dir=log_dir))
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
