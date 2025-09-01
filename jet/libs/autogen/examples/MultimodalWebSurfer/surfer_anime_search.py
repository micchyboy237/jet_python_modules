import os
import shutil
import asyncio
from jet.libs.autogen.examples.MultimodalWebSurfer.config import make_surfer
from typing import Any, List, Dict
from jet.logger import CustomLogger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


async def main():
    surfer = make_surfer(
        debug_dir=f"{OUTPUT_DIR}/debug_screens",
        browser_data_dir=f"{OUTPUT_DIR}/browser_data_dir",
    )
    logger.info(
        "🔍 Starting search for Solo Leveling episode 12 on aniwatchtv.to...")
    task_steps: List[str] = [
        "Visit https://aniwatchtv.to",
        "Wait for the page to load fully",
        "Type 'Solo Leveling' into the search bar.",
        "Submit search",
    ]
    result = ""
    try:
        for step in task_steps:
            logger.debug(f"Executing step: {step}")
            step_result = await surfer.run(task=step)
            logger.debug(f"Step result: {step_result}")

    except Exception as e:
        logger.error(f"Unexpected error during task execution: {str(e)}")
        raise
    finally:
        logger.debug("Closing browser...")
        await surfer.close()
        logger.debug("Browser closed.")
    logger.info(f"✅ Search complete\n{result}")

if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
