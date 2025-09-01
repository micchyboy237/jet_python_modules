import os
import shutil
import asyncio
from typing import List, Dict, Literal
from jet.libs.autogen.examples.MultimodalWebSurfer.config import make_surfer
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

    # Define task steps with explicit instructions and edge case handling
    task_steps: List[Dict[str, str]] = [
        {
            "description": "Navigate to aniwatchtv.to",
            "task": "Visit https://aniwatchtv.to"
        },
        {
            "description": "Search for Solo Leveling",
            "task": (
                "Scroll down the page multiple times to ensure the search bar is visible. "
                "Wait for the search bar to load (up to 5 seconds). "
                "Find a textbox or searchbox element with role 'searchbox' or 'textbox', "
                "input 'Solo Leveling', and press Enter to submit the search. "
                "If the search bar is not found, visit https://aniwatchtv.to/search?keyword=Solo+Leveling."
            )
        },
        {
            "description": "Navigate to episode 12",
            "task": (
                "Scroll down the search results to find the link for 'Solo Leveling' "
                "(not 'Solo Leveling Season 2'). Click the main title link to go to the show’s page. "
                "If the 'Solo Leveling' link is not found, look for related links like 'Solo Leveling Season 1' "
                "or other links containing 'Solo Leveling' and click one. "
                "On the show’s page, scroll down and click the link or button for episode 12. "
                "If episode 12 is not found, check for an episode list or season 1 links and select episode 12."
            )
        },
        {
            "description": "Extract watch link",
            "task": (
                "Use answer_question to extract the URL of the current page as the watch link. "
                "The question to answer is: 'What is the current page URL?'"
            )
        }
    ]

    result = ""
    max_retries = 2
    for step in task_steps:
        logger.debug(f"Executing step: {step['description']}")
        attempt = 0
        while attempt <= max_retries:
            try:
                step_result = await surfer.run(task=step["task"])
                logger.debug(f"Step result: {step_result}")
                result += f"{step['description']}: {step_result}\n"
                break
            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1} failed for '{step['description']}': {str(e)}")
                attempt += 1
                if attempt > max_retries:
                    logger.error(
                        f"Max retries reached for '{step['description']}'. Moving to next step.")
                    result += f"{step['description']}: Failed after {max_retries} retries: {str(e)}\n"
                    break
                # Handle specific edge cases
                if "No such element" in str(e) and "search" in step["description"].lower():
                    logger.debug(
                        "Search bar not found, using direct search URL...")
                    fallback_task = "Visit https://aniwatchtv.to/search?keyword=Solo+Leveling"
                    try:
                        step_result = await surfer.run(task=fallback_task)
                        logger.debug(f"Fallback result: {step_result}")
                        result += f"{step['description']} (fallback): {step_result}\n"
                        break
                    except Exception as fallback_e:
                        logger.error(f"Fallback failed: {str(fallback_e)}")
                elif "No such element" in str(e) and "episode 12" in step["description"].lower():
                    logger.debug(
                        "Episode 12 link not found, trying alternative navigation...")
                    fallback_task = (
                        "Scroll down the page to find any link containing 'Solo Leveling'. "
                        "Click the link to navigate to the show’s page, then find and click episode 12."
                    )
                    try:
                        step_result = await surfer.run(task=fallback_task)
                        logger.debug(f"Fallback result: {step_result}")
                        result += f"{step['description']} (fallback): {step_result}\n"
                        break
                    except Exception as fallback_e:
                        logger.error(f"Fallback failed: {str(fallback_e)}")
                # Wait before retrying to allow dynamic content to load
                await asyncio.sleep(2)

    try:
        logger.debug("Closing browser...")
        await surfer.close()
        logger.debug("Browser closed.")
    except Exception as e:
        logger.error(f"Error closing browser: {str(e)}")

    logger.info(f"✅ Search complete\n{result}")

if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
