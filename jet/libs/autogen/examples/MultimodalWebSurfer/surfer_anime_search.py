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
    task_steps: List[Dict[str, Any]] = [
        {
            "description": "Navigate to aniwatchtv.to homepage",
            "task": "Visit https://aniwatchtv.to",
            "tool": "visit_url",
            "parameters": {
                "reasoning": "Navigate to the homepage to begin the search for Solo Leveling.",
                "url": "https://aniwatchtv.to"
            }
        },
        {
            "description": "Wait for the page to load fully",
            "task": "Wait for the page to load fully",
            "tool": "sleep",
            "parameters": {
                "reasoning": "Identify the search bar to input the query for Solo Leveling.",
                "element_type": "textbox",
                "name": "keyword"  # Assuming the search bar has a name attribute like 'keyword'
            }
        },
        {
            "description": "Input 'Solo Leveling' into the search bar",
            "task": "Type 'Solo Leveling' into the search bar.",
            "tool": "input_text",
            "parameters": {
                "reasoning": "Enter the search term to find Solo Leveling in the website’s search.",
                "input_field_id": None,  # Will be dynamically set from find_form_elements result
                "text_value": "Solo Leveling"
            }
        },
    ]
    result = ""
    try:
        last_form_elements = None
        for step in task_steps:
            logger.debug(f"Executing step: {step['description']}")
            try:
                step_result = await surfer.run(task=step["task"])
                logger.debug(f"Step result: {step_result}")

                # Handle find_form_elements results
                if step["tool"] == "find_form_elements":
                    last_form_elements = step_result
                    result += f"{step['description']}: Found elements {step_result}\n"

                # Update input_text parameters dynamically
                elif step["tool"] == "input_text" and last_form_elements:
                    if isinstance(last_form_elements, list) and last_form_elements:
                        step["parameters"]["input_field_id"] = last_form_elements[0].get(
                            "id")
                        step_result = await surfer.run(
                            task=f"Type '{step['parameters']['text_value']}' into input field {step['parameters']['input_field_id']}."
                        )
                        result += f"{step['description']}: {step_result}\n"
                    else:
                        raise ValueError(
                            "No valid form elements found for input_text")

                # Update click parameters for search button dynamically
                elif step["tool"] == "click" and "submit the query" in step["description"].lower() and last_form_elements:
                    if isinstance(last_form_elements, list) and last_form_elements:
                        step["parameters"]["target_id"] = last_form_elements[0].get(
                            "id")
                        step_result = await surfer.run(
                            task=f"Click the element with id {step['parameters']['target_id']}."
                        )
                        result += f"{step['description']}: {step_result}\n"
                    else:
                        raise ValueError(
                            "No valid form elements found for click")

                else:
                    result += f"{step['description']}: {step_result}\n"
            except Exception as e:
                logger.error(
                    f"Error in step '{step['description']}': {str(e)}")
                if "No such element" in str(e) and "search" in step["description"].lower():
                    logger.debug(
                        "Search bar not found, using direct search URL...")
                    fallback_task = "Visit https://aniwatchtv.to/search?keyword=Solo+Leveling"
                    step_result = await surfer.run(task=fallback_task)
                    logger.debug(f"Fallback result: {step_result}")
                    result += f"{step['description']} (fallback): {step_result}\n"
                else:
                    raise
    except Exception as e:
        logger.error(f"Unexpected error during task execution: {str(e)}")
        result += f"Error: {str(e)}"
    finally:
        logger.debug("Closing browser...")
        await surfer.close()
        logger.debug("Browser closed.")
    logger.info(f"✅ Search complete\n{result}")

if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
