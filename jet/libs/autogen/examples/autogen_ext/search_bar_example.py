import os
import shutil
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
from playwright.async_api import async_playwright, Page
from autogen_ext.agents.web_surfer._types import InteractiveRegion
from jet.libs.autogen.playwright_controller import PlaywrightController
from jet.logger import CustomLogger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

# Define FAKE_HTML with a form, submit button, styled search result, and input refocus
FAKE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake Page</title>
</head>
<body>
    <h1 id="header">Welcome to the Fake Page</h1>
    <form id="search-form">
        <input type="text" id="input-box" />
        <button type="submit" id="submit-button">Search</button>
    </form>
    <div id="search-result"></div>
    <div id="long-content" style="height: 2000px;">Long content for scrolling</div>
    <script>
        document.getElementById('search-form').addEventListener('submit', function(event) {
            event.preventDefault();
            const query = document.getElementById('input-box').value;
            document.getElementById('search-result').innerHTML = 
                '<h2 style="color: red; text-align: center;">Search Result: ' + query + '</h2>';
            document.getElementById('input-box').focus();
        });
    </script>
</body>
</html>
"""


async def search_bar_example() -> None:
    """
    Demonstrates using PlaywrightController to find a search bar and submit button by ID,
    input text, click submit, save screenshots, and verify a styled changed state.
    """
    # Define screenshot directory and timestamp for unique filenames
    screenshot_dir = f"{OUTPUT_DIR}/screenshots"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(screenshot_dir, exist_ok=True)

    async with async_playwright() as p:
        # Initialize browser and page
        browser = await p.chromium.launch(headless=False)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.set_content(FAKE_HTML)

            # Save screenshot of initial page
            initial_screenshot_path = os.path.join(
                screenshot_dir, f"initial_page_{timestamp}.png")
            await page.screenshot(path=initial_screenshot_path)
            logger.info(
                f"Saved initial page screenshot to: {initial_screenshot_path}")

            # Initialize PlaywrightController
            controller = PlaywrightController()

            # Find the search bar's __elementId
            rects: Dict[str, InteractiveRegion] = await controller.get_interactive_rects(page)
            search_bar_id: Optional[str] = next(
                (rect for rect in rects if rects[rect]
                 ["tag_name"] == "input, type=text"),
                None
            )
            if search_bar_id is None:
                logger.error("Search bar input not found in interactive rects")
                return

            # Find the submit button's __elementId
            submit_button_id: Optional[str] = next(
                (rect for rect in rects if rects[rect]
                 ["aria_name"] == "Search"),
                None
            )
            if submit_button_id is None:
                logger.error("Submit button not found in interactive rects")
                return

            # Input search query
            search_query = "example search query"
            await controller.fill_id(page, search_bar_id, search_query, press_enter=False)
            logger.info(f"Input search query: {search_query}")

            # Save screenshot after filling search bar
            filled_screenshot_path = os.path.join(
                screenshot_dir, f"search_bar_filled_{timestamp}.png")
            await page.screenshot(path=filled_screenshot_path)
            logger.info(
                f"Saved search bar filled screenshot to: {filled_screenshot_path}")

            # Click the submit button
            await controller.click_id(page, submit_button_id)
            logger.info(f"Clicked submit button with ID: {submit_button_id}")

            # Wait for DOM update to ensure changed state is rendered
            await page.wait_for_timeout(500)

            # Save screenshot after clicking submit to capture changed state
            submitted_screenshot_path = os.path.join(
                screenshot_dir, f"search_submitted_{timestamp}.png")
            await page.screenshot(path=submitted_screenshot_path)
            logger.info(
                f"Saved search submitted screenshot showing changed state to: {submitted_screenshot_path}")

            # Verify the input value
            result_value = await page.evaluate("document.getElementById('input-box').value")
            if result_value == search_query:
                logger.info(
                    f"Success: Search bar contains expected value '{result_value}'")
            else:
                logger.error(
                    f"Failure: Expected search bar value '{search_query}', got '{result_value}'")

            # Verify the search result (changed state)
            result_text = await page.evaluate("document.getElementById('search-result').innerText")
            expected_result = f"Search Result: {search_query}"
            if result_text == expected_result:
                logger.info(f"Success: Search result displays '{result_text}'")
            else:
                logger.error(
                    f"Failure: Expected search result '{expected_result}', got '{result_text}'")

            # Verify the input is focused
            result_focused_id = await controller.get_focused_rect_id(page)
            if result_focused_id == search_bar_id:
                logger.info(
                    f"Success: Search bar with ID '{search_bar_id}' is focused after submission")
            else:
                logger.error(
                    f"Failure: Expected focused ID '{search_bar_id}', got '{result_focused_id}'"
                )

            # Save final screenshot after verification
            final_screenshot_path = os.path.join(
                screenshot_dir, f"final_state_{timestamp}.png")
            await page.screenshot(path=final_screenshot_path)
            logger.info(
                f"Saved final state screenshot to: {final_screenshot_path}")

        finally:
            await browser.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(search_bar_example())
