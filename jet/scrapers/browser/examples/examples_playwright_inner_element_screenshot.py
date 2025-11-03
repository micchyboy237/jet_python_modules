import os
import shutil
from pathlib import Path
from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from playwright.sync_api import sync_playwright

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def example_inner_element_screenshot():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE)
        page = browser.new_page()
        page.goto("https://example.com")

        # Locate the element you want to capture
        element = page.query_selector("h1")

        if element:
            screenshot_path = Path(os.path.join(OUTPUT_DIR, "element_screenshot.png")).resolve()
            # Take a screenshot of just that element
            element.screenshot(path=str(screenshot_path))
            print(f"✅ Screenshot saved as {str(screenshot_path)}")
        else:
            print("❌ Element not found")

        browser.close()

if __name__ == "__main__":
    example_inner_element_screenshot()
