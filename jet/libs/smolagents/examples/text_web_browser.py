#!/usr/bin/env python3
"""
Web Browser Automation Agent using Helium + smolagents + Selenium
================================================================

Dependencies to install:
    pip install smolagents selenium helium pillow python-dotenv

Required:
- Chrome browser installed
- ChromeDriver (usually auto-managed by helium)
- Optional: .env file with any API tokens if your model requires them
"""

import os
import shutil
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import List, Optional

from PIL import Image
from dotenv import load_dotenv

import helium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from smolagents import CodeAgent, OpenAIModel, tool, InferenceClientModel
from smolagents.agents import ActionStep

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────
#  0. Load environment (API keys, etc.)
# ────────────────────────────────────────────────
load_dotenv()


# ────────────────────────────────────────────────
#  1. Browser Tools
# ────────────────────────────────────────────────


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    elements = helium.get_driver().find_elements(
        By.XPATH, f"//*[contains(text(), '{text}')]"
    )
    if nth_result > len(elements):
        raise Exception(
            f"Match n°{nth_result} not found (only {len(elements)} matches found)"
        )
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    helium.get_driver().execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    helium.get_driver().back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows!
    This does not work on cookie consent banners.
    """
    webdriver.ActionChains(helium.get_driver()).send_keys(Keys.ESCAPE).perform()
    return "Sent ESC key to attempt closing popup/modal."


# ────────────────────────────────────────────────
#  2. Screenshot callback
# ────────────────────────────────────────────────


# Set up screenshot callback
def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Give page time to settle
    driver = helium.get_driver()
    if driver is None:
        memory_step.observations = (
            memory_step.observations or ""
        ) + "\n[Warning] No active browser driver."
        return

    # ── Always save screenshot for human debugging ───────────────────────
    screenshot_dir = OUTPUT_DIR / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    filename = screenshot_dir / f"step_{memory_step.step_number:03d}.png"

    driver.save_screenshot(str(filename))  # simpler, no PIL needed here
    print(f"[Debug screenshot] {filename}")

    # ── Build rich text observation (this becomes the real context) ──────
    try:
        page_title = driver.title.strip()[:180] or "(no title)"
    except:
        page_title = "(title fetch failed)"

    url_line = f"Current URL: {driver.current_url}"
    title_line = f"Page title: {page_title}"
    img_line = f"Screenshot saved for inspection: {filename}"

    new_text = f"{url_line}\n{title_line}\n{img_line}"

    # Optional: try to get any highlighted/selected text or focused element
    try:
        focused_text = driver.execute_script(
            "return window.getSelection().toString() || document.activeElement.value || '';"
        ).strip()[:120]
        if focused_text:
            new_text += f'\nFocused/selected text: "{focused_text}"'
    except:
        pass

    # Append to existing observations (cumulative context)
    if memory_step.observations is None:
        memory_step.observations = new_text
    else:
        memory_step.observations += "\n\n" + new_text

    # Critical: NEVER set this for text-only model
    # memory_step.observations_images = None  # (already not set)


# ────────────────────────────────────────────────
#  3. Browser Setup
# ────────────────────────────────────────────────


# At the top of the file (or in if __name__ == "__main__":)
# ChromeDriverManager().install()  # run once → downloads correct version


def init_browser(headless: bool = False):
    # Configure Chrome options
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")

    if headless:
        chrome_options.add_argument("--headless=new")

    # Initialize the browser
    driver = helium.start_chrome(
        options=chrome_options,
        headless=headless,
    )
    return driver


# ────────────────────────────────────────────────
#  4. Helium + Agent Instructions
# ────────────────────────────────────────────────

HELIUM_GUIDE = """
You can use helium to access websites. Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
Code:
```py
go_to('github.com/trending')
```<end_code>

You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>

If it's a link:
Code:
```py
click(Link("Top products"))
```<end_code>

If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.

To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
Code:
```py
scroll_down(num_pixels=1200) # This will scroll one viewport down
```<end_code>

When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
```py
close_popups()
```<end_code>

You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>
""".strip()


# ────────────────────────────────────────────────
#  5. Main Agent Setup & Run
# ────────────────────────────────────────────────


def create_local_model(
    temperature: float = 0.3,
    max_tokens: Optional[int] = 2048,
    model_id: str = "local-model",
) -> OpenAIModel:
    return OpenAIModel(
        model_id=model_id,
        api_base="http://shawn-pc.local:8080/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def main():
    # You can change this model if you have access
    # model_id = "Qwen/Qwen2-VL-72B-Instruct"
    # model_id = "mistralai/Pixtral-12B-2409"   # alternative (if supported)

    # model = InferenceClientModel(model_id=model_id)
    model = create_local_model()

    # Initialize browser
    driver = init_browser(headless=True)  # ← change to True in production

    # Create agent
    agent = CodeAgent(
        tools=[go_back, close_popups, search_item_ctrl_f],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=25,
        verbosity_level=2,  # 0 = quiet, 1 = normal, 2 = verbose
        add_base_tools=True,
    )

    # Critical: preload Helium symbols into the sandboxed namespace
    agent.python_executor("from helium import *")
    # agent.python_executor("get_driver = get_driver")

    # Example task 1 – Wikipedia
    task = """
Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence containing the word "1992" that mentions a construction accident.
"""

    # Example task 2 – GitHub trending (alternative)
    # task = """
    # Go to https://github.com/trending
    # Click on the first repository in the list.
    # Go to the contributors graph or main contributor profile if visible.
    # Tell me the username of the top contributor and — if possible — their commit count in the last year.
    # """

    print("\n" + "=" * 70)
    print("Starting agent with task:")
    print(task.strip())
    print("=" * 70 + "\n")

    final_answer = agent.run(task + "\n\n" + HELIUM_GUIDE)

    print("\n" + "═" * 70)
    print("FINAL AGENT OUTPUT:")
    print(final_answer)
    print("═" * 70 + "\n")

    # Optional: keep browser open for inspection
    print("Browser will stay open for 30 seconds...")
    sleep(30)

    # Clean up
    try:
        helium.kill_browser()
    except:
        pass


if __name__ == "__main__":
    main()
