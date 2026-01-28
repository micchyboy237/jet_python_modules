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
# shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
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
    Searches for text on the current page using XPath contains(text()) and scrolls the nth matching element into view.

    Args:
        text: The exact or partial text content to search for on the page.
        nth_result: Which occurrence to focus on (1 = first match, 2 = second, etc.). Defaults to 1.

    Returns:
        A message describing how many matches were found and which one was focused.
    """
    elements = helium.get_driver().find_elements(
        By.XPATH, f"//*[contains(text(), '{text}')]"
    )
    if not elements:
        return f"No elements containing '{text}' were found."

    if nth_result > len(elements):
        return f"Only {len(elements)} matches found for '{text}'. Cannot jump to #{nth_result}."

    elem = elements[nth_result - 1]
    helium.get_driver().execute_script(
        "arguments[0].scrollIntoView({block: 'center'});", elem
    )
    return f"Found {len(elements)} matches for '{text}'. Focused on occurrence #{nth_result}."


@tool
def go_back() -> str:
    """
    Navigates the browser back to the previous page in history.

    Returns:
        Confirmation message that navigation back was performed.
    """
    helium.get_driver().back()
    return "Navigated back to previous page."


@tool
def close_popups() -> str:
    """
    Sends ESC key to attempt closing any visible modal, popup or overlay (except cookie banners).

    Returns:
        Message confirming that ESC key was sent.
    """
    webdriver.ActionChains(helium.get_driver()).send_keys(Keys.ESCAPE).perform()
    sleep(0.4)
    return "Sent ESC key to attempt closing popup/modal."


# ────────────────────────────────────────────────
#  2. Screenshot callback
# ────────────────────────────────────────────────


def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(0.9)  # let animations / paint settle

    driver = helium.get_driver()
    if driver is None:
        return

    # Optional: clean up older screenshots to save memory
    for prev_step in agent.memory.steps:
        if (
            isinstance(prev_step, ActionStep)
            and prev_step.step_number <= memory_step.step_number - 3
        ):
            prev_step.observations_images = None

    try:
        png_bytes = driver.get_screenshot_as_png()
        img = Image.open(BytesIO(png_bytes))
        print(f"[Screenshot] captured at step {memory_step.step_number} → {img.size}")

        # Optional: save to disk for debugging
        screenshot_dir = OUTPUT_DIR / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{str(screenshot_dir)}/step_{memory_step.step_number:03d}.png"
        img.save(filename)
        print(f"[Screenshot saved] {filename}")

        memory_step.observations_images = [img.copy()]

        # Also attach current URL
        url_line = f"Current URL: {driver.current_url}"
        print(f"[Observation] {url_line}")
        if memory_step.observations:
            memory_step.observations += "\n" + url_line
        else:
            memory_step.observations = url_line

    except Exception as e:
        print(f"[Screenshot] failed: {e}")


# ────────────────────────────────────────────────
#  3. Browser Setup
# ────────────────────────────────────────────────


# At the top of the file (or in if __name__ == "__main__":)
# ChromeDriverManager().install()  # run once → downloads correct version


def init_browser(headless: bool = False):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1100,1400")
    chrome_options.add_argument("--window-position=20,20")
    chrome_options.add_argument("--disable-pdf-viewer")

    if headless:
        chrome_options.add_argument("--headless=new")

    # No need to do anything else – Helium will use the driver that webdriver-manager prepared
    driver = helium.start_chrome(
        options=chrome_options,
        headless=headless,
    )
    return driver


# ────────────────────────────────────────────────
#  4. Helium + Agent Instructions
# ────────────────────────────────────────────────

HELIUM_GUIDE = """
You are controlling a real web browser using **Helium** commands (already imported via `from helium import *`).

Important commands:

- go_to('https://example.com')               → navigate to URL
- click("Sign in")                            → click element with exact text
- click(Link("Blog"))                         → click link with that text
- write("search term", into="q")              → type into input/textarea
- write("text", into=TextField("label"))      → more precise
- scroll_down(num_pixels=800)                 → scroll viewport
- scroll_up(num_pixels=400)
- S("#some-id").exists()                      → check existence (returns bool)
- Text("some text").exists()
- kill_browser()                              → close browser (use only at very end)

Rules:
- After every click / write / navigation → STOP and let the screenshot show the result
- NEVER call screenshot(), driver.get_screenshot_as_png(), print(screenshot()) or similar!
- Screenshots are automatically taken after each step via a callback — you will see them in observations.
- Do NOT try to take, print or save screenshots yourself — it is forbidden and will crash.
- Never try to log in to personal accounts
- For popups/modals with close button → prefer close_popups() tool over clicking X
- Use search_item_ctrl_f("text") if you need to locate something by text content
- Be conservative with number of steps — aim for clarity over speed

Never guess element selectors — use Helium's text/link based selectors whenever possible.
""".strip()


# ────────────────────────────────────────────────
#  5. Main Agent Setup & Run
# ────────────────────────────────────────────────


def create_local_model(
    temperature: float = 0.7,
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
        tools=[search_item_ctrl_f, go_back, close_popups],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=25,
        verbosity_level=2,  # 0 = quiet, 1 = normal, 2 = verbose
        add_base_tools=True,
    )

    # Critical: preload Helium symbols into the sandboxed namespace
    agent.python_executor("from helium import *")
    agent.python_executor("get_driver = get_driver")

    # Example task 1 – Wikipedia
    task = """
    Navigate to https://en.wikipedia.org/wiki/Chicago
    Find and quote one sentence that contains the word "1992" and mentions any kind of accident or disaster.
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
