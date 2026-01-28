#!/usr/bin/env python3
"""
Web Browser Automation Agent using Helium + smolagents + Selenium
with local SearXNG support at http://searxng.local
"""

from io import BytesIO
import os
import shutil
from pathlib import Path
from time import sleep
from typing import Optional

from PIL import Image
from dotenv import load_dotenv

import helium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from seleniumbase import Driver

from smolagents import CodeAgent, OpenAIModel, tool

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

# ────────────────────────────────────────────────
# Browser Tools
# ────────────────────────────────────────────────


@tool
def open_search_engine(query: str = "") -> str:
    """
    Opens the local SearXNG instance and (optionally) performs a search with the given query.
    Use this as the preferred way to search the web instead of Google / other engines.

    Args:
        query: The search query to pre-fill in SearXNG. Leave empty to open the homepage only.
    """
    url = "http://searxng.local:8888"
    if query:
        from urllib.parse import quote

        url += f"/?q={quote(query)}"

    helium.go_to(url)
    return f"Opened local SearXNG at {url}"


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
    if not elements:
        return f"No matches found for '{text}'"
    if nth_result > len(elements):
        return (
            f"Only {len(elements)} matches found for '{text}' (asked for #{nth_result})"
        )

    elem = elements[nth_result - 1]
    helium.get_driver().execute_script(
        "arguments[0].scrollIntoView({block: 'center'});", elem
    )
    return f"Focused match #{nth_result} of {len(elements)} for '{text}'"


@tool
def go_back() -> str:
    """Goes back to previous page."""
    helium.get_driver().back()
    return "Went back one page"


@tool
def close_popups() -> str:
    """Attempts to close modal/popups by sending ESC key."""
    webdriver.ActionChains(helium.get_driver()).send_keys(Keys.ESCAPE).perform()
    return "Sent ESC to try closing any popup/modal"


# ────────────────────────────────────────────────
# Screenshot callback (unchanged)
# ────────────────────────────────────────────────


def save_screenshot(memory_step, agent):
    sleep(0.8)
    driver = helium.get_driver()
    if not driver:
        return

    screenshot_dir = OUTPUT_DIR / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    fname = screenshot_dir / f"step_{memory_step.step_number:03d}.png"
    driver.save_screenshot(str(fname))

    title = (driver.title or "(no title)").strip()[:160]
    obs = f"URL: {driver.current_url}\nTitle: {title}\nScreenshot: {fname}"

    # Optional: selected text
    try:
        sel = driver.execute_script("return window.getSelection().toString().trim();")
        if sel:
            obs += f"\nSelected text: {sel[:100]}"
    except:
        pass

    png_bytes = driver.get_screenshot_as_png()
    image = Image.open(BytesIO(png_bytes))

    # Optional: save to disk for debugging
    screenshot_dir = OUTPUT_DIR / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot_file = f"{str(screenshot_dir)}/step_{memory_step.step_number:03d}.png"
    image.save(screenshot_file)
    print(f"[Screenshot saved] {screenshot_file}")

    # Save text obs under observations dir
    texts_dir = OUTPUT_DIR / "observations"
    texts_dir.mkdir(parents=True, exist_ok=True)
    text_file = texts_dir / f"step_{memory_step.step_number:03d}.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(obs)
    print(f"[Observations saved] {text_file}")

    memory_step.observations = (memory_step.observations or "") + "\n" + obs


# ────────────────────────────────────────────────
# Browser init
# ────────────────────────────────────────────────


def init_browser(headless: bool = True) -> "Driver":
    """
    Initialize an anti-detection browser instance using SeleniumBase UC mode.
    """
    driver = Driver(
        browser="chrome",
        uc=True,
        headless=headless,
        agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        window_size="1280,800",
    )

    # ──── ADD THIS LINE ────
    import helium

    helium.set_driver(driver)
    # ───────────────────────

    # Optional extra stealth (already good with uc=True)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            """
        },
    )

    return driver


# ────────────────────────────────────────────────
# Updated Helium + Agent Instructions
# ────────────────────────────────────────────────

HELIUM_GUIDE = """
You are controlling a real browser with Helium.

Preferred way to search:
    Use the tool `open_search_engine("your question")` — it opens http://searxng.local
    Only use Google/Bing/DuckDuckGo when explicitly asked or when SearXNG fails.

Basic commands:
    go_to('https://example.com')
    click("Sign in")               # text on button/link
    click(Link("About"))           # more precise for links
    write("search phrase", into="q")  # or into(TextField(...)))

    scroll_down(800)   # pixels
    scroll_up(400)

    if Text("Accept all cookies").exists():
        click("Accept all cookies")

Popups:
    Use close_popups() instead of trying to click × icons

After every click / navigation / form submit → wait for next screenshot.
Never try to log in / authenticate anywhere.
""".strip()


# ────────────────────────────────────────────────
# Local model helper
# ────────────────────────────────────────────────


def create_local_model(temperature=0.35, max_tokens=2048):
    return OpenAIModel(
        model_id="local-model",
        api_base="http://shawn-pc.local:8080/v1",
        api_key="sk-no-need",
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────


def main():
    model = create_local_model()
    driver = init_browser(headless=False)  # ← set False for debugging

    agent = CodeAgent(
        tools=[open_search_engine, go_back, close_popups, search_item_ctrl_f],
        model=model,
        additional_authorized_imports=["helium", "urllib.parse"],
        step_callbacks=[save_screenshot],
        max_steps=30,
        verbosity_level=2,
        add_base_tools=True,
    )

    # Preload Helium
    agent.python_executor("from helium import *")

    task = """
Use the local search engine to find information about the tallest building in the Philippines as of 2026.
Then go to the most relevant Wikipedia page (or official page) and tell me:
- official name
- height in meters
- number of floors
- city
"""

    print("=" * 75)
    print("Task:")
    print(task.strip())
    print("=" * 75 + "\n")

    answer = agent.run(task + "\n\n" + HELIUM_GUIDE)

    print("\n" + "═" * 75)
    print("FINAL ANSWER:")
    print(answer)
    print("═" * 75 + "\n")

    print("Browser stays open for 10 seconds...")
    sleep(10)
    try:
        helium.kill_browser()
    except:
        pass


if __name__ == "__main__":
    main()
