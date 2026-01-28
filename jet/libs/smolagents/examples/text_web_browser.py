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

from random import uniform  # NEW: add for random sleep

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
def find_text_on_page(text: str, nth_result: int = 1) -> str:
    """
    Finds visible DOM text on the current page and scrolls to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """

    # Safely build XPath literal to avoid injection / syntax errors
    def _xpath_literal(s: str) -> str:
        if "'" not in s:
            return f"'{s}'"
        if '"' not in s:
            return f'"{s}"'
        parts = s.split("'")
        return "concat(" + ', "\'", '.join(f"'{p}'" for p in parts) + ")"

    elements = helium.get_driver().find_elements(
        By.XPATH, f"//*[contains(text(), {_xpath_literal(text)})]"
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
def extract_visible_text(max_chars: int = 4000) -> str:
    """
    Extracts visible text from the current page using DOM APIs.
    Safer and cleaner than XPath-based scraping.

    Args:
        max_chars: Maximum characters to return (default: 4000)
    """
    driver = helium.get_driver()
    if not driver:
        return "No active browser session."

    try:
        text = driver.execute_script(
            "return document.body && document.body.innerText || '';"
        )
    except Exception as e:
        return f"Failed to extract DOM text: {e}"

    text = text.strip()
    if not text:
        return "No visible text found on page."

    return text[:max_chars]


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


@tool
def visit_webpage_safe(url: str, referer: str = "https://searxng.local:8888") -> str:
    """
    Navigate to the given URL with added stealth measures and human-like delays.

    Args:
        url: The webpage URL to visit (required).
        referer: Optional HTTP Referer header to send (helps avoid some anti-bot checks).
                 Defaults to the local SearXNG instance.

    Returns:
        A short confirmation message indicating successful navigation.
    """
    sleep(uniform(1.8, 4.2))
    helium.get_driver().execute_cdp_cmd(
        "Network.setExtraHTTPHeaders", {"headers": {"Referer": referer}}
    )
    helium.go_to(url)
    sleep(uniform(2.5, 5.0))
    return f"Navigated to {url} (safe mode)"


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
    except Exception as e:
        obs += f"\n[Selection error: {e}]"

    # Save text obs under observations dir
    texts_dir = OUTPUT_DIR / "observations"
    texts_dir.mkdir(parents=True, exist_ok=True)
    text_file = texts_dir / f"step_{memory_step.step_number:03d}.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(obs)
    print(f"[Observations saved] {text_file}")

    # ── Step-level context summarization / truncation ──
    prev = memory_step.observations or ""
    combined = (prev + "\n" + obs).strip()

    MAX_CONTEXT_CHARS = 6000
    KEEP_TAIL_CHARS = 2500

    if len(combined) > MAX_CONTEXT_CHARS:
        summary = (
            "[Summary of previous steps]\n"
            + combined[: MAX_CONTEXT_CHARS - KEEP_TAIL_CHARS].split("\n")[-20:]
        )
        summary_text = "\n".join(summary)
        tail = combined[-KEEP_TAIL_CHARS:]
        memory_step.observations = summary_text + "\n...\n" + tail
    else:
        memory_step.observations = combined


# ────────────────────────────────────────────────
# Browser init
# ────────────────────────────────────────────────


def init_browser(headless: bool = True) -> "Driver":
    """
    Initialize an anti-detection browser instance using SeleniumBase UC mode.
    """
    # Optional: rotate user-agent per script run (helps long-running agents)
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ]

    selected_ua = random.choice(user_agents)

    driver = Driver(
        browser="chrome",
        uc=True,
        headless=headless,
        agent=selected_ua,
        window_size="1000,1350",
        window_position="0,0",
        d_p_r=1.0,
        chromium_arg="--disable-pdf-viewer",
    )

    helium.set_driver(driver)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            """
        },
    )

    # Increase timeout for slow / strict sites like Wikipedia
    driver.set_page_load_timeout(45)

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
        tools=[
            open_search_engine,
            go_back,
            close_popups,
            find_text_on_page,
            extract_visible_text,
            visit_webpage_safe,  # include the new "safe" navigation helper
        ],
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
    except Exception as e:
        print(f"[Browser shutdown warning] {e}")


if __name__ == "__main__":
    main()
