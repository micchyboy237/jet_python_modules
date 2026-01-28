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

import argparse
import json
import os
import random
import shutil
from io import BytesIO
from pathlib import Path
from time import sleep
from datetime import datetime
from typing import List, Optional

from PIL import Image
from dotenv import load_dotenv

import helium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from seleniumbase import Driver

from smolagents import CodeAgent, tool, InferenceClientModel
from smolagents.agents import ActionStep
from smolagents.utils import make_json_serializable

from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.helium_tools import (
    ALL_TOOLS,
    click,
    go_back,
    go_to,
    scroll_down,
    scroll_up,
)

# ────────────────────────────────────────────────
#  0. Load environment (API keys, etc.)
# ────────────────────────────────────────────────
load_dotenv()


# ────────────────────────────────────────────────
#  1. Custom Browser Tools
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
def save_screenshot(memory_step: ActionStep, agent: CodeAgent, base_dir: Path) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    current_step = memory_step.step_number
    driver = helium.get_driver()
    if driver is not None:
        for previous_memory_step in agent.memory.steps:
            if (
                isinstance(previous_memory_step, ActionStep)
                and previous_memory_step.step_number <= current_step - 2
            ):
                previous_memory_step.observations_images = None

        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))

        # Optional: save to disk for debugging
        screenshot_dir = base_dir / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        filename = screenshot_dir / f"step_{current_step:03d}.png"
        image.save(filename)
        print(f"[Screenshot saved] {filename}")

        messages_dir = base_dir / "messages"
        messages_dir.mkdir(parents=True, exist_ok=True)
        msg_path = messages_dir / f"step_{current_step:03d}.json"

        with open(msg_path, "w", encoding="utf-8") as f:
            json.dump(
                _strip_observations_images(
                    make_json_serializable([agent.memory.steps[-1]])
                ),
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[Messages saved] {msg_path}")

        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]

    # Update observations with current URL
    obs = f"Current url: {driver.current_url}"
    prev_obs = memory_step.observations

    print(f"[Observation] Step {current_step} -> {obs}")

    # Save text obs under observations dir
    texts_dir = base_dir / "observations"
    texts_dir.mkdir(parents=True, exist_ok=True)
    text_file = texts_dir / f"step_{current_step:03d}.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(obs)
    print(f"[Observations saved] {text_file}")

    memory_step.observations = obs if prev_obs is None else prev_obs + "\n" + obs


# ────────────────────────────────────────────────
#  3. Browser Setup
# ────────────────────────────────────────────────


# At the top of the file (or in if __name__ == "__main__":)
# ChromeDriverManager().install()  # run once → downloads correct version


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


from datetime import datetime


def create_local_model(
    temperature: float = 0.3,
    max_tokens: Optional[int] = 4096,
    model_id: str = "local-model",
    logs_dir: str | Path | None = None,
) -> OpenAIModel:
    return OpenAIModel(
        model_id=model_id,
        api_base="http://shawn-pc.local:8080/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=True,
        logs_dir=str(logs_dir) if logs_dir else None,
    )


def _strip_observations_images(steps):
    # Remove 'observations_images' from each step before saving
    stripped_steps = []
    for step in steps:
        if isinstance(step, dict):
            step_copy = dict(step)
        else:
            step_copy = dict(step) if hasattr(step, "dict") else step.__dict__.copy()
        step_copy.pop("observations_images", None)
        stripped_steps.append(step_copy)
    return stripped_steps


def main(
    headless: bool = False,
    task: str | None = None,
    out_dir: Path | None = None,
) -> None:
    # Determine output directory with timestamp for this run if not provided
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        base = Path(__file__).parent / "generated" / Path(__file__).stem
        out_dir = base / f"run_{timestamp}"

    out_dir = out_dir.resolve()  # normalize path
    out_dir.mkdir(parents=True, exist_ok=True)  # create if missing

    print(f"Output directory: {out_dir}")

    # Default task (Wikipedia example) if none provided
    default_task = (
        "Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence "
        'containing the word "1992" that mentions a construction accident.'
    )

    # Use provided task or fall back to default
    final_task = task.strip() if task and task.strip() else default_task

    # You can change this model if you have access
    # model_id = "Qwen/Qwen2-VL-72B-Instruct"
    # model_id = "mistralai/Pixtral-12B-2409"   # alternative (if supported)

    # model = InferenceClientModel(model_id=model_id)
    model = create_local_model(
        logs_dir=out_dir / "llm_logs",
    )

    # Initialize browser with the chosen mode
    driver = init_browser(headless=headless)

    # Create agent
    agent = CodeAgent(
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[lambda step: save_screenshot(step, agent, base_dir=out_dir)],
        max_steps=25,
        verbosity_level=2,  # 0 = quiet, 1 = normal, 2 = verbose
        add_base_tools=False,  # True: include all helium tools
        tools=[
            go_to,
            scroll_down,
            scroll_up,
            click,
            search_item_ctrl_f,
            go_back,
            close_popups,
        ],
        # tools=ALL_TOOLS,
    )

    # Critical: preload Helium symbols into the sandboxed namespace
    agent.python_executor("from helium import *")

    print("\n" + "=" * 70)
    print("Starting agent with task:")
    print(final_task)
    print("Browser mode: " + ("HEADLESS" if headless else "VISIBLE"))
    print("=" * 70 + "\n")

    final_answer = agent.run(final_task + "\n\n" + HELIUM_GUIDE)

    print("\n" + "═" * 70)
    print("FINAL AGENT OUTPUT:")
    print(final_answer)
    print("═" * 70 + "\n")

    full_steps = agent.memory.get_full_steps()
    succinct_steps = agent.memory.get_succinct_steps()
    full_code = agent.memory.return_full_code()

    # Save outputs
    full_steps_path = out_dir / "full_steps.json"
    succinct_steps_path = out_dir / "succinct_steps.json"
    full_code_path = out_dir / "full_code.py"

    with open(full_steps_path, "w", encoding="utf-8") as f:
        json.dump(
            _strip_observations_images(make_json_serializable(full_steps)),
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(succinct_steps_path, "w", encoding="utf-8") as f:
        json.dump(
            _strip_observations_images(make_json_serializable(succinct_steps)),
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(full_code_path, "w", encoding="utf-8") as f:
        json.dump(full_code, f, ensure_ascii=False, indent=2)

    print(f"[Full steps saved] {full_steps_path}")
    print(f"[Succinct steps saved] {succinct_steps_path}")
    print(f"[Full code saved] {full_code_path}")

    # Optional: keep browser open for inspection (only in visible mode)
    if not headless:
        print("Browser will stay open for 10 seconds...")
        sleep(10)

    # Clean up
    try:
        helium.kill_browser()
    except Exception:
        pass


def cli_args():
    parser = argparse.ArgumentParser(
        description="Run browser agent with Helium + LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py                                      # default task, visible browser
  python script.py --headless                            # default task, headless
  python script.py -H                                    # same as --headless
  python script.py "Go to google.com and search for xAI" # custom task, visible
  python script.py -t "Visit example.com" -H             # custom task + headless
  python script.py -t "Visit x.ai" --headless            # custom task + headless
        """,
    )

    # Task – positional or -t / --task
    parser.add_argument(
        "task_pos",
        nargs="?",
        default=None,
        help="The task for the agent (positional – optional)",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task_opt",
        default=None,
        help="The task for the agent (alternative to positional)",
    )

    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no visible window)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory (else timestamped path will be created).",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DEFAULT_HEADLESS = False
    DEFAULT_TASK = (
        "Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence "
        'containing the word "1992" that mentions a construction accident.'
    )

    args = cli_args()

    # Resolve task: prefer -t/--task if given, otherwise use positional
    chosen_task = args.task_opt if args.task_opt is not None else args.task_pos

    main(
        headless=args.headless or DEFAULT_HEADLESS,
        task=chosen_task or DEFAULT_TASK,
        out_dir=args.out_dir or OUTPUT_DIR,
    )
