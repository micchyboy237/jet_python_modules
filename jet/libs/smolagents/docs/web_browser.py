# demo_web_browser_automation_local.py
"""
Web browser automation demos using Helium + Selenium + smolagents
Adapted for LOCAL text-only llama.cpp model (no vision support)

Limitations:
- Your local model cannot see screenshots → agent relies only on text observations
- Screenshot callback still runs (for logging), but images are NOT attached to memory
"""

import time
from io import BytesIO
from typing import Optional

import helium
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from rich.console import Console
from rich.panel import Panel

from smolagents import CodeAgent, tool, ActionStep, OpenAIModel

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Reuse your local model factory
# ──────────────────────────────────────────────────────────────────────────────

def create_local_model(
    temperature: float = 0.7,
    max_tokens: Optional[int] = 2048,
    model_id: str = "local-model",
) -> OpenAIModel:
    return OpenAIModel(
        model_id=model_id,
        base_url="http://shawn-pc.local:8080/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Browser setup (global driver)
# ──────────────────────────────────────────────────────────────────────────────

def init_browser(headless: bool = False):
    """Initialize Chrome via Helium with reasonable options."""
    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=1200,900")
    options.add_argument("--disable-notifications")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    if headless:
        options.add_argument("--headless=new")

    driver = helium.start_chrome(headless=headless, options=options)
    console.print(f"[dim]Browser initialized (headless={headless})[/dim]")
    return driver


# Global driver (lazy init in first demo)
DRIVER = None


# ──────────────────────────────────────────────────────────────────────────────
# Tools (same as tutorial, but typed & robust)
# ──────────────────────────────────────────────────────────────────────────────

@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """Search for text on page using XPath and scroll to nth match."""
    global DRIVER
    elements = DRIVER.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if not elements:
        return f"No matches found for '{text}'"
    if nth_result > len(elements):
        return f"Only {len(elements)} matches found for '{text}' (requested #{nth_result})"

    elem = elements[nth_result - 1]
    DRIVER.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
    return f"Focused on match {nth_result}/{len(elements)} for '{text}'"


@tool
def go_back() -> str:
    """Navigate to previous page."""
    global DRIVER
    DRIVER.back()
    time.sleep(0.8)
    return f"Navigated back. Current URL: {DRIVER.current_url}"


@tool
def close_popups() -> str:
    """Try to close modal/popups with ESC key."""
    global DRIVER
    webdriver.ActionChains(DRIVER).send_keys(Keys.ESCAPE).perform()
    time.sleep(0.6)
    return "Sent ESC to attempt closing popups"


# ──────────────────────────────────────────────────────────────────────────────
# Screenshot callback (text-only version)
# ──────────────────────────────────────────────────────────────────────────────

def log_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    """Log screenshot info & current URL (no image attachment - text model)"""
    global DRIVER
    time.sleep(1.2)  # wait for page to settle

    if DRIVER is None:
        return

    try:
        url = DRIVER.current_url
        size = DRIVER.get_window_size()
        png_bytes = DRIVER.get_screenshot_as_png()
        img = Image.open(BytesIO(png_bytes))
        console.print(
            f"[dim]Step {memory_step.step_number:2d} | "
            f"URL: {url[:90]}{'...' if len(url) > 90 else ''} | "
            f"Screen: {size['width']}×{size['height']} | "
            f"Screenshot taken ({img.size[0]}×{img.size[1]} px)[/dim]"
        )

        # Still update observations with URL
        url_line = f"Current URL: {url}"
        memory_step.observations = (
            url_line if memory_step.observations is None
            else memory_step.observations + "\n" + url_line
        )

        # Optional: keep last 1–2 screenshots if you later add vision model
        # for now we don't attach to avoid serialization issues
        memory_step.observations_images = None

    except Exception as e:
        console.print(f"[yellow]Screenshot failed: {str(e)}[/yellow]")


# ──────────────────────────────────────────────────────────────────────────────
# Agent factory
# ──────────────────────────────────────────────────────────────────────────────

HELIUM_INSTRUCTIONS = """
You can control the browser using Helium commands (already imported: from helium import *).

Important commands:
- go_to('https://example.com')           → navigate
- click("Sign in")                       → click visible text
- click(Link("Blog"))                    → click link by text
- write("search term", into="Search")    → type into field
- scroll_down(num_pixels=800)            → scroll
- S("#search-input").value               → get element value (CSS selector)
- Text("Welcome").exists()               → check existence

Rules:
- After each click / navigation, wait and observe result via text feedback.
- Use close_popups() for modals.
- Never attempt real logins.
- Use search_item_ctrl_f("text") to find & focus text.
- Prefer exact text matches when clicking.
"""


def create_browser_agent(max_steps: int = 18, verbosity_level: int = 2) -> CodeAgent:
    """Create CodeAgent configured for browser automation (text-only feedback)."""
    global DRIVER
    if DRIVER is None:
        DRIVER = init_browser(headless=False)  # change to True if preferred

    model = create_local_model(temperature=0.75)

    agent = CodeAgent(
        tools=[search_item_ctrl_f, go_back, close_popups],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[log_screenshot],
        max_steps=max_steps,
        verbosity_level=verbosity_level,
    )

    # Pre-load Helium
    agent.python_executor("from helium import *", agent.state)

    return agent


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────

def demo_browser_1_wikipedia_find():
    """Demo 1: Find sentence on Wikipedia page"""
    console.rule("Demo 1: Wikipedia – find 1992 construction accident", style="blue")

    agent = create_browser_agent(max_steps=12)

    task = (
        "Navigate to https://en.wikipedia.org/wiki/Chicago "
        "and find a sentence containing the word '1992' "
        "that mentions a construction accident or similar event. "
        "Quote the full sentence."
    )

    console.print(f"\n[bold cyan]Task:[/bold cyan]\n{task}")
    start = time.time()

    try:
        result = agent.run(task + HELIUM_INSTRUCTIONS)
        console.print(Panel(result, title="Final Answer", border_style="green"))
    except Exception as e:
        console.print(f"[red]Agent failed:[/red] {str(e)}")
    finally:
        console.print(f"[dim]Duration: {time.time() - start:.1f}s[/dim]")


def demo_browser_2_github_trending_author():
    """Demo 2: Find top trending repo author stats (harder)"""
    console.rule("Demo 2: GitHub trending – author commit stats", style="blue")

    agent = create_browser_agent(max_steps=20, verbosity_level=2)

    task = (
        "Go to https://github.com/trending "
        "Click on the top repository. "
        "Go to the repository owner’s profile. "
        "Tell me approximately how many commits they made in the last year "
        "(look for contribution graph or activity summary)."
    )

    console.print(f"\n[bold cyan]Task:[/bold cyan]\n{task}")
    start = time.time()

    try:
        result = agent.run(task + HELIUM_INSTRUCTIONS)
        console.print(Panel(result, title="Final Answer", border_style="green"))
    except Exception as e:
        console.print(f"[red]Agent failed:[/red] {str(e)}")
    finally:
        console.print(f"[dim]Duration: {time.time() - start:.1f}s[/dim]")


def main():
    console.rule("Web Browser Automation Demos – LOCAL text-only model", style="bold magenta")

    console.print(
        "[yellow]Note:[/yellow] Local llama.cpp is text-only → no screenshot vision feedback\n"
        "[dim]Agent navigates using Helium commands + text observations only[/dim]\n"
    )

    # Choose which demo to run
    demo_browser_1_wikipedia_find()
    # demo_browser_2_github_trending_author()

    console.rule("Finished", style="bold green")

    # Optional: clean up browser
    # if DRIVER is not None:
    #     helium.kill_browser()
    #     console.print("[dim]Browser closed[/dim]")


if __name__ == "__main__":
    main()