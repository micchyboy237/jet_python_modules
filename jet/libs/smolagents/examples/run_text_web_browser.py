# run_text_web_browser.py
"""
Text-only web browser agent using Helium + clean page text extraction
No vision / screenshots required
"""

import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime

from typing import Optional

from smolagents import CodeAgent, tool
from jet.libs.smolagents.docs.web_browser import (
    cli_args,
    init_browser,
    create_local_model,
    search_item_ctrl_f,
    make_json_serializable,
)
from jet.libs.smolagents.helium_tools import (
    go_to,
    scroll_down,
    scroll_up,
    click,
    go_back,
    close_popups,
)

# ────────────────────────────────────────────────
#  Reuse most logic from web_browser.py
# ────────────────────────────────────────────────


def get_raw_page_text(max_chars: int = 12000) -> str:
    """Extract current page URL and visible body text (limited length)."""
    import helium
    from selenium.webdriver.common.by import By

    driver = helium.get_driver()
    if not driver:
        return "No active browser session."
    try:
        url = driver.current_url
        body_text = driver.find_element(By.TAG_NAME, "body").text.strip()
        truncated = body_text[:max_chars]
        if len(body_text) > max_chars:
            truncated += (
                f"\n\n[... {len(body_text) - max_chars:,} characters truncated ...]"
            )
        return f"Current URL: {url}\n\nVisible page text:\n{truncated}"
    except Exception as e:
        return f"Failed to extract page text: {str(e)}"


@tool
def summarize_observation(
    summary_focus: str = "key facts, numbers, names, dates, important links",
) -> str:
    """
    Create a concise summary of the current page.
    Use this when the full observation is too long or you want to focus on specific information.

    Args:
        summary_focus: What to prioritize in the summary (default: key facts, numbers, names, dates, important links)
    """
    raw_text = get_raw_page_text(max_chars=8000)
    # For now we return raw truncated text — later you can call LLM here if you want
    # But to keep it lightweight, agent will summarize via its own reasoning
    return (
        f"[Page Summary Request]\nFocus: {summary_focus}\n\n"
        f"{raw_text}\n\nPlease provide a concise summary (150–400 words) keeping only the most relevant information."
    )


def save_page_observation(memory_step, agent, base_dir: Path):
    import time

    time.sleep(1.1)
    step_num = memory_step.step_number

    obs = get_raw_page_text(max_chars=9000)  # still generous but safer
    dir_obs = base_dir / "observations"
    dir_obs.mkdir(exist_ok=True, parents=True)
    (dir_obs / f"step_{step_num:03d}.txt").write_text(obs, encoding="utf-8")

    print(f"[Text obs saved] step {step_num:03d}")

    memory_step.observations = obs
    memory_step.observations_images = None


TEXT_HELIUM_GUIDE = """
You can use helium to access websites. Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"

Important memory & context rules (VERY IMPORTANT):
• Page observations can be very long → NEVER copy the full text into your thoughts
• After go_to(), click(), scroll_... → you receive page text automatically
• If the observation is long/noisy → call summarize_observation() to get a focused summary
• Keep your own thoughts very short and concise
• After finding important information, write a brief "Memory note" (2–5 sentences max)
  Example:
  thought: Important: The Chicago construction accident in 1992 killed 3 workers at the ...
  code:
      print("MEMORY: 1992 Chicago crane collapse at 123 N Wacker Dr killed 3, injured 8")
• Only reference previous memory notes or summaries — do NOT repeat large past observations
• Goal: keep total conversation length short so we stay under context limit

Then you can go to pages!
Code:
```py
go_to('github.com/trending')
``` <end_code>

You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
``` <end_code>

If it's a link:
Code:
```py
click(Link("Top products"))
``` <end_code>

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


def trim_agent_memory(agent, max_action_steps_to_keep: int = 10):
    """Keep system/task messages + last N action steps to prevent context overflow."""
    from smolagents.agents import ActionStep

    important = [s for s in agent.memory.steps if not isinstance(s, ActionStep)]
    actions = [s for s in agent.memory.steps if isinstance(s, ActionStep)]

    if len(actions) > max_action_steps_to_keep:
        new_actions = actions[-max_action_steps_to_keep:]
        agent.memory.steps = important + new_actions
        print(f"[Memory trimmed] {len(actions)} → {len(new_actions)} action steps kept")


def main(
    headless: bool = True,
    task: str | None = None,
    out_dir: Path | None = None,
):
    if out_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_dir = Path(__file__).parent / "generated" / Path(__file__).stem

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}")

    default_task = (
        "Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence "
        'containing the word "1992" that mentions a construction accident.'
    )
    task = (task or default_task).strip()

    model = create_local_model(
        temperature=0.3,
        max_tokens=4092,
        logs_dir=out_dir / "llm_logs",
    )

    driver = init_browser(headless=headless)

    agent = CodeAgent(
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[
            lambda step: save_page_observation(step, agent, out_dir),
            lambda step: trim_agent_memory(agent, max_action_steps_to_keep=10),
        ],
        max_steps=20,
        verbosity_level=2,
        add_base_tools=False,
        tools=[
            go_to,
            scroll_down,
            scroll_up,
            click,
            go_back,
            close_popups,
            search_item_ctrl_f,
            summarize_observation,  # ← new tool
            # write, select, ... can be added back if needed
        ],
    )

    agent.python_executor("from helium import *")

    print("\n" + "═" * 70)
    print("TEXT-ONLY Browser Agent")
    print("Task:", task)
    print("Browser:", "HEADLESS" if headless else "VISIBLE")
    print("═" * 70 + "\n")

    final = agent.run(task + "\n\n" + TEXT_HELIUM_GUIDE)

    print("\n" + "═" * 70)
    print("FINAL ANSWER:")
    print(final)
    print("═" * 70)

    # Save artifacts (same as original)
    # ... (copy from original main() if needed)

    if not headless:
        print("Browser stays open 15 seconds...")
        import time

        time.sleep(15)

    try:
        import helium

        helium.kill_browser()
    except:
        pass


if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = cli_args()
    task = args.task_opt or args.task_pos
    main(
        headless=args.headless,
        task=task,
        out_dir=args.out_dir or OUTPUT_DIR,
    )
