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

from smolagents import CodeAgent
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


def extract_page_text() -> str:
    from selenium.webdriver.common.by import By
    import helium

    driver = helium.get_driver()
    if not driver:
        return "No browser session."
    try:
        url = driver.current_url
        body_text = driver.find_element(By.TAG_NAME, "body").text.strip()
        return f"URL: {url}\n\nVisible text content:\n{body_text[:15000]}"
    except Exception as e:
        return f"Text extraction failed: {str(e)}"


def save_page_observation(memory_step, agent, base_dir: Path):
    import time

    time.sleep(1.1)
    step_num = memory_step.step_number

    obs = extract_page_text()

    dir_obs = base_dir / "observations"
    dir_obs.mkdir(exist_ok=True, parents=True)
    (dir_obs / f"step_{step_num:03d}.txt").write_text(obs, encoding="utf-8")

    print(f"[Text obs saved] step {step_num:03d}")

    memory_step.observations = obs
    memory_step.observations_images = None


TEXT_HELIUM_GUIDE = """You are controlling a web browser using Helium commands.
You do NOT see screenshots — only extracted visible TEXT is provided each step.

→ Read the "Visible text content" section carefully.
→ Links, buttons, forms, menus are described in text.
→ If you need more content → scroll_down(800), click, etc.
→ After every page-changing action you get fresh text.

Important commands:
go_to('https://...')
click("Button text")
click(Link("Link text"))
write("query", into="Search")
scroll_down(1200)
scroll_up(800)
close_popups()

Do NOT attempt logins.
""".strip()


def main(
    headless: bool = True,
    task: str | None = None,
    out_dir: Path | None = None,
):
    if out_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_dir = Path(__file__).parent / "generated" / "text_web_browser" / f"run_{ts}"

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}")

    default_task = (
        "Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence "
        'containing the word "1992" that mentions a construction accident.'
    )
    task = (task or default_task).strip()

    model = create_local_model(
        temperature=0.4,
        max_tokens=3800,
        logs_dir=out_dir / "llm_logs",
    )

    driver = init_browser(headless=headless)

    agent = CodeAgent(
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[lambda step: save_page_observation(step, agent, out_dir)],
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
    OUTPUT_DIR = Path(__file__).parent / "generated" / "text_web_browser"
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = cli_args()
    task = args.task_opt or args.task_pos
    main(
        headless=args.headless,
        task=task,
        out_dir=args.out_dir,
    )
