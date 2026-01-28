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


TEXT_HELIUM_GUIDE = """
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
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = cli_args()
    task = args.task_opt or args.task_pos
    main(
        headless=args.headless,
        task=task,
        out_dir=args.out_dir,
    )
