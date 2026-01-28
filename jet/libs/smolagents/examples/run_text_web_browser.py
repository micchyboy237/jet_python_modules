# run_text_web_browser.py
"""
Text-only web browser agent with BM25-powered memory retrieval
Uses rank_bm25 to index past page observations for efficient recall
"""

import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime
from typing import List, Dict, Optional

from smolagents import CodeAgent, tool
from jet.libs.smolagents.docs.web_browser import (
    cli_args,
    init_browser,
    create_local_model,
    search_item_ctrl_f,
)
from jet.libs.smolagents.helium_tools import (
    go_to,
    scroll_down,
    scroll_up,
    click,
    go_back,
    close_popups,
)

# ← NEW: Import rank_bm25
from rank_bm25 import BM25Okapi


# ────────────────────────────────────────────────
# Global BM25 Memory Index (updated live)
# ────────────────────────────────────────────────


class MemoryIndex:
    def __init__(self):
        self.documents: List[
            Dict[str, str]
        ] = []  # Each: {"step": int, "url": str, "text": str}
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    def add_observation(self, step: int, url: str, text: str):
        # Simple chunking: split long texts into ~400-token chunks
        max_chunk_chars = 1600
        chunks = [
            text[i : i + max_chunk_chars]
            for i in range(0, len(text), max_chunk_chars - 200)
        ]

        for i, chunk in enumerate(chunks):
            doc = {
                "step": step,
                "chunk_id": i,
                "url": url,
                "preview": chunk.strip()[:140] + "..."
                if len(chunk) > 140
                else chunk.strip(),
                "text": chunk.strip(),
            }
            self.documents.append(doc)
            self.tokenized_corpus.append(chunk.lower().split())

        # Rebuild index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> str:
        if not self.bm25 or not query.strip():
            return "No memory index available yet."

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        if not top_indices:
            return "No relevant past observations found."

        results = "\n\n=== Relevant Past Observations ===\n"
        for idx in top_indices:
            doc = self.documents[idx]
            results += (
                f"[Step {doc['step']:03d} | Chunk {doc['chunk_id']} | From: {doc['url']}]\n"
                f"{doc['text']}\n"
                f"{'─' * 60}"
            )
        return results + "\n"


# Global instance
MEMORY_INDEX = MemoryIndex()


def get_raw_page_text(max_chars: int = 12000) -> str:
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
                f"\n\n[Truncated: {len(body_text) - max_chars:,} chars omitted]"
            )
        return f"Current URL: {url}\n\nVisible page text:\n{truncated}"
    except Exception as e:
        return f"Failed to extract page text: {str(e)}"


@tool
def summarize_observation(
    summary_focus: str = "key facts, numbers, names, dates, important links",
) -> str:
    """
    Create a concise summary of the current page content.
    Use this tool when the full page observation is too long or contains too much noise
    and you want to focus on specific kinds of information.

    This tool returns the current page text prefixed with instructions — the LLM
    will then produce the actual summary in the next step.

    Args:
        summary_focus: The specific aspects to prioritize in the summary.
                       Example: "dates, casualty numbers, names of people and locations"
                       Default focuses on key facts, numbers, names, dates, important links.
    """
    raw_text = get_raw_page_text(max_chars=9000)
    return (
        f"[Page Summary Request]\nFocus: {summary_focus}\n\n"
        f"{raw_text}\n\n"
        "Please provide a concise summary (150–400 words) keeping only the most "
        "relevant information according to the focus above."
    )


@tool
def retrieve_from_memory(query: str) -> str:
    """
    Search past page observations using keyword/BM25 retrieval.
    Use this to recall previously seen information (e.g., names, dates, events).

    Args:
        query: Natural language or keyword query (e.g., "Chicago construction accident 1992")
    """
    return MEMORY_INDEX.retrieve(query, top_k=6)


def save_page_observation(memory_step, agent, base_dir: Path):
    import time

    time.sleep(1.1)
    step_num = memory_step.step_number

    obs = get_raw_page_text(max_chars=8000)
    url = "unknown"
    try:
        import helium

        url = helium.get_driver().current_url
    except:
        pass

    # Save to disk
    dir_obs = base_dir / "observations"
    dir_obs.mkdir(exist_ok=True, parents=True)
    (dir_obs / f"step_{step_num:03d}.txt").write_text(obs, encoding="utf-8")

    # Add to BM25 memory index
    MEMORY_INDEX.add_observation(step=step_num, url=url, text=obs)

    # Attach to memory step (light version only for context)
    short_obs = obs[:3000] + ("\n\n[Truncated for context]" if len(obs) > 3000 else "")
    memory_step.observations = short_obs
    memory_step.observations_images = None

    obs_len = len(obs)
    est_tokens = obs_len // 4
    print(
        f"[Obs saved] Step {step_num:03d} | {obs_len:,} chars ≈ {est_tokens:,} tokens | Indexed in BM25"
    )


TEXT_HELIUM_GUIDE = """
You are a text-only web browser agent with memory retrieval.

Key tools:
- go_to(url), click(text), scroll_down(), close_popups(), etc.
- retrieve_from_memory(query): Search all past page observations (very useful!)
- summarize_observation(): When current page is too long

Memory strategy:
• After important pages, use retrieve_from_memory("key topic") to confirm recall
• Never repeat full observations — use retrieval instead
• Keep thoughts short. One line.
• When you find the answer, extract and remember it clearly.

Example:
thought: Need to recall any mention of 1992 accidents in Chicago
code:
retrieve_from_memory("Chicago construction accident 1992 crane collapse")

Always end with a clear final answer when task is complete.
""".strip()


def trim_agent_memory(agent, max_total_steps: int = 6):
    from smolagents.agents import ActionStep

    if len(agent.memory.steps) > max_total_steps:
        old_count = len(agent.memory.steps)
        agent.memory.steps = agent.memory.steps[-max_total_steps:]
        print(
            f"[Memory trimmed] Steps: {old_count} → {len(agent.memory.steps)} (kept last {max_total_steps})"
        )


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
            lambda step: trim_agent_memory(agent, max_total_steps=6),
        ],
        max_steps=25,
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
            summarize_observation,
            retrieve_from_memory,  # ← Critical new tool
        ],
    )

    agent.python_executor("from helium import *")

    print("\n" + "═" * 80)
    print("TEXT-ONLY Browser Agent + BM25 Memory Retrieval".center(80))
    print("Task:".center(80))
    print(task.center(80))
    print("═" * 80 + "\n")

    final = agent.run(task + "\n\n" + TEXT_HELIUM_GUIDE)

    print("\n" + "═" * 80)
    print("FINAL ANSWER:")
    print(final)
    print("═" * 80)

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
