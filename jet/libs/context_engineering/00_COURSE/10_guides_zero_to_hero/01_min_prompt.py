#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Prompt Exploration: Fundamentals of Context Engineering
==============================================================

This notebook introduces the core principles of context engineering by exploring minimal, atomic prompts and their direct impact on LLM output and behavior.

Key concepts covered:
1. Constructing atomic prompts for maximum clarity and control
2. Measuring effectiveness through token count and model response quality
3. Iterative prompt modification for rapid feedback cycles
4. Observing context drift and minimal prompt boundaries
5. Foundations for scaling from atomic prompts to protocolized shells

Usage:
    # In Jupyter or Colab:
    %run 01_min_prompt.py
    # or
    # Edit and run each section independently to experiment with prompt effects

Notes:
    - Each section of this notebook is designed for hands-on experimentation.
    - Modify prompts and observe changes in tokenization and output fidelity.
    - Use this as a foundation for building up to advanced context engineering workflows.

"""

import os
import time
import json
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import logging
import shutil
import pathlib

from jet._token.token_utils import token_counter
from jet.adapters.llama_cpp.llm import LlamacppLLM
from pydantic import BaseModel
from typing import List as TypingList
import textwrap
from datetime import datetime

OUTPUT_DIR = pathlib.Path(__file__).parent / "generated" / pathlib.Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Root for all example subfolders
EXAMPLE_OUTPUT_ROOT = OUTPUT_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = "qwen3-instruct-2507:4b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1000

def _ensure_example_dir(example_name: str) -> pathlib.Path:
    example_dir = EXAMPLE_OUTPUT_ROOT / example_name
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir

def _save_result(
    example_name: str,
    prompt: str,
    response: str,
    metadata: dict,
    plot_fig = None,
) -> None:
    dir_path = _ensure_example_dir(example_name)

    # prompt
    (dir_path / "prompt.md").write_text(f"# Prompt\n\n{prompt}\n", encoding="utf-8")

    # response
    (dir_path / "response.md").write_text(f"# Response\n\n{response}\n", encoding="utf-8")

    # metadata
    full_metadata = {
        "example": example_name,
        "timestamp": datetime.now().isoformat(),
        **metadata,
    }
    (dir_path / "metadata.json").write_text(json.dumps(full_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # plot
    if plot_fig:
        plot_path = dir_path / "roi_curve.png"
        plot_fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(plot_fig)

    # html report
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{example_name}</title></head>
<body>
<h1>{example_name.replace("_", " ").title()}</h1>
<h2>Prompt</h2><pre>{textwrap.dedent(prompt)}</pre>
<h2>Response</h2><pre>{response}</pre>
<h2>Metadata</h2><pre>{json.dumps(full_metadata, indent=2)}</pre>
{"<h2>Plot</h2><img src='roi_curve.png'/>" if plot_fig else ""}
</body></html>"""
    (dir_path / "report.html").write_text(html, encoding="utf-8")

class SimpleLLM:
    """Minimal LLM interface for demonstration."""

    def __init__(self, model_name: str = "dummy-model"):
        """Initialize LLM interface."""
        self.model_name = model_name
        self.metadata = {}

    def _raw_chat_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Tuple[str, dict]:
        client = LlamacppLLM(model=DEFAULT_MODEL, verbose=True)
        start = time.time()
        response = ""
        for chunk in client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ):
            response += chunk
        latency = time.time() - start
        prompt_tokens = token_counter(prompt, model=DEFAULT_MODEL)
        response_tokens = token_counter(response, model=DEFAULT_MODEL)
        metadata = {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens,
            "latency": latency,
            "tokens_per_second": response_tokens / latency if latency > 0 else 0,
        }
        return response, metadata

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Tuple[str, Dict[str, Any]]:
        return self._raw_chat_stream(prompt, temperature=temperature, max_tokens=max_tokens)

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return self.metadata

llm = SimpleLLM()

def example_01_atomic_prompt() -> None:
    example_name = "example_01_atomic_prompt"
    print(f"\n=== {example_name.replace('_', ' ').title()} ===")
    prompt = "Write a short poem about programming."
    response, metadata = llm.generate(prompt)
    _save_result(example_name, prompt, response, metadata)
    print(f"Saved to {EXAMPLE_OUTPUT_ROOT / example_name}")

def example_02_adding_constraints() -> None:
    example_name = "example_02_adding_constraints"
    parent_dir = _ensure_example_dir(example_name)
    print(f"\n=== {example_name.replace('_', ' ').title()} ===")

    prompts = [
        "Write a short poem about programming.",
        "Write a short poem about programming in 4 lines.",
        "Write a short haiku about programming using only simple words.",
    ]

    run_results = []
    for idx, prompt in enumerate(prompts, start=1):
        run_name = f"run_{idx:03d}"
        run_dir = parent_dir / run_name
        print(f"  → Running {run_name} ...")

        response, metadata = llm.generate(prompt)

        # Save individual run
        _save_result(f"{example_name}/{run_name}", prompt, response, metadata)

        run_results.append(
            {
                "run": run_name,
                "prompt_tokens": metadata["prompt_tokens"],
                "response_tokens": metadata["response_tokens"],
                "latency": metadata["latency"],
            }
        )
        print(f"    Saved to {run_dir.resolve()}")

    # ---- ROI plot (saved once in parent folder) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    tokens = [r["prompt_tokens"] for r in run_results]
    quality = [3, 6, 8]  # manual scores
    ax.plot(tokens, quality, "o-", color="teal")
    ax.set_xlabel("Prompt Tokens")
    ax.set_ylabel("Subjective Quality (1-10)")
    ax.set_title("Token-Quality ROI Curve")
    ax.grid(True)
    for i, (x, y) in enumerate(zip(tokens, quality)):
        ax.annotate(f"run_{i+1:03d}", (x, y), xytext=(5, 5), textcoords="offset points")

    plot_path = parent_dir / "roi_curve.png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"ROI plot saved to {plot_path}")

def example_03_minimal_context_enhancement() -> None:
    example_name = "example_03_minimal_context_enhancement"
    print(f"\n=== {example_name.replace('_', ' ').title()} ===")
    prompt = """Task: Write a haiku about programming.

A haiku is a three-line poem with 5, 7, and 5 syllables per line.

Focus on the feeling of solving a difficult bug."""
    response, metadata = llm.generate(prompt)
    _save_result(example_name, prompt, response, metadata)
    print(f"Saved to {EXAMPLE_OUTPUT_ROOT / example_name}")

def example_04_structured_stream_demo() -> None:
    example_name = "example_04_structured_stream_demo"
    print(f"\n=== {example_name.replace('_', ' ').title()} ===")

    class Haiku(BaseModel):
        line1: str
        line2: str
        line3: str

    class HaikuList(BaseModel):
        haikus: TypingList[Haiku]

    client = LlamacppLLM(model=DEFAULT_MODEL, verbose=True)
    messages = [{"role": "user", "content": "Generate 3 different programming haikus in JSON."}]
    full_response = ""
    collected: TypingList[Haiku] = []
    for haiku in client.chat_structured_stream(messages, HaikuList):
        collected.append(haiku)
        full_response += f"\n{haiku.model_dump_json(indent=2)}\n"
    metadata = {"structured_items": len(collected), "model": DEFAULT_MODEL}
    _save_result(example_name, messages[0]["content"], full_response, metadata)
    print(f"Collected {len(collected)} haikus → saved to {EXAMPLE_OUTPUT_ROOT / example_name}")

def main() -> None:
    examples = [
        example_01_atomic_prompt,
        example_02_adding_constraints,
        example_03_minimal_context_enhancement,
        example_04_structured_stream_demo,
    ]
    for ex in examples:
        ex()
    print("\nAll examples completed. Results are in separate folders under:")
    print(EXAMPLE_OUTPUT_ROOT.resolve())

if __name__ == "__main__":
    main()
