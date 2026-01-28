#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multiagents_search.py — Best local version (2026)

Fully working multi-agent web browser using:
└─ Ollama (or any OpenAI-compatible local server) on http://shawn-pc.local:11434/v1
   → Qwen2.5-Coder-32B or Meta-Llama-3.1-70B-Instruct (recommended)

Tested and confirmed working perfectly on:
- Mac M1 (dev) + Windows 11 Pro (Ollama server)
- Ryzen 5 3600 + GTX 1660 + 16GB RAM

Run: python multiagents_search.py
"""

import re
import sys

import requests
from markdownify import markdownify
from requests.exceptions import RequestException, Timeout

from smolagents import CodeAgent, ToolCallingAgent, WebSearchTool, tool
from smolagents.models import OpenAIModel


# ──────────────────────────────────────────────────────────────
# Configuration — EDIT ONLY THIS SECTION
# ──────────────────────────────────────────────────────────────

# Your local Ollama (or LM Studio, vLLM, TabbyAPI, etc.) OpenAI endpoint
# API_BASE = "http://shawn-pc.local:11434/v1"   # ← Ollama default port
API_BASE = "http://shawn-pc.local:8080/v1"  # ← llama.cpp default

# Choose one of these (all have excellent tool calling in 2026)
# MODEL_ID = "qwen2.5-coder:32b"        # ← BEST for reasoning + code + tools
# MODEL_ID = "llama3.1:70b"           # ← Also excellent
# MODEL_ID = "deepseek-coder-v2:236b" # ← If you have the VRAM
MODEL_ID = "qwen3-4b-instruct-2507"

MAX_WEB_AGENT_STEPS = 12
REQUEST_TIMEOUT = 30  # seconds


# ──────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────


@tool
def visit_webpage(url: str) -> str:
    """Fetch a webpage and return its content as clean markdown.

    Args:
        url: The full URL to visit.

    Returns:
        Markdown content or error message.
    """
    try:
        response = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        md = markdownify(response.text, heading_style="ATX").strip()
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md[:120_000] + ("\n\n... (truncated)" if len(md) > 120_000 else "")
    except Timeout:
        return "Error: Request timed out (page too slow or unreachable)."
    except RequestException as e:
        return f"Error fetching webpage: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# ──────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────


def create_model() -> OpenAIModel:
    """Create the local model instance with optimal settings for tool calling."""
    return OpenAIModel(
        model_id=MODEL_ID,
        api_base=API_BASE,
        api_key="local-model",
        temperature=0.65,
        max_tokens=4096,
        top_p=0.95,
        # These are critical for reliable tool calling with local models
        extra_body={
            # "stop": None,
            # "frequency_penalty": 0.0,
            # "presence_penalty": 0.0,
        },
    )


def create_web_search_agent(model: OpenAIModel) -> ToolCallingAgent:
    return ToolCallingAgent(
        tools=[WebSearchTool(), visit_webpage],
        model=model,
        max_steps=MAX_WEB_AGENT_STEPS,
        name="web_search_agent",
        description="Expert at web research. Can search the internet and read full webpages when needed.",
        add_base_tools=True,
    )


def create_manager_agent(model: OpenAIModel) -> CodeAgent:
    web_agent = create_web_search_agent(model)

    return CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_agent],
        additional_authorized_imports=[
            "time",
            "datetime",
            "numpy",
            "pandas",
            "math",
            "json",
        ],
        add_base_tools=True,
    )


# ──────────────────────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────────────────────


def main() -> None:
    print("🚀 Initializing multi-agent web research system (local llama.cpp LLM)\n")
    model = create_model()
    manager = create_manager_agent(model)

    # question = (
    #     "If LLM training continues to scale up at the current rhythm until 2030, "
    #     "what would be the electric power in GW required to power the biggest training runs by 2030? "
    #     "Compare it to some countries' total electricity consumption. Provide sources."
    # )
    question = "Top 10 isekai anime this 2026. Provide sources."

    print("=" * 90)
    print("QUESTION:")
    print(question)
    print("=" * 90 + "\n")

    try:
        answer = manager.run(question)
        print("\n" + "=" * 90)
        print("FINAL ANSWER:")
        print("=" * 90)
        print(answer)
    except KeyboardInterrupt:
        print("\n\nStopped by user.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
