"""
Examples showing different ways to use the save_step_state callback.
"""

import shutil
from pathlib import Path
from typing import Any

# ─── Imports ────────────────────────────────────────────────────────────────
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.step_callbacks.save_step_state import save_step_state
from smolagents import CodeAgent, Tool

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOOLS_RUN_DIR = OUTPUT_DIR / "agent_tool_runs"
LLM_CALLS_DIR = OUTPUT_DIR / "llm_calls"

# ─── Dummy lookup tool ──────────────────────────────────────────────────────


class DummyLookupTool(Tool):
    name = "dummy_lookup"
    description = "Returns fake lookup results (demo only)"
    output_type = "string"

    inputs: dict[str, dict[str, Any]] = {
        "query": {
            "type": "string",
            "description": "The query string to look up",
        }
    }

    def forward(self, query: str) -> str:
        return f"Mock external result for query: {query}"


dummy_tool_instance = DummyLookupTool()
# or: DummyLookupTool()  # no arguments needed — everything is class-level


# ─── Local model factory (reused across all demos) ──────────────────────────
def create_local_model(
    temperature: float = 0.4,
    max_tokens: int | None = 4096,
    model_id: LLAMACPP_LLM_KEYS | None = None,
    logs_dir: str | Path | None = None,
) -> OpenAIModel:
    if model_id is None:
        model_id = "qwen3-instruct-2507:4b"

    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        logs_dir=logs_dir,
    )


# ─── Demo Functions ─────────────────────────────────────────────────────────


def demo_basic_usage():
    """
    Classic pattern: one callback reused for one agent
    Most common way when developing / debugging a single agent
    """
    base_dir = TOOLS_RUN_DIR / "demo_basic_usage"
    llm_out_dir = LLM_CALLS_DIR / "demo_basic_usage"

    callback = save_step_state(
        agent_name="math_reasoner",
        base_dir=base_dir,
        save_images=False,
    )

    model = create_local_model(
        temperature=0.3,
        max_tokens=2048,
        logs_dir=llm_out_dir,
        # model_id="llama-3.1-8b-instruct-q5_k_m",  # ← uncomment & change if needed
    )

    agent = CodeAgent(
        model=model,
        tools=[dummy_tool_instance],  # now has .name and .inputs
        step_callbacks=[callback],
        verbosity_level=1,
    )

    print("\nRunning basic math demo...")
    agent.run("What is the prime factorization of 2025?")
    # → files like 0001_plan_math_reasoner.json, 0002_action_math_reasoner.json, ...


def demo_inline_notebook_style():
    """Quick setup — ideal for Jupyter cells or short test scripts"""
    base_dir = TOOLS_RUN_DIR / "demo_inline_notebook_style"
    llm_out_dir = LLM_CALLS_DIR / "demo_inline_notebook_style"

    model = create_local_model(temperature=0.7, logs_dir=llm_out_dir)

    agent = CodeAgent(
        model=model,
        tools=[dummy_tool_instance],
        step_callbacks=[
            save_step_state(
                agent_name="fast_debug",
                base_dir=base_dir,
                save_images=False,
            )
        ],
        verbosity_level=1,
    )

    agent.run("Why can't you divide by zero?")


def demo_multiple_agents_same_run_dir():
    """
    Multiple agents sharing the same base directory → global step numbering
    Useful when comparing agent personalities, prompts, temperatures, etc.
    """
    base_dir = TOOLS_RUN_DIR / "demo_multiple_agents_same_run_dir"
    llm_out_dir_low = LLM_CALLS_DIR / "demo_multiple_agents_same_run_dir" / "low_temp"
    llm_out_dir_high = LLM_CALLS_DIR / "demo_multiple_agents_same_run_dir" / "high_temp"

    model_low_temp = create_local_model(temperature=0.2, logs_dir=llm_out_dir_low)
    model_high_temp = create_local_model(temperature=0.9, logs_dir=llm_out_dir_high)

    cb_research = save_step_state("research_style", base_dir=base_dir)
    cb_concise = save_step_state("concise_style", base_dir=base_dir)

    research_agent = CodeAgent(
        model=model_high_temp,
        tools=[dummy_tool_instance],
        step_callbacks=[cb_research],
    )

    concise_agent = CodeAgent(
        model=model_low_temp,
        tools=[dummy_tool_instance],
        step_callbacks=[cb_concise],
    )

    print("\nComparing reasoning styles...")
    research_agent.run("Explain how a CPU works in simple terms.")
    concise_agent.run("Explain how a CPU works in simple terms.")


def demo_experiment_style_with_images():
    """
    Per-experiment folder + save_images=True
    (useful when your tools can return images or plots)
    """
    base_dir = TOOLS_RUN_DIR / "demo_experiment_style_with_images"
    llm_out_dir = LLM_CALLS_DIR / "demo_experiment_style_with_images"

    callback = save_step_state(
        agent_name="detailed_thinker",
        base_dir=base_dir,
        save_images=True,  # ← enable if agent/tools produce images
    )

    model = create_local_model(
        temperature=0.5,
        max_tokens=4096,
        logs_dir=llm_out_dir,
    )

    agent = CodeAgent(
        model=model,
        tools=[dummy_tool_instance],
        step_callbacks=[callback],
    )

    agent.run("Describe step by step how rainbows are formed.")


def demo_silent_background_run():
    """No extra prints from the callback — clean logs"""
    base_dir = TOOLS_RUN_DIR / "demo_silent_background_run"
    llm_out_dir = LLM_CALLS_DIR / "demo_silent_background_run"

    quiet_callback = save_step_state(
        agent_name="silent_worker",
        base_dir=base_dir,
        save_images=False,
    )

    model = create_local_model(temperature=0.1, logs_dir=llm_out_dir)

    agent = CodeAgent(
        model=model,
        tools=[dummy_tool_instance],
        step_callbacks=[quiet_callback],
        verbosity_level=2,  # still no print from save_step_state itself
    )

    agent.run("Compute the 12th triangular number.")


# ─── Launcher / Discovery ───────────────────────────────────────────────────

if __name__ == "__main__":
    from time import perf_counter

    print("Demos using local OpenAI-compatible model (llama.cpp server required)\n")
    print("  1 = demo_basic_usage()")
    print("  2 = demo_inline_notebook_style()")
    print("  3 = demo_multiple_agents_same_run_dir()")
    print("  4 = demo_experiment_style_with_images()")
    print("  5 = demo_silent_background_run()")
    print()
    print("  (press Enter or type nothing → run ALL demos)")
    print()

    choice = input("Which demo to run? (1–5 or Enter for all) → ").strip()

    demos = [
        ("basic_usage", demo_basic_usage),
        ("inline_notebook_style", demo_inline_notebook_style),
        ("multiple_agents_same_dir", demo_multiple_agents_same_run_dir),
        ("experiment_style_images", demo_experiment_style_with_images),
        ("silent_background", demo_silent_background_run),
    ]

    if choice in {"1", "2", "3", "4", "5"}:
        idx = int(choice) - 1
        name, func = demos[idx]
        print(f"\n→ Running only {name} ...\n")
        func()
    else:
        print("\n→ Running ALL demos ...\n")
        total_start = perf_counter()

        for i, (name, func) in enumerate(demos, 1):
            start = perf_counter()
            print(f"[{i}/{len(demos)}] {name}")
            func()
            duration = perf_counter() - start
            print(f"   → completed in {duration:.1f}s\n")

        total_duration = perf_counter() - total_start
        print(f"All demos finished in {total_duration:.1f} seconds.")
