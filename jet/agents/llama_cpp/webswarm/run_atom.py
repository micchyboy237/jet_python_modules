import asyncio
import json
import logging
import os
import sys

# Add this file's directory to sys.path for module imports
module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_dir)
os.chdir(module_dir)

from config import AtomConfig
from graphs.atom_graph import build_atom_graph
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


async def run_atom(task: str, config: AtomConfig | None = None) -> dict:
    config = config or AtomConfig()
    graph = build_atom_graph(config)

    initial_state = {
        "task": task,
        "messages": [
            SystemMessage(content="You are an atomic fact-finding agent."),
            HumanMessage(content=task),
        ],
        "step_count": 0,
        "max_steps": config.atom_max_steps,
        "result": None,
        "is_complete": False,
    }

    logger.info(f"Starting Atom agent for: {task[:100]}...")
    final_state = await graph.ainvoke(initial_state)

    # Parse structured result from final message
    result = final_state.get("result")
    if result is None:
        last_content = final_state["messages"][-1].content
        try:
            # Extract JSON block from response
            start = last_content.index("{")
            end = last_content.rindex("}") + 1
            result = json.loads(last_content[start:end])
        except (ValueError, json.JSONDecodeError):
            result = {
                "answer": last_content,
                "sources": [],
                "confidence": 0.5,
                "reason": "Unstructured response",
            }

    logger.info(
        f"Atom completed in {final_state['step_count']} steps. Confidence: {result.get('confidence')}"
    )
    return {
        "result": result,
        "steps": final_state["step_count"],
        "messages": len(final_state["messages"]),
    }


# --- Quick Validation ---
if __name__ == "__main__":
    test_tasks = [
        "What year was the Eiffel Tower completed?",
        "Who is the current CEO of NVIDIA as of 2025?",
        "What is the IATA code for Tokyo Haneda Airport?",
    ]

    async def validate():
        config = AtomConfig()
        for task in test_tasks:
            output = await run_atom(task, config)
            print(f"\n{'=' * 60}")
            print(f"Task: {task}")
            print(f"Answer: {output['result'].get('answer')}")
            print(f"Sources: {output['result'].get('sources')}")
            print(f"Steps: {output['steps']} | Messages: {output['messages']}")

    asyncio.run(validate())
