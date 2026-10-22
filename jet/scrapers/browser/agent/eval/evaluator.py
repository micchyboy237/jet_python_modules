"""
Browser Agent Evaluator Wrapper.
Provides a simple interface to trigger the Deepeval harness.
"""

import asyncio
from pathlib import Path

from jet.scrapers.browser.agent.eval_harness import BrowserAgentEvaluator


async def run_evaluation(golden_tasks_path: Path = None):
    """
    Run evaluation using golden tasks from a JSON file.
    """
    evaluator = BrowserAgentEvaluator()

    if golden_tasks_path and golden_tasks_path.exists():
        import json

        with open(golden_tasks_path, "r") as f:
            # Handle both single task object and list of tasks
            data = json.load(f)
            tasks = data if isinstance(data, list) else [data]
    else:
        # Default fallback tasks if no file provided
        tasks = [
            {
                "url": "https://example.com",
                "goal": "What is the title of this page?",
                "expected_output": "Example Domain",
            }
        ]

    return await evaluator.evaluate_tasks(tasks)


if __name__ == "__main__":
    golden_path = Path(__file__).parent / "tests" / "agent_golden_tasks.json"
    asyncio.run(run_evaluation(golden_path))
