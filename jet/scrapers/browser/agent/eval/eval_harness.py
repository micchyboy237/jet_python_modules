"""
Deepeval Evaluation Harness for Playwright Browser Agent.
Reuses jet.adapters.deepeval.chat.ChatLlamaCpp for consistent LLM behavior.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from jet.adapters.deepeval.chat import ChatLlamaCpp
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL

# Reuse existing infrastructure
from jet.scrapers.browser.agent.core import LocalAgent
from jet.scrapers.browser.agent.registry import PlaywrightToolRegistry
from jet.scrapers.browser.agent.toolbox import PlaywrightToolbox
from jet.scrapers.browser.agent.utils import get_step_dir

logger = logging.getLogger(__name__)


class BrowserAgentEvaluator:
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent / "eval_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the LLM adapter for evaluation metrics
        # Using temperature=0.0 for deterministic scoring
        self.eval_model = ChatLlamaCpp(
            model=LLM_MODEL, base_url=LLM_BASE_URL, temperature=0.0, max_tokens=4096
        )

    async def run_agent_task(
        self, url: str, goal: str, max_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Runs the browser agent for a single task and returns the result.
        """
        run_dir = self.output_dir / "runs" / f"task_{hash(goal) % 10000}"
        run_dir.mkdir(parents=True, exist_ok=True)

        toolbox = PlaywrightToolbox(headless=False, output_dir=run_dir)
        registry = PlaywrightToolRegistry(toolbox)

        agent = LocalAgent(
            api_base=LLM_BASE_URL,
            model=LLM_MODEL,
            registry=registry,
            output_dir=run_dir,
        )

        try:
            await toolbox.start()

            # Navigate to initial URL
            logger.info(f"Navigating to {url}")
            await toolbox.browser_navigate(url)

            # Get initial snapshot
            step_0_dir = get_step_dir(0, run_dir)
            initial_snapshot = await toolbox.browser_snapshot(
                filename="initial_snapshot.md", step_dir=step_0_dir
            )

            # Run the agent
            logger.info(f"Starting agent for goal: {goal[:50]}...")
            final_result = await agent.think_and_act(
                goal, initial_snapshot, max_steps=max_steps
            )

            # Extract final answer if available
            final_answer_path = run_dir / "final_answer.txt"
            actual_output = ""
            if final_answer_path.exists():
                actual_output = final_answer_path.read_text()
            else:
                # Fallback: check if the agent returned a string directly
                actual_output = str(final_result)

            return {
                "success": True,
                "actual_output": actual_output,
                "goal": goal,
                "url": url,
                "run_dir": str(run_dir),
            }
        except Exception as e:
            logger.error(f"Task failed: {e}", exc_info=True)
            return {
                "success": False,
                "actual_output": f"Error: {str(e)}",
                "goal": goal,
                "url": url,
                "run_dir": str(run_dir),
            }
        finally:
            if not toolbox.is_closed:
                await toolbox.close()

    async def evaluate_tasks(
        self, tasks: List[Dict[str, str]], metrics: List[Any] = None
    ):
        """
        Runs agent on tasks and evaluates them using deepeval metrics.
        """
        if metrics is None:
            metrics = [
                AnswerRelevancyMetric(
                    threshold=0.7, model=self.eval_model, async_mode=True
                ),
                FaithfulnessMetric(
                    threshold=0.7, model=self.eval_model, async_mode=True
                ),
            ]

        results = []
        for i, task in enumerate(tasks):
            print(f"\n--- Running Task {i + 1}/{len(tasks)} ---")
            print(f"Goal: {task['goal'][:80]}...")

            run_result = await self.run_agent_task(task["url"], task["goal"])

            if run_result["success"]:
                test_case = LLMTestCase(
                    input=run_result["goal"],
                    actual_output=run_result["actual_output"],
                    expected_output=task.get("expected_output"),
                    # Optional: Pass context if you want to use ContextualRelevancyMetric
                    # context=[task.get("context", "")]
                )

                task_metrics = {}
                for metric in metrics:
                    try:
                        print(f"Evaluating {metric.__name__}...")
                        metric.measure(test_case)
                        score = metric.score
                        reason = metric.reason if hasattr(metric, "reason") else "N/A"
                        task_metrics[metric.__name__] = {
                            "score": score,
                            "reason": reason,
                            "success": score >= metric.threshold,
                        }
                        print(f"  Score: {score:.2f} (Threshold: {metric.threshold})")
                    except Exception as e:
                        print(f"  Error evaluating {metric.__name__}: {e}")
                        task_metrics[metric.__name__] = {"error": str(e)}

                results.append(
                    {
                        "task_index": i,
                        "task": task,
                        "result": run_result,
                        "metrics": task_metrics,
                    }
                )
            else:
                print(f"Task failed during execution: {run_result['actual_output']}")
                results.append(
                    {"task_index": i, "task": task, "result": run_result, "error": True}
                )

        # Save summary
        summary_path = self.output_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nEvaluation complete. Summary saved to {summary_path}")

        return results


async def main():
    evaluator = BrowserAgentEvaluator()

    # Define test tasks (Golden Tasks)
    tasks = [
        {
            "url": "https://ph.indeed.com",
            "goal": "Search for 'AI Engineer' in 'Remote' location and list the first 3 job titles.",
            "expected_output": "List of 3 job titles related to AI Engineer.",
        },
        {
            "url": "https://example.com",
            "goal": "Find the title of the page.",
            "expected_output": "Example Domain",
        },
    ]

    # Run evaluation
    await evaluator.evaluate_tasks(tasks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
