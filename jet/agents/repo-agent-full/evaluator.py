"""
Evaluation harness for repo-agent-full.
Uses LLM-as-a-Judge with Jet Telemetry for observability.
"""

import json
from pathlib import Path
from typing import Dict, List

from config import models
from jet.adapters.llama_cpp.llm_utils_observed import achat
from jet_telemetry import chain, evaluator
from utils import logger

GOLDEN_DATASET_PATH = Path(__file__).parent / "tests" / "golden_dataset.jsonl"


class EvalResult:
    def __init__(
        self,
        query: str,
        intent: str,
        expected_files: List[str],
        min_score: float,
        actual_intent: str,
        actual_files: List[str],
        relevance_score: float,
        pass_fail: bool,
    ):
        self.query = query
        self.intent = intent
        self.expected_files = expected_files
        self.min_score = min_score
        self.actual_intent = actual_intent
        self.actual_files = actual_files
        self.relevance_score = relevance_score
        self.pass_fail = pass_fail

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "expected_intent": self.intent,
            "actual_intent": self.actual_intent,
            "expected_files": self.expected_files,
            "actual_files": self.actual_files,
            "relevance_score": round(self.relevance_score, 4),
            "min_required_score": self.min_score,
            "pass": self.pass_fail,
        }


@evaluator(name="llm-relevance-judge")
async def judge_relevance(query: str, answer: str, expected_context: str) -> float:
    """
    Uses local LLM to score answer relevance on 0-1 scale.
    Returns float between 0.0 and 1.0.
    """
    prompt = f"""You are an expert evaluator. Score how well this answer addresses the query.
Query: "{query}"
Expected Context: "{expected_context}"
Answer: "{answer[:500]}..."

Return ONLY a JSON object: {{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    try:
        result = await achat(
            prompt_or_messages=[{"role": "user", "content": prompt}],
            model=models.LLM_MODEL,
            temperature=0.0,
            max_tokens=100,
            project_name="repo-agent-eval",
        )
        content = result.content.strip()
        # Extract JSON from response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(content[start:end])
            return float(parsed.get("score", 0.0))
        return 0.0
    except Exception as e:
        logger.warning(f"LLM Judge failed: {e}")
        return 0.0


@chain(name="run-golden-evaluation")
async def evaluate_golden_dataset(agent_instance, repo_path: str) -> List[EvalResult]:
    """
    Runs all golden queries through the agent and evaluates results.
    """
    if not GOLDEN_DATASET_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found: {GOLDEN_DATASET_PATH}")

    logger.info(f"Loading golden dataset from {GOLDEN_DATASET_PATH}")
    golden_queries = []
    with open(GOLDEN_DATASET_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                golden_queries.append(json.loads(line))

    results = []
    for i, gq in enumerate(golden_queries):
        logger.info(
            f"Evaluating query {i + 1}/{len(golden_queries)}: '{gq['query'][:50]}...'"
        )

        # Run agent
        agent_result = await agent_instance.run(gq["query"])
        actual_intent = agent_result.get("intent", "UNKNOWN")
        actual_answer = agent_result.get("answer", "")
        actual_sources = [s["file"] for s in agent_result.get("sources", [])]

        # Build expected context string for judge
        expected_context = (
            ", ".join(gq["expected_files"])
            if gq["expected_files"]
            else "General knowledge"
        )

        # Judge relevance
        relevance = await judge_relevance(
            query=gq["query"], answer=actual_answer, expected_context=expected_context
        )

        # Determine pass/fail
        intent_match = actual_intent == gq["intent"]
        score_pass = relevance >= gq["min_relevance_score"]
        files_overlap = (
            bool(set(actual_sources) & set(gq["expected_files"]))
            or len(gq["expected_files"]) == 0
        )

        passed = intent_match and score_pass and files_overlap

        results.append(
            EvalResult(
                query=gq["query"],
                intent=gq["intent"],
                expected_files=gq["expected_files"],
                min_score=gq["min_relevance_score"],
                actual_intent=actual_intent,
                actual_files=actual_sources,
                relevance_score=relevance,
                pass_fail=passed,
            )
        )

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(
            f"  [{status}] Intent={actual_intent}, Score={relevance:.2f}, Files={actual_sources}"
        )

    return results


def print_eval_report(results: List[EvalResult]):
    """Prints a summary report to console."""
    total = len(results)
    passed = sum(1 for r in results if r.pass_fail)
    avg_score = sum(r.relevance_score for r in results) / max(total, 1)

    print("\n" + "=" * 70)
    print("📊 EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Queries: {total}")
    print(f"Passed: {passed} ({passed / max(total, 1) * 100:.1f}%)")
    print(f"Average Relevance Score: {avg_score:.3f}")
    print("-" * 70)

    for r in results:
        icon = "✅" if r.pass_fail else "❌"
        print(f"{icon} Query: '{r.query}'")
        print(f"   Expected Intent: {r.intent} | Actual: {r.actual_intent}")
        print(f"   Relevance: {r.relevance_score:.3f} (min: {r.min_score})")
        print(f"   Sources Found: {r.actual_files}")
        print()

    print("=" * 70)
