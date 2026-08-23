# jet_python_modules/jet/adapters/llama_cpp/tasks/rag/react_web_searcher/main.py
"""
ReAct Web Searcher CLI

Title:
    ReAct Web Search Agent Runner

Description:
    Command-line interface for the ReAct web search pipeline. Executes
    autonomous multi-step research using SearXNG search, URL reading,
    and synthesis tools. Optionally validates the final answer against
    collected evidence using the LLM-as-a-Judge evaluation pipeline.
    
    Supports an analyze-only mode that previews query decomposition
    without executing the full search loop.

Args Description:
    query               User question to research (positional)
    --model             LLM model for reasoning and tool use (default: qwen3.5-uncensored:2b)
    --max-iterations    Maximum ReAct loop iterations (default: 10)
    --no-validation     Disable post-answer faithfulness/hallucination checking
    --analyze-only      Only run query analysis and exit (no search execution)
    --json              Output results as JSON instead of human-readable format

Usage Examples:
    # Simple factual search
    python main.py "What is the capital of France?"

    # Complex research with extended iteration budget
    python main.py "Compare renewable energy policies in EU vs US 2024" \\
        --max-iterations 15 --model qwen3.5-uncensored:2b

    # Preview query decomposition without searching
    python main.py "Explain CRISPR gene editing applications" --analyze-only

    # Search without post-validation (faster, less safe)
    python main.py "Latest SpaceX launch schedule" --no-validation --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Sequence

from jet.adapters.llama_cpp.tasks.rag.react_web_searcher import (
    QueryAnalyzer,
    ReactEngine,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="react-search",
        description="Run ReAct web search agent for autonomous research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        help="User question to research",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5-uncensored:2b",
        help="LLM model for reasoning and tool use (default: qwen3.5-uncensored:2b)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum ReAct loop iterations (default: 10)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        default=False,
        help="Disable post-answer faithfulness/hallucination checking",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        default=False,
        help="Only run query analysis and exit (no search execution)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON instead of human-readable format",
    )
    return parser


async def run_analyze_only(args: argparse.Namespace) -> None:
    analyzer = QueryAnalyzer(model=args.model)
    analysis = await analyzer.analyze(args.query)

    if args.json:
        output = {
            "mode": "analyze_only",
            "query": args.query,
            "complexity": analysis.complexity.value,
            "reasoning": analysis.reasoning,
            "refined_query": analysis.refined_query,
            "sub_queries": analysis.sub_queries,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print(f"Query Analysis (analyze-only mode)")
        print(f"{'=' * 60}")
        print(f"Original Query:  {args.query}")
        print(f"Complexity:      {analysis.complexity.value}")
        print(f"Reasoning:       {analysis.reasoning}")
        print(f"Refined Query:   {analysis.refined_query}")
        if analysis.sub_queries:
            print(f"Sub-queries ({len(analysis.sub_queries)}):")
            for i, sq in enumerate(analysis.sub_queries, 1):
                print(f"  {i}. {sq}")
        else:
            print("Sub-queries:     None (simple query)")
        print(f"{'=' * 60}\n")


async def run_search(args: argparse.Namespace) -> None:
    engine = ReactEngine(
        model=args.model,
        max_iterations=args.max_iterations,
        enable_validation=not args.no_validation,
    )
    result = await engine.search(args.query)

    if args.json:
        output = {
            "mode": "search",
            "query": args.query,
            "answer": result.answer,
            "confidence": result.confidence,
            "total_tokens": result.total_tokens,
            "truncated": result.truncated,
            "num_steps": len(result.steps),
            "steps": [
                {
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation_preview": s.observation[:200],
                }
                for s in result.steps
            ],
            "eval_result": result.eval_result,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print(f"ReAct Web Search Result")
        print(f"{'=' * 60}")
        print(f"Query:          {args.query}")
        print(f"Confidence:     {result.confidence}")
        print(f"Agent Steps:    {len(result.steps)}")
        print(f"Total Tokens:   {result.total_tokens}")
        print(f"Truncated:      {result.truncated}")
        print(f"{'-' * 60}")
        for i, step in enumerate(result.steps, 1):
            print(f"  Step {i}: {step.action}({list(step.action_input.keys())})")
        print(f"{'-' * 60}")
        print(f"Answer:\n{result.answer}")
        if result.eval_result:
            ev = result.eval_result
            status = "FAIL" if ev.get("has_critical_failure") else "PASS"
            print(f"{'-' * 60}")
            print(f"Validation [{status}]:")
            print(f"  Faithfulness:      {ev.get('faithfulness', 'N/A')}")
            print(f"  Hallucination:     {ev.get('hallucination_rate', 'N/A')}")
            print(f"  Answer Relevancy:  {ev.get('answer_relevancy', 'N/A')}")
            print(f"  Eval Tokens:       {ev.get('total_eval_tokens', 'N/A')}")
        print(f"{'=' * 60}\n")


async def run(args: argparse.Namespace) -> None:
    if args.analyze_only:
        await run_analyze_only(args)
    else:
        await run_search(args)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
