# jet_python_modules/jet/adapters/llama_cpp/tasks/rag/eval/llm_as_a_judge/main.py
"""
RAG Evaluation CLI

Title:
    RAG LLM-as-a-Judge Evaluation Runner

Description:
    Command-line interface for running RAG evaluation pipelines using the
    JetLLMJudge adapter. Supports all three evaluation stages:
      - pre_gen: Fast retrieval quality gate (blocks bad context)
      - prod: Reference-free safety monitoring (faithfulness, hallucination)
      - offline: Full benchmark suite with ground-truth recall
    
    All evaluations use structured output parsing via llm_utils.achat and
    automatically truncate contexts to fit the judge model's window.

Args Description:
    stage           Evaluation stage to run: pre_gen, prod, or offline
    --query         User query string (required for all stages)
    --contexts      One or more retrieved context chunks (required for all stages)
    --response      Generated response text (required for prod and offline)
    --reference     Ground-truth reference answer (required for offline only)
    --model         Judge model identifier (default: qwen3.5-uncensored:2b)
    --json          Output results as JSON instead of human-readable table

Usage Examples:
    # Pre-generation gate check
    python main.py pre_gen \\
        --query "What is DNS?" \\
        --contexts "DNS translates domain names to IPs." \\
                   "Recursive resolvers query root servers."

    # Production async evaluation
    python main.py prod \\
        --query "What is DNS?" \\
        --contexts "DNS translates domain names to IPs." \\
        --response "DNS converts domain names into IP addresses."

    # Offline benchmark with ground truth
    python main.py offline \\
        --query "What is DNS?" \\
        --contexts "DNS translates domain names to IPs." \\
        --response "DNS converts domain names into IP addresses." \\
        --reference "DNS resolution maps domain names to IP addresses." \\
        --model qwen3.5-uncensored:2b --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Sequence

from jet.adapters.llama_cpp.tasks.rag.eval.llm_as_a_judge import RAGEvaluator

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Run RAG evaluation stages via LLM-as-a-Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage",
        choices=["pre_gen", "prod", "offline"],
        help="Evaluation stage to execute",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User query string",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        required=True,
        help="One or more retrieved context chunks",
    )
    parser.add_argument(
        "--response",
        default=None,
        help="Generated response (required for prod and offline stages)",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Ground-truth reference answer (required for offline stage)",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5-uncensored:2b",
        help="Judge model identifier (default: qwen3.5-uncensored:2b)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON instead of human-readable format",
    )
    return parser


async def run(args: argparse.Namespace) -> None:
    evaluator = RAGEvaluator(model=args.model)

    if args.stage == "pre_gen":
        result = await evaluator.evaluate_pre_generation_gate(
            query=args.query,
            contexts=args.contexts,
        )
    elif args.stage == "prod":
        if not args.response:
            print("Error: --response is required for 'prod' stage", file=sys.stderr)
            sys.exit(1)
        result = await evaluator.evaluate_production_async(
            query=args.query,
            contexts=args.contexts,
            response=args.response,
        )
    elif args.stage == "offline":
        if not args.response:
            print("Error: --response is required for 'offline' stage", file=sys.stderr)
            sys.exit(1)
        if not args.reference:
            print("Error: --reference is required for 'offline' stage", file=sys.stderr)
            sys.exit(1)
        result = await evaluator.evaluate_offline(
            query=args.query,
            contexts=args.contexts,
            response=args.response,
            reference=args.reference,
        )
    else:
        print(f"Error: Unknown stage '{args.stage}'", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output = {
            "stage": result.stage.value,
            "query": result.query,
            "passed_gate": result.passed_gate,
            "total_eval_tokens": result.total_eval_tokens,
            "has_critical_failure": result.has_critical_failure,
            "metrics": {},
            "metadata": result.metadata,
        }
        for field_name in (
            "contextual_precision",
            "contextual_recall",
            "faithfulness",
            "hallucination_rate",
            "answer_relevancy",
        ):
            value = getattr(result, field_name, None)
            if value is not None:
                output["metrics"][field_name] = round(value, 4)
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print(f"Stage:              {result.stage.value}")
        print(f"Query:              {result.query[:80]}")
        print(f"Passed Gate:        {result.passed_gate}")
        print(f"Critical Failure:   {result.has_critical_failure}")
        print(f"Total Eval Tokens:  {result.total_eval_tokens}")
        print(f"{'-' * 60}")
        if result.contextual_precision is not None:
            print(f"Contextual Precision: {result.contextual_precision:.4f}")
        if result.contextual_recall is not None:
            print(f"Contextual Recall:    {result.contextual_recall:.4f}")
        if result.faithfulness is not None:
            print(f"Faithfulness:         {result.faithfulness:.4f}")
        if result.hallucination_rate is not None:
            print(f"Hallucination Rate:   {result.hallucination_rate:.4f}")
        if result.answer_relevancy is not None:
            print(f"Answer Relevancy:     {result.answer_relevancy:.4f}")
        if result.metadata:
            print(f"{'-' * 60}")
            for k, v in result.metadata.items():
                print(f"  {k}: {v}")
        print(f"{'=' * 60}\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
