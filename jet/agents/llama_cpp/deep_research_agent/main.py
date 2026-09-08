"""CLI entry point for the Deep Research Agent pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .orchestrator import AgenticRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep Research Agent: Adaptive retrieval with verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m deep_research_agent.main "Compare M1 vs Ryzen 3600 for local LLM inference"
  python -m deep_research_agent.main "What is RAG?" --max-depth 0 --json
  python -m deep_research_agent.main "Latest llama.cpp benchmarks" --snippet-threshold 0.8
        """,
    )
    parser.add_argument("query", help="Search query to answer")
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model override (default: from LLAMA_CPP_LLM_MODEL env)",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Embedding model override (default: from LLAMA_CPP_EMBED_MODEL env)",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="Rerank model override (default: from LLAMA_CPP_RERANK_MODEL env)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum link-following depth (default: 2)",
    )
    parser.add_argument(
        "--snippet-threshold",
        type=float,
        default=0.7,
        help="Rerank score threshold to skip navigation (default: 0.7)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output raw JSON instead of formatted text",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def format_output(result: dict) -> str:
    """Human-readable formatted output."""
    lines = [
        "=" * 70,
        "ANSWER",
        "=" * 70,
        result["answer"],
        "",
    ]

    if result["citations"]:
        lines.append("-" * 70)
        lines.append(f"CITATIONS ({len(result['citations'])})")
        lines.append("-" * 70)
        for i, cite in enumerate(result["citations"], 1):
            lines.append(f"  [{i}] {cite['claim']}")
            lines.append(f"      Source: {cite['source_url']}")
            lines.append(f'      Quote: "{cite["evidence_quote"][:120]}..."')
            lines.append("")

    if result["unresolved_sub_queries"]:
        lines.append("-" * 70)
        lines.append("⚠ UNRESOLVED SUB-QUERIES")
        lines.append("-" * 70)
        for uq in result["unresolved_sub_queries"]:
            lines.append(f"  • {uq}")
        lines.append("")

    lines.append("-" * 70)
    lines.append(
        f"Confidence: {result['confidence']:.0%} | "
        f"Evidence chunks: {result['evidence_count']}"
    )
    lines.append("=" * 70)
    return "\n".join(lines)


async def async_main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    agent = AgenticRAG(
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        rerank_model=args.rerank_model,
        max_depth=args.max_depth,
        snippet_threshold=args.snippet_threshold,
    )

    result = await agent.run(args.query)

    if args.output_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_output(result))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
