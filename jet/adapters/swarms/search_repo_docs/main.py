#!/usr/bin/env python3
"""
CLI entry point for multi-agent repository documentation search.

Usage:
    python -m jet.adapters.swarms.search_repo_docs.main ./docs "How do I configure memory?"
    python -m jet.adapters.swarms.search_repo_docs.main ./docs ./examples "async patterns" --top-k 20
    python -m jet.adapters.swarms.search_repo_docs.main ./docs "API reference" --no-stream --enable-thinking
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

# Add swarms module from local path if not installed
swarms_path = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/swarms"
if swarms_path not in sys.path:
    sys.path.append(swarms_path)

from .config import SearchConfig
from .index import build_index
from .swarm import build_swarm


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent repo doc search via Swarms + llama.cpp RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s ./docs 'How does AgentRearrange handle errors?'\n"
            "  %(prog)s ./docs ./examples 'async agent patterns' --top-k 20\n"
            "  %(prog)s ./docs 'API reference' --no-stream --enable-thinking\n"
            "  %(prog)s ./docs 'memory config' --model qwen3.5-uncensored:4b\n"
        ),
    )

    parser.add_argument(
        "data_dirs",
        nargs="+",
        metavar="DATA_DIR_OR_QUERY",
        help="One or more doc directories followed by the query string",
    )
    parser.add_argument(
        "--model", dest="llm_model", default=None, help="Override LLM model name"
    )
    parser.add_argument(
        "--embed-model",
        dest="embed_model",
        default=None,
        help="Override embedding model name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Chunks to retrieve before reranking (default: 15)",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=5,
        help="Results to keep after reranking (default: 5)",
    )
    parser.add_argument(
        "--no-reranker", action="store_true", default=False, help="Disable reranker"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming output",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable model chain-of-thought (default: disabled)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Embedding chunk size (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200)"
    )

    args = parser.parse_args(argv)

    # Split positional args: existing dirs vs query tokens
    dirs: List[str] = []
    query_parts: List[str] = []
    found_non_dir = False
    for arg in args.data_dirs:
        if not found_non_dir and os.path.isdir(arg):
            dirs.append(arg)
        else:
            found_non_dir = True
            query_parts.append(arg)

    if not dirs:
        parser.error("At least one valid directory path is required before the query")
    if not query_parts:
        parser.error("A query string is required after the directory path(s)")

    args.data_dirs = dirs
    args.query = " ".join(query_parts)
    return args


def run_streaming(swarm, query: str) -> None:
    """Stream tokens from the pipeline using AgentRearrange.run_stream()."""
    try:
        for agent_name, token in swarm.run_stream(task=query):
            # Only stream Synthesizer and Verifier output to avoid noise
            if agent_name in ("Synthesizer", "Verifier"):
                print(token, end="", flush=True)
        print()  # Final newline
    except Exception as e:
        print(
            f"\n[WARN] Streaming failed ({e}), falling back to batch mode",
            file=sys.stderr,
        )
        result = swarm.run(query)
        print(result)


def run_batch(swarm, query: str) -> None:
    """Run the full pipeline and print the final verified answer."""
    result = swarm.run(query)
    print(result)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = SearchConfig.from_args(args)

    print(f"[INFO] Directories:      {cfg.data_dirs}")
    print(f"[INFO] Query:            {cfg.query}")
    print(f"[INFO] Model:            {cfg.llm_model or '(from config)'}")
    print(f"[INFO] Streaming:        {cfg.use_stream}")
    print(f"[INFO] Thinking enabled: {cfg.enable_thinking}")
    print(
        f"[INFO] Pipeline:         Query-Decomposer → Retriever → Analyzer → Synthesizer → Verifier"
    )
    print("-" * 70)

    # Build index (configures Settings globally)
    index = build_index(cfg)

    # Build swarm (uses Settings.llm configured above)
    swarm = build_swarm(cfg)

    # Store index reference for potential direct access
    swarm._index = index

    # Execute
    if cfg.use_stream:
        run_streaming(swarm, cfg.query)
    else:
        run_batch(swarm, cfg.query)


if __name__ == "__main__":
    main()
