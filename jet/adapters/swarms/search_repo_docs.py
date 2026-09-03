#!/usr/bin/env python3
"""
Reusable CLI for searching local repository docs using Swarms + LlamaIndex
with llama.cpp OpenAI-compatible servers. Supports one or more doc directories.

Usage:
    python search_repo_docs.py ./docs "How do I configure memory?"
    python search_repo_docs.py ./docs ./examples ./tutorials "async agent patterns" --top-k 20
    python search_repo_docs.py ./src/api ./src/examples "error handling" --rerank-top-n 3
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

# Add swarms module from local path if not already present
swarms_path = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/swarms"
if swarms_path not in sys.path:
    sys.path.append(swarms_path)

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI as OpenAIClient
from swarms import Agent, AgentRearrange


# ---------------------------------------------------------------------------
# Custom Reranker for llama.cpp /rerank endpoint
# ---------------------------------------------------------------------------
class LlamaCppReranker(BaseNodePostprocessor):
    """Wraps a llama.cpp OpenAI-compatible rerank endpoint."""

    # Pydantic fields are required by BaseNodePostprocessor
    top_n: int = Field(default=5, description="Number of nodes to return.")
    model: str = Field(description="Rerank model name.")
    base_url: str = Field(description="llama.cpp rerank API base URL.")

    _client: Any = PrivateAttr()

    def __init__(self, base_url: str, model: str, top_n: int = 5, **kwargs):
        super().__init__(top_n=top_n, model=model, base_url=base_url, **kwargs)
        self._client = OpenAIClient(base_url=base_url, api_key="not-needed")

    @classmethod
    def class_name(cls) -> str:
        return "LlamaCppReranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        texts = [
            str(n.node.get_content(metadata_mode=MetadataMode.EMBED)) for n in nodes
        ]
        try:
            response = self._client.post(
                "/rerank",
                json={
                    "model": self.model,
                    "query": query_bundle.query_str,
                    "documents": texts,
                    "top_n": self.top_n,
                },
            )
            results = response.json().get("results", [])
        except Exception as e:
            print(
                f"[WARN] Reranker failed ({e}), returning original nodes",
                file=sys.stderr,
            )
            return nodes[: self.top_n]

        reranked: List[NodeWithScore] = []
        for r in results[: self.top_n]:
            idx = r["index"]
            node = nodes[idx]
            node.score = r["relevance_score"]
            reranked.append(node)
        return reranked


# ---------------------------------------------------------------------------
# Multi-dir document loader with safe IDs
# ---------------------------------------------------------------------------
def load_documents_from_dirs(data_dirs: List[str]) -> list:
    """
    Load documents from multiple directories with unique IDs.

    Prepends the relative directory path to each filename to prevent collisions
    when multiple dirs contain files with the same basename (e.g., README.md).
    """
    all_documents = []
    for dir_path in data_dirs:
        resolved = Path(dir_path).resolve()
        if not resolved.is_dir():
            print(
                f"[WARN] Skipping non-existent directory: {dir_path}", file=sys.stderr
            )
            continue

        reader = SimpleDirectoryReader(
            input_dir=str(resolved),
            recursive=True,
            filename_as_id=False,  # We set custom IDs below
        )
        docs = reader.load_data()

        # Create unique IDs: relative_dir/filename to avoid cross-dir collisions
        for doc in docs:
            source_file = Path(doc.metadata.get("file_path", ""))
            try:
                rel_path = source_file.relative_to(resolved)
            except ValueError:
                rel_path = source_file.name
            doc.id_ = f"{resolved.name}/{rel_path}"
            doc.metadata["source_dir"] = str(resolved)
            doc.metadata["file_id"] = doc.id_

        all_documents.extend(docs)
        print(f"[INFO] Loaded {len(docs)} documents from {dir_path}")

    if not all_documents:
        raise FileNotFoundError(
            f"No documents found in any of the specified directories: {data_dirs}"
        )

    return all_documents


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------
def build_swarm(
    data_dirs: Union[str, List[str]],
    *,
    llm_model: Optional[str] = None,
    embed_model: Optional[str] = None,
    top_k: int = 15,
    rerank_top_n: int = 5,
    use_reranker: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> AgentRearrange:
    """
    Build and return a configured AgentRearrange swarm backed by local docs.

    Args:
        data_dirs: One or more paths to docs/examples directories.
        llm_model: Override LLM model name (default: from config).
        embed_model: Override embedding model name (default: from config).
        top_k: Chunks to retrieve before reranking.
        rerank_top_n: Results to keep after reranking.
        use_reranker: Whether to apply reranking post-retrieval.
        chunk_size: Embedding chunk size in tokens.
        chunk_overlap: Overlap between consecutive chunks.
    """
    from jet.adapters.llama_cpp.config import (
        EMBED_BASE_URL,
        EMBED_DIMS,
        EMBED_DOC_PREFIX,
        EMBED_MODEL,
        EMBED_QUERY_PREFIX,
        LLM_BASE_URL,
        LLM_MODEL,
        RERANK_BASE_URL,
        RERANK_MODEL,
    )

    # Normalize to list
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    # Resolve overrides
    _llm_model = llm_model or LLM_MODEL
    _embed_model = embed_model or EMBED_MODEL

    # --- Global LlamaIndex settings ------------------------------------------
    Settings.llm = OpenAILike(
        model=_llm_model,
        api_base=LLM_BASE_URL,
        api_key="not-needed",
        is_chat_model=True,
        timeout=120,
    )

    Settings.embed_model = OpenAIEmbedding(
        model_name=_embed_model,
        api_base=EMBED_BASE_URL,
        api_key="not-needed",
        dimensions=EMBED_DIMS,
        query_prefix=EMBED_QUERY_PREFIX,
        text_prefix=EMBED_DOC_PREFIX,
        embed_batch_size=32,
    )

    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap

    # --- Load documents from all directories ---------------------------------
    documents = load_documents_from_dirs(data_dirs)

    # --- Build index directly (avoids LlamaIndexDB input_files limitation) ---
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    # Wrap index in a minimal memory object compatible with AgentRearrange
    class _MemoryAdapter:
        """Thin adapter exposing .index for AgentRearrange's memory_system contract."""

        def __init__(self, vector_index: VectorStoreIndex):
            self.index = vector_index

        def query(self, query_str: str) -> str:
            engine = self.index.as_query_engine(similarity_top_k=top_k)
            return str(engine.query(query_str))

    memory = _MemoryAdapter(index)

    # --- Agent ---------------------------------------------------------------
    docs_agent = Agent(
        agent_name="Repo-Docs-Searcher",
        description=(
            "Searches local repo docs using semantic retrieval. "
            "Always cites exact file paths. If info is missing, say so."
        ),
        llm=Settings.llm,
        max_loops=1,
        system_prompt=(
            "You are a documentation search expert. Use ONLY retrieved context. "
            "Always cite exact file paths (including source directory) and line ranges. "
            "If the answer is not in the provided context, state that clearly."
        ),
    )

    # --- Swarm ---------------------------------------------------------------
    swarm = AgentRearrange(
        name="Local-Repo-RAG-Swarm",
        agents=[docs_agent],
        memory_system=memory,
        flow=f"{docs_agent.agent_name}",
        max_loops=1,
    )

    # Attach reranker if configured
    if use_reranker and RERANK_BASE_URL and RERANK_MODEL:
        swarm._reranker = LlamaCppReranker(  # type: ignore[attr-defined]
            base_url=RERANK_BASE_URL,
            model=RERANK_MODEL,
            top_n=rerank_top_n,
        )
    else:
        swarm._reranker = None  # type: ignore[attr-defined]

    # Stash index reference for direct query-engine access
    swarm._index = index  # type: ignore[attr-defined]

    return swarm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search local repo docs via Swarms + llama.cpp RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s ./docs 'How does AgentRearrange handle errors?'\n"
            "  %(prog)s ./docs ./examples ./tutorials 'async agent patterns' --top-k 20\n"
            "  %(prog)s ./src/api ./src/examples 'error handling' --rerank-top-n 3\n"
            "  %(prog)s ./docs 'API reference' --model qwen3.5-uncensored:4b --no-reranker\n"
        ),
    )

    # Positional: one or more directories + query
    parser.add_argument(
        "data_dirs",
        nargs="+",
        metavar="DATA_DIR_OR_QUERY",
        help="One or more doc directories followed by the query string",
    )
    parser.add_argument(
        "--model",
        dest="llm_model",
        default=None,
        help="Override LLM model name (default: from LLAMA_CPP_LLM_MODEL env)",
    )
    parser.add_argument(
        "--embed-model",
        dest="embed_model",
        default=None,
        help="Override embedding model name (default: from LLAMA_CPP_EMBED_MODEL env)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of chunks to retrieve before reranking (default: 15)",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=5,
        help="Number of results to keep after reranking (default: 5)",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        default=False,
        help="Disable reranker even if RERANK_BASE_URL is configured",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Embedding chunk size in tokens (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks (default: 200)",
    )

    args = parser.parse_args(argv)

    # Split positional args: everything that is an existing dir vs the query
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


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    print(f"[INFO] Directories: {args.data_dirs}")
    print(f"[INFO] Query:       {args.query}")
    print(f"[INFO] Model:      {args.llm_model or '(from config)'}")
    print("-" * 60)

    swarm = build_swarm(
        data_dirs=args.data_dirs,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        use_reranker=not args.no_reranker,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Run with optional reranker post-processing
    reranker = getattr(swarm, "_reranker", None)
    index = getattr(swarm, "_index", None)

    if reranker is not None and index is not None:
        try:
            engine = index.as_query_engine(
                similarity_top_k=args.top_k,
                node_postprocessors=[reranker],
            )
            response = engine.query(args.query)
            print(str(response))
        except Exception as e:
            print(f"[WARN] Reranked query failed ({e}), falling back", file=sys.stderr)
            print(swarm.run(args.query))
    else:
        print(swarm.run(args.query))


if __name__ == "__main__":
    main()
