#!/usr/bin/env python3
"""
Reusable CLI for searching local repository docs using Swarms + LlamaIndex
with llama.cpp OpenAI-compatible servers. Supports one or more doc directories.

Usage:
    python search_repo_docs.py ./docs "How do I configure memory?"
    python search_repo_docs.py ./docs ./examples "async agent patterns" --top-k 20
    python search_repo_docs.py ./docs "API reference" --no-stream --enable-thinking
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

# Add swarms module from local path if not installed
swarms_path = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/swarms"
if swarms_path not in sys.path:
    sys.path.append(swarms_path)

import httpx
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from swarms import Agent, AgentRearrange

# ---------------------------------------------------------------------------
# File extensions to index
# ---------------------------------------------------------------------------
REQUIRED_EXTENSIONS = [
    ".md",
    ".mdx",
    ".py",
    ".ipynb",
    ".txt",
    ".rst",
    ".yaml",
    ".yml",
    ".json",
]


# ---------------------------------------------------------------------------
# Custom Reranker for llama.cpp /rerank endpoint (uses httpx, not OpenAI SDK)
# ---------------------------------------------------------------------------
class LlamaCppReranker(BaseNodePostprocessor):
    """Wraps a llama.cpp OpenAI-compatible rerank endpoint using httpx."""

    top_n: int = Field(default=5, description="Number of nodes to return.")
    model: str = Field(description="Rerank model name.")
    base_url: str = Field(description="llama.cpp rerank API base URL.")

    _client: Any = PrivateAttr()

    def __init__(self, base_url: str, model: str, top_n: int = 5, **kwargs):
        super().__init__(top_n=top_n, model=model, base_url=base_url, **kwargs)
        self._client = httpx.Client(base_url=base_url, timeout=60.0)

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
            response.raise_for_status()
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
    Prepends the directory name to each filepath to prevent ID collisions.
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
            filename_as_id=False,
            required_exts=REQUIRED_EXTENSIONS,
        )
        docs = reader.load_data()

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
    use_stream: bool = True,
    enable_thinking: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> AgentRearrange:
    """
    Build and return a configured AgentRearrange swarm backed by local docs.

    All models/URLs are read from jet.adapters.llama_cpp.config; keyword args
    override individual values without mutating the global config module.
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

    # --- Build extra_body for enable_thinking control ------------------------
    # OpenAILike passes additional_kwargs -> extra_body -> chat_template_kwargs
    # to the OpenAI SDK, which sends it in the request body to llama.cpp.
    # Ref: https://github.com/run-llama/llama_index/issues/18635
    additional_kwargs = {
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            }
        }
    }

    # --- Global LlamaIndex settings ------------------------------------------
    Settings.llm = OpenAILike(
        model=_llm_model,
        api_base=LLM_BASE_URL,
        api_key="not-needed",
        is_chat_model=True,
        timeout=120,
        additional_kwargs=additional_kwargs,
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

    # --- Build index ---------------------------------------------------------
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    # Wrap index in a minimal memory object for AgentRearrange
    class _MemoryAdapter:
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

    # Stash references for query-time access
    swarm._index = index  # type: ignore[attr-defined]
    swarm._use_stream = use_stream  # type: ignore[attr-defined]
    swarm._top_k = top_k  # type: ignore[attr-defined]

    return swarm


# ---------------------------------------------------------------------------
# Query runner with streaming support
# ---------------------------------------------------------------------------
def run_query(swarm: AgentRearrange, query: str) -> None:
    """
    Execute a query against the swarm with optional streaming + reranking.

    Streaming uses LlamaIndex's native streaming query engine which returns
    a StreamingResponse with a response_gen generator. Each chunk is printed
    and flushed immediately for natural real-time output.
    Ref: https://developers.llamaindex.ai/.../query_engine/streaming/
    """
    reranker = getattr(swarm, "_reranker", None)
    index = getattr(swarm, "_index", None)
    use_stream = getattr(swarm, "_use_stream", False)
    top_k = getattr(swarm, "_top_k", 15)

    postprocessors = [reranker] if reranker else []

    if index is not None and use_stream:
        # --- STREAMING PATH ---
        try:
            engine = index.as_query_engine(
                streaming=True,
                similarity_top_k=top_k,
                node_postprocessors=postprocessors,
            )
            streaming_response = engine.query(query)

            # Iterate over token chunks as they arrive, flush each immediately
            for text in streaming_response.response_gen:
                print(text, end="", flush=True)
            print()  # Final newline

            # Print source citations after streaming completes
            if hasattr(streaming_response, "source_nodes"):
                print("\n" + "=" * 60)
                print("📄 Sources:")
                print("=" * 60)
                for i, node in enumerate(streaming_response.source_nodes, 1):
                    file_id = node.node.metadata.get("file_id", "unknown")
                    score = f"{node.score:.4f}" if node.score else "N/A"
                    print(f"  [{i}] {file_id} (score: {score})")
            return
        except Exception as e:
            print(f"[WARN] Streaming failed ({e}), falling back", file=sys.stderr)

    if index is not None and not use_stream:
        # --- NON-STREAMING PATH (with reranker) ---
        try:
            engine = index.as_query_engine(
                similarity_top_k=top_k,
                node_postprocessors=postprocessors,
            )
            response = engine.query(query)
            print(str(response))

            if hasattr(response, "source_nodes"):
                print("\n" + "=" * 60)
                print("📄 Sources:")
                print("=" * 60)
                for i, node in enumerate(response.source_nodes, 1):
                    file_id = node.node.metadata.get("file_id", "unknown")
                    score = f"{node.score:.4f}" if node.score else "N/A"
                    print(f"  [{i}] {file_id} (score: {score})")
            return
        except Exception as e:
            print(f"[WARN] Direct query failed ({e}), falling back", file=sys.stderr)

    # --- FALLBACK: swarm.run() ---
    print(swarm.run(query))


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
            "  %(prog)s ./docs ./examples 'async agent patterns' --top-k 20\n"
            "  %(prog)s ./docs 'API reference' --no-stream --enable-thinking\n"
            "  %(prog)s ./docs 'memory config' --model qwen3.5-uncensored:4b\n"
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
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming; wait for full response before printing",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable model thinking/chain-of-thought (default: disabled)",
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

    # Split positional args: directories vs query
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

    print(f"[INFO] Directories:      {args.data_dirs}")
    print(f"[INFO] Query:            {args.query}")
    print(f"[INFO] Model:            {args.llm_model or '(from config)'}")
    print(f"[INFO] Streaming:        {not args.no_stream}")
    print(f"[INFO] Thinking enabled: {args.enable_thinking}")
    print("-" * 60)

    swarm = build_swarm(
        data_dirs=args.data_dirs,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        use_reranker=not args.no_reranker,
        use_stream=not args.no_stream,
        enable_thinking=args.enable_thinking,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    run_query(swarm, args.query)


if __name__ == "__main__":
    main()
