"""Swarm assembly: wires agents into AgentRearrange with the correct flow."""

from __future__ import annotations

from llama_index.core import Settings
from swarms import AgentRearrange

from .agents import (
    create_analyzer,
    create_query_decomposer,
    create_retriever,
    create_synthesizer,
    create_verifier,
)
from .config import SearchConfig
from .reranker import LlamaCppReranker

FLOW = "Query-Decomposer -> Retriever -> Analyzer -> Synthesizer -> Verifier"


def build_swarm(cfg: SearchConfig) -> AgentRearrange:
    """Build the five-agent pipeline with shared LLM and optional reranker."""
    llm = Settings.llm

    agents = [
        create_query_decomposer(llm),
        create_retriever(llm),
        create_analyzer(llm),
        create_synthesizer(llm),
        create_verifier(llm),
    ]

    swarm = AgentRearrange(
        name="Multi-Agent-Doc-Search",
        description="Five-agent pipeline for accurate, verified repo doc search",
        agents=agents,
        flow=FLOW,
        max_loops=2,  # Allows Verifier → Synthesizer revision loop
        team_awareness=True,  # Each agent sees all prior outputs
        output_type="final",  # Return only the Verifier's final output
        verbose=False,
    )

    # Attach reranker metadata for downstream use
    rerank_url, rerank_model = cfg.resolve_rerank()
    if rerank_url and rerank_model:
        swarm._reranker = LlamaCppReranker(
            base_url=rerank_url,
            model=rerank_model,
            top_n=cfg.rerank_top_n,
        )
    else:
        swarm._reranker = None

    swarm._use_stream = cfg.use_stream
    swarm._top_k = cfg.top_k

    return swarm
