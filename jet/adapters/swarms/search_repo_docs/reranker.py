"""Custom LlamaCpp reranker using httpx (not OpenAI SDK)."""

from __future__ import annotations

import sys
from typing import Any, List, Optional

import httpx
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle


class LlamaCppReranker(BaseNodePostprocessor):
    """Wraps a llama.cpp OpenAI-compatible /rerank endpoint."""

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
