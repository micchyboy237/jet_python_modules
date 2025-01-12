import os
from typing import Any, List, Optional

from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL
from jet.vectors.reranker.utils import create_bm25_retriever
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode

dispatcher = get_dispatcher(__name__)


class BM25Rerank(BaseNodePostprocessor):
    model: str = Field(description="BM25 model name.",
                       default=OLLAMA_SMALL_EMBED_MODEL)
    top_n: int = Field(description="Top N nodes to return.", default=10)
    base_url: Optional[str] = Field(
        description="BM25 base url.", default=None)

    _client: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "BM25Rerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle, nodes=nodes, top_n=self.top_n, model_name=self.model
            )
        )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            bm25_retriever = create_bm25_retriever(
                nodes, similarity_top_k=self.top_n
            )
            new_nodes = bm25_retriever.retrieve(query_bundle)

            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
