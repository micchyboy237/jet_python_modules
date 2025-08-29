import logging
from typing import Any, List, Literal, Optional

import numpy as np
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode

from jet.models.model_registry.transformers.cross_encoder_model_registry import CrossEncoderRegistry
from jet.models.model_types import RerankModelType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dispatcher = get_dispatcher(__name__)


class CrossEncoderRerank(BaseNodePostprocessor):
    model: str = Field(description="CrossEncoder model name.")
    top_n: int = Field(description="Top N nodes to return.")
    device: str = Field(
        description="Device to run model on (e.g., 'cpu', 'mps', 'cuda').", default="cpu")

    _model: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: RerankModelType = "cross-encoder/ms-marco-MiniLM-L12-v2",
        device: Optional[Literal["cpu", "mps"]] = None
    ):
        # super().__init__(top_n=top_n, model=model, device=device)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.error(
                "sentence-transformers not found. Please install with `pip install sentence-transformers`.")
            raise ImportError(
                "Cannot import sentence-transformers, please `pip install sentence-transformers`."
            )

        try:
            self._model = CrossEncoderRegistry.load_model(model, device=device)
            logger.info(
                f"Initialized CrossEncoderRerank with model: {model}, device: {device}")
        except Exception as e:
            logger.error(f"Failed to initialize CrossEncoder model: {str(e)}")
            raise

    @classmethod
    def class_name(cls) -> str:
        return "CrossEncoderRerank"

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
            logger.warning("Query bundle is None, returning empty list.")
            return []

        if len(nodes) == 0:
            logger.info(
                "No nodes provided for reranking, returning empty list.")
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
            try:
                query_text = query_bundle.query_str
                logger.debug(
                    f"Reranking {len(nodes)} nodes with query: {query_text}")

                # Prepare query-document pairs
                pairs = [
                    [query_text, node.node.get_content(
                        metadata_mode=MetadataMode.EMBED)]
                    for node in nodes
                ]

                # Get relevance scores
                scores = self._model.predict(pairs)
                logger.debug(f"Generated scores: {scores}")

                # Create new nodes with scores
                new_nodes = [
                    NodeWithScore(node=node.node, score=float(score))
                    for node, score in zip(nodes, scores)
                ]

                # Sort by score and take top_n
                new_nodes = sorted(new_nodes, key=lambda x: x.score, reverse=True)[
                    :self.top_n]
                logger.info(f"Reranked {len(new_nodes)} nodes.")

                event.on_end(payload={EventPayload.NODES: new_nodes})

            except Exception as e:
                logger.error(f"Error during reranking: {str(e)}")
                raise

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
