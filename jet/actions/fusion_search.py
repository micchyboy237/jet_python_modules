import os
from typing import Generator
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import Document, NodeWithScore, BaseNode, TextNode, ImageNode
from llama_index.core.node_parser import TokenTextSplitter

from jet.llm.utils import display_jet_source_nodes
from jet.vectors import get_source_node_attributes
from jet.logger import logger
from jet.actions import call_ollama_chat
from jet.llm.llm_types import OllamaChatOptions
from jet.llm.query import setup_index, query_llm
from jet.llm.ollama.base import initialize_ollama_settings, large_llm_model
initialize_ollama_settings()


def fusion_search(
    queries: list[str],
    candidates: list[str | Document],
    *,
    top_k=3,
    nlist=100,
):
    text_splitter = TokenTextSplitter(
        separator=" ", chunk_size=512, chunk_overlap=128
    )

    query = "Tell me about yourself."

    documents = [queries]

    candidates_docs = [
        Document(text=candidate)
        if isinstance(candidate, str) else candidate
        for candidate in candidates
    ]
    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=20,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    query_nodes = setup_index(candidates_docs)

    # logger.newline()
    # logger.info("RECIPROCAL_RANK: query...")
    # response = query_nodes(query, FUSION_MODES.RECIPROCAL_RANK)

    # logger.newline()
    # logger.info("DIST_BASED_SCORE: query...")
    # response = query_nodes(query, FUSION_MODES.DIST_BASED_SCORE)

    logger.newline()
    logger.info("RELATIVE_SCORE: sample query...")
    result = query_nodes(
        query, FUSION_MODES.RELATIVE_SCORE)
    display_jet_source_nodes(query, result["nodes"])
    node_results = [get_source_node_attributes(
        node) for node in result["nodes"]]
