from typing import Generator
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from rag.retrievers import (
    load_documents,
    setup_index,
    query_llm,
    setup_index,
)


class RAG:
    def __init__(self, system, data_dir, rag_dir, extensions):
        self.system = system
        self.data_dir = data_dir
        self.rag_dir = rag_dir
        self.extensions = extensions
        self.documents = load_documents(self.rag_dir, self.extensions)
        self.query_nodes = setup_index(self.documents, self.data_dir)

    def query(self, query) -> str | Generator[str, None, None]:
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

        result = self.query_nodes(query, FUSION_MODES.RELATIVE_SCORE)

        yield from query_llm(query, result['texts'])

    def get_results(self, query) -> str | Generator[str, None, None]:
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

        result = self.query_nodes(query, FUSION_MODES.RELATIVE_SCORE)

        return result
