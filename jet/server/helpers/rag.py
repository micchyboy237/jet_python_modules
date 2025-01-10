from typing import Generator
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from jet.llm.query.retrievers import query_llm, setup_index

DEFAULT_SYSTEM = ""
DEFAULT_DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
DEFAULT_EXTENSIONS = [".md", ".mdx", ".rst"]


class RAG:
    def __init__(self, system=DEFAULT_SYSTEM, rag_dir=DEFAULT_DATA_DIR, extensions=DEFAULT_EXTENSIONS):
        self.system = system
        self.rag_dir = rag_dir
        self.extensions = extensions
        self.documents = self.load_documents(self.rag_dir, self.extensions)
        self.query_nodes = setup_index(self.documents)

    def load_documents(self, rag_dir, extensions):
        documents = SimpleDirectoryReader(
            rag_dir, required_exts=extensions, recursive=True).load_data()
        return documents

    def query(self, query) -> str | Generator[str, None, None]:
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

        result = self.query_nodes(query, FUSION_MODES.RELATIVE_SCORE)

        yield from query_llm(query, result['texts'])

    def get_results(self, query) -> str | Generator[str, None, None]:
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

        result = self.query_nodes(query, FUSION_MODES.RELATIVE_SCORE)

        return result
