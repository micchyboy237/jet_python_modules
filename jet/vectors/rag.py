from typing import List, Dict, TypedDict
from llama_index.core import VectorStoreIndex, Settings, download_loader, SimpleDirectoryReader
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import Document, QueryBundle, BaseNode, NodeWithScore
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.node_parser import SentenceSplitter


# TypedDict definitions for settings and results
class SettingsDict(TypedDict):
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    base_url: str


class ResultDict(TypedDict):
    settings: SettingsDict
    questions: List[Dict]
    chat: Dict
    rerank: Dict
    hyde: Dict


# Wikipedia loader class
class WikipediaLoader:
    @staticmethod
    def load(pages: List[str]) -> List[Document]:
        WikipediaReader = download_loader("WikipediaReader")
        loader = WikipediaReader()
        return loader.load_data(pages=pages, auto_suggest=False, redirect=False)


# Settings creation class
class SettingsManager:
    @staticmethod
    def create(settings: SettingsDict):
        Settings.chunk_size = settings["chunk_size"]
        Settings.chunk_overlap = settings["chunk_overlap"]
        Settings.embed_model = OllamaEmbedding(
            model_name=settings["embedding_model"],
            base_url=settings["base_url"],
        )
        Settings.llm = Ollama(
            model=settings["llm_model"],
            base_url=settings["base_url"],
            request_timeout=120.0
        )
        return Settings


# Query processing classes
class IndexManager:
    @staticmethod
    def create_nodes(documents: List[Document], chunk_size: int, chunk_overlap: int):
        # parse nodes
        parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes

    @staticmethod
    def create_index(embed_model: OllamaEmbedding, nodes: list[BaseNode]) -> VectorStoreIndex:
        # build index
        return VectorStoreIndex(
            nodes=nodes, embed_model=embed_model, show_progress=True)

    @staticmethod
    def create_retriever(index: VectorStoreIndex, top_k: int):
        retriever = index.as_retriever(similarity_top_k=top_k)
        return retriever


class QueryProcessor:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def generate_prompt(self, template: str, context: List[str], question: str) -> str:
        qa_template = PromptTemplate(template)
        return qa_template.format(context_str="\n\n".join(context), query_str=question)

    def rerank_nodes(self, index: VectorStoreIndex, query: str, top_n: int, model: str):
        retriever = index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)
        reranker = FlagEmbeddingReranker(top_n=top_n, model=model)
        query_bundle = QueryBundle(query_str=query)
        return reranker._postprocess_nodes(nodes, query_bundle=query_bundle)

    def hyde_query_transform(self, index: VectorStoreIndex, query: str):
        hyde = HyDEQueryTransform(include_original=True)
        hyde_query_engine = TransformQueryEngine(
            index.as_query_engine(similarity_top_k=4), hyde)
        return hyde_query_engine.query(query)

    def query_generate(self, prompt: str, model: BaseModel = None):
        llm = self.llm.as_structured_llm(model) if model else self.llm
        return llm.stream_complete(prompt)

    def query_chat(self, messages: List[ChatMessage], model: BaseModel = None):
        llm = self.llm.as_structured_llm(model) if model else self.llm
        return llm.stream_chat(messages)


def main():
    import os
    import json
    from jet.logger import logger
    from jet.file import save_json
    from jet.transformers import make_serializable
    from langchain_community.document_loaders.generic import GenericLoader
    from langchain_community.document_loaders.parsers import LanguageParser

    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # logger.log("OPENAI_API_KEY:", openai_api_key, colors=["GRAY", "INFO"])

    # Configuration
    settings = SettingsDict(
        llm_model="llama3.1",
        embedding_model="nomic-embed-text",
        chunk_size=512,
        chunk_overlap=50,
        base_url="http://localhost:11434",
    )
    reranking_model = "BAAI/bge-reranker-base"
    retriever_top_k = 3
    reranker_top_n = 3
    # pages = ["Emma_Stone", "La_La_Land", "Ryan_Gosling"]
    prompt_template = (
        "We have provided context information below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
        "Don't give an answer unless it is supported by the context above.\n"
    )
    questions = [
        "What is CrewAI?",
        "Give me code sample usage of CrewAI.",
    ]

    results = {
        "settings": settings,
        "questions": [],
        "chat": {},
        "rerank": {},
        "hyde": {},
    }
    settings_manager = SettingsManager.create(settings)
    # Merge settings
    logger.log("Settings:", json.dumps(settings), colors=["GRAY", "DEBUG"])
    save_json(results)

    # Load documents
    logger.debug("Loading documents...")
    # documents = WikipediaLoader.load(pages)
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/chatbot/open-webui/backend/crewAI/docs"
    documents = SimpleDirectoryReader(
        input_dir=base_dir,
        recursive=True,
    ).load_data()
    # loader = GenericLoader.from_filesystem(
    #     base_dir,
    #     glob="**/*",
    #     suffixes=[".mdx"],
    #     parser=LanguageParser("markdown"),
    # )
    # documents = loader.load()
    logger.log("Documents:", len(documents), colors=["GRAY", "DEBUG"])
    save_json(documents, file_path="generated/documents.json")

    # Create nodes
    logger.debug("Creating nodes...")
    nodes = IndexManager.create_nodes(
        documents=documents,
        chunk_size=settings_manager.chunk_size,
        chunk_overlap=settings_manager.chunk_overlap,
    )
    logger.log("Nodes:", len(nodes), colors=["GRAY", "DEBUG"])
    save_json(nodes, file_path="generated/nodes.json")

    # Create index
    logger.debug("Creating index...")
    index = IndexManager.create_index(
        embed_model=settings_manager.embed_model,
        nodes=nodes,
    )

    # Create retriever
    logger.debug("Creating retriever...")
    retriever = IndexManager.create_retriever(index, retriever_top_k)

    # Initialize QueryProcessor
    query_processor = QueryProcessor(llm=settings_manager.llm)

    # Process questions
    for question in questions:
        logger.log("Processing question:", question, colors=["GRAY", "DEBUG"])
        contexts: list[NodeWithScore] = retriever.retrieve(question)
        context_list = [node.get_content() for node in contexts]
        prompt = query_processor.generate_prompt(
            prompt_template, context_list, question)
        generation_response = query_processor.query_generate(prompt)

        response = ""
        stream_response = []
        for chunk in generation_response:
            response += chunk.delta
            stream_response.append(chunk)
            logger.success(chunk.delta, flush=True)
        results["questions"].append(
            {"question": question, "prompt": prompt, "contexts": contexts, "response": response, "stream_response": stream_response})
        save_json(results)

    # HyDE transformation
    hyde_query = "Give me code sample usage of CrewAI."
    logger.log("HyDE transforming on query",
               hyde_query, colors=["GRAY", "DEBUG"])
    hyde_response = query_processor.hyde_query_transform(index, hyde_query)
    results["hyde"] = {"query": hyde_query, **make_serializable(hyde_response)}
    save_json(results)

    # Reranking
    rerank_query = "Give me code sample usage of CrewAI."
    logger.log("Reranking on query:", rerank_query, colors=["GRAY", "DEBUG"])
    ranked_nodes = query_processor.rerank_nodes(
        index, rerank_query, reranker_top_n, reranking_model)
    results["rerank"] = {"query": rerank_query, "response": ranked_nodes}
    save_json(results)

    # Chat response
    logger.debug("Generating chat response...")
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="Give me code sample usage of CrewAI."
        ),
    ]
    chat_response = query_processor.query_chat(messages)

    response = ""
    stream_response = []
    for chunk in chat_response:
        response += chunk.delta
        stream_response.append(chunk)
        logger.success(chunk.delta, flush=True)
    results["chat"] = {"messages": messages,
                       "response": response, "stream_response": stream_response}
    save_json(results)


if __name__ == "__main__":
    main()
