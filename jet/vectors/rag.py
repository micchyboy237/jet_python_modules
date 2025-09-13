from typing import List, Dict, Literal, Sequence, TypedDict
from llama_index.core import VectorStoreIndex, Settings, download_loader, SimpleDirectoryReader
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import Document, QueryBundle, BaseNode, NodeWithScore
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine.transform_query_engine import TransformQueryEngine
from jet.llm.ollama.base import OllamaEmbedding
from jet.llm.ollama.base import Ollama
from llama_index.core.llms import (
    LLM,
    ChatMessage,
    MessageRole,
    ChatResponseGen,
    CompletionResponseGen,
)
from llama_index.core.utils import set_global_tokenizer
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.node_parser import NodeParser, SentenceSplitter
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.query_engine import BaseQueryEngine, MultiStepQueryEngine
from jet.llm.utils.llama_index_utils import display_jet_source_node


DEFAULT_SETTINGS = {
    "llm_model": "llama3.1",
    "embedding_model": "nomic-embed-text",
    "chunk_size": 768,
    "chunk_overlap": 50,
    "base_url": "http://localhost:11434",
}


class SettingsDict(TypedDict, total=False):
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    base_url: str


class ResultDict(TypedDict, total=False):
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
    def create(settings: SettingsDict = {}):
        settings = {**DEFAULT_SETTINGS, **settings}

        # Settings.chunk_size = settings["chunk_size"]
        # Settings.chunk_overlap = settings["chunk_overlap"]
        Settings.embed_model = SettingsManager.create_embed_model(
            model=settings["embedding_model"],
            base_url=settings["base_url"],
        )
        Settings.llm = SettingsManager.create_llm(
            model=settings["llm_model"],
            base_url=settings["base_url"],
        )

        from jet._token.token_utils import get_ollama_tokenizer
        tokenizer = get_ollama_tokenizer(settings["llm_model"])
        set_global_tokenizer(tokenizer)
        return Settings

    @staticmethod
    def create_llm(model: str, base_url: str, temperature: float = 0):
        llm = Ollama(
            temperature=temperature,
            context_window=4096,
            # num_predict=-1,
            # num_keep=1,
            request_timeout=300.0,
            model=model,
            base_url=base_url,
        )
        Settings.llm = llm
        return llm

    @staticmethod
    def create_embed_model(model: str, base_url: str):
        embed_model = OllamaEmbedding(
            model_name=model,
            base_url=base_url,
        )
        Settings.embed_model = embed_model
        return embed_model


# Query processing classes
class IndexManager:
    @staticmethod
    def create_nodes(documents: List[Document], parser: NodeParser):
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes

    @staticmethod
    def create_query_nodes(retriever: BaseRetriever, query: str):
        nodes: list[NodeWithScore] = retriever.retrieve(query)
        return nodes

    @staticmethod
    def create_reranker(
        type: Literal["flag", "llm", "sentence"] = "flag",
        top_n: int = 3,
        *,
        llm: LLM = None,
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        reranker = None
        if type == "flag":
            reranker = FlagEmbeddingReranker(
                model=reranker_model,
                top_n=top_n,
            )
        elif type == "llm":
            reranker = RankGPTRerank(
                llm=llm,
                top_n=top_n,
            )
        elif type == "sentence":
            reranker = SentenceTransformerRerank(
                model=reranker_model,
                top_n=top_n,
                keep_retrieval_score=True,
            )
        return reranker

    @staticmethod
    def create_index(
        embed_model: OllamaEmbedding,
        nodes: list[BaseNode] = [],
        documents: Sequence[Document] = [],
    ) -> VectorStoreIndex:
        # build index
        if documents:
            return VectorStoreIndex(
                embed_model=embed_model,
                nodes=documents,
                show_progress=True,
            )

        return VectorStoreIndex(
            nodes=nodes, embed_model=embed_model, show_progress=True)

    @staticmethod
    def create_retriever(index: VectorStoreIndex, similarity_top_k: int):
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        return retriever

    @staticmethod
    def create_query_engine(index: VectorStoreIndex, similarity_top_k: int, **kwargs: any):
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k, **kwargs)
        return query_engine


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
        # Return the updated chunks
        query_bundle = QueryBundle(query_str=query)
        ranked_nodes = reranker._postprocess_nodes(
            nodes, query_bundle=query_bundle)
        for ranked_node in ranked_nodes:
            print('----------------------------------------------------')
            display_jet_source_node(ranked_node, source_length=500)
        # Initialize the query engine with Re-Ranking
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            node_postprocessors=[reranker]
        )

        # Print the response from the model
        response = query_engine.query(
            "Compare the families of Emma Stone and Ryan Gosling")

        print(response)

    def hyde_query_transform(self, query_engine: BaseQueryEngine, query: str):
        hyde = HyDEQueryTransform(include_original=True)
        hyde_query_engine = TransformQueryEngine(query_engine, hyde)
        return hyde_query_engine.query(query)

    def multi_step_query(self, llm: LLM, query_engine: BaseQueryEngine, query: str):
        # Multi-step query setup
        step_decompose_transform_gpt3 = StepDecomposeQueryTransform(
            llm, verbose=True)
        index_summary = "Breaks down the initial query"
        multi_step_query_engine = MultiStepQueryEngine(
            query_engine=query_engine,
            query_transform=step_decompose_transform_gpt3,
            index_summary=index_summary
        )
        response = multi_step_query_engine.query(query)
        return response

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
    from jet.transformers.object import make_serializable
    from langchain_community.document_loaders.generic import GenericLoader
    from langchain_community.document_loaders.parsers import LanguageParser

    # base_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/chatbot/open-webui/backend/crewAI/docs"
    context_files = [
        "/Users/jethroestrada/Desktop/External_Projects/AI/chatbot/open-webui/backend/crewAI/docs/installation.mdx",
        "/Users/jethroestrada/Desktop/External_Projects/AI/chatbot/open-webui/backend/crewAI/docs/introduction.mdx",
        "/Users/jethroestrada/Desktop/External_Projects/AI/chatbot/open-webui/backend/crewAI/docs/quickstart.mdx",
    ]

    query = "How do I use CrewAI?"
    similarity_top_k = 4
    reranker_model = "BAAI/bge-reranker-base"
    sentence_reranker_model = "cross-encoder/stsb-distilroberta-base"
    reranker_top_n = 3

    # Configuration
    settings = SettingsDict(
        llm_model="llama3.1",
        embedding_model="nomic-embed-text",
        chunk_size=512,
        chunk_overlap=50,
        base_url="http://localhost:11434",
    )

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
    ]
    questions.append(query)

    results = {
        "settings": settings,
        "questions": [],
        "chats": [],
        "rerank": {},
        "hyde": {},
    }
    settings_manager = SettingsManager.create(settings)
    # Merge settings
    logger.log("Settings:", json.dumps(settings), colors=["GRAY", "DEBUG"])
    save_json(results, file_path="generated/crewai/results.json")

    # Load documents
    logger.debug("Loading documents...")
    # documents = WikipediaLoader.load(pages)
    documents = SimpleDirectoryReader(
        input_files=context_files,
        # input_dir=base_dir,
        # recursive=True,
    ).load_data()
    # loader = GenericLoader.from_filesystem(
    #     base_dir,
    #     glob="**/*",
    #     suffixes=[".mdx"],
    #     parser=LanguageParser("markdown"),
    # )
    # documents = loader.load()
    logger.log("Documents:", len(documents), colors=["GRAY", "DEBUG"])
    save_json(documents, file_path="generated/crewai/documents.json")

    # Create all nodes
    logger.debug("Creating nodes...")
    all_nodes = IndexManager.create_nodes(
        documents=documents, parser=settings_manager.node_parser)

    # Create index
    logger.debug("Creating index...")
    index = IndexManager.create_index(
        embed_model=settings_manager.embed_model,
        nodes=all_nodes,
    )

    # Create retriever
    logger.debug("Creating retriever...")
    retriever = IndexManager.create_retriever(
        index=index, similarity_top_k=3)

    # Create query nodes
    query_nodes = IndexManager.create_query_nodes(
        retriever=retriever, query=query)
    # Print the chunks
    for node in query_nodes:
        print('----------------------------------------------------')
        display_jet_source_node(node, source_length=500)
    results["initial_query_nodes"] = {
        "query": query,
        "response": query_nodes,
    }
    save_json(results, file_path="generated/crewai/results.json")

    # Create initial query engine
    query_engine = IndexManager.create_query_engine(
        index=index, similarity_top_k=4)

    # generate the response
    logger.log("Generating initial query:", query, colors=["GRAY", "DEBUG"])
    response = query_engine.query(query)
    results["initial_query"] = {
        "query": query,
        "response": response,
    }
    save_json(results, file_path="generated/crewai/results.json")

    # Start query reranking

    # Initialize QueryProcessor
    query_processor = QueryProcessor(llm=settings_manager.llm)

    # Flag reranking
    logger.log("Flag reranking on query:", query, colors=["GRAY", "DEBUG"])
    reranker = IndexManager.create_reranker(
        "flag", top_n=reranker_top_n, reranker_model=reranker_model)
    query_engine = IndexManager.create_query_engine(
        index=index, similarity_top_k=3, node_postprocessors=[reranker])
    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker._postprocess_nodes(
        query_nodes, query_bundle=query_bundle)
    for ranked_node in ranked_nodes:
        print('----------------------------------------------------')
        display_jet_source_node(ranked_node, source_length=500)
    # Print the response from the model
    response = query_engine.query(query)
    results["rerank"] = {
        "flag": {
            "query": query,
            "response": response,
        }
    }
    save_json(results, file_path="generated/crewai/results.json")

    # LLM reranking
    logger.log("LLM reranking on query:", query, colors=["GRAY", "DEBUG"])
    reranker = IndexManager.create_reranker(
        "llm", top_n=reranker_top_n, llm=settings_manager.llm)
    query_engine = IndexManager.create_query_engine(
        index=index, similarity_top_k=3, node_postprocessors=[reranker])
    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker._postprocess_nodes(
        query_nodes, query_bundle=query_bundle)
    for ranked_node in ranked_nodes:
        print('----------------------------------------------------')
        display_jet_source_node(ranked_node, source_length=500)
    # Print the response from the model
    response = query_engine.query(query)
    results["rerank"] = {
        "llm": {
            "query": query,
            "response": response,
        }
    }
    save_json(results, file_path="generated/crewai/results.json")

    # Sentence reranking
    logger.log("Sentence reranking on query:", query, colors=["GRAY", "DEBUG"])
    reranker = IndexManager.create_reranker(
        "sentence", top_n=reranker_top_n, reranker_model=sentence_reranker_model)
    query_engine = IndexManager.create_query_engine(
        index=index, similarity_top_k=3, node_postprocessors=[reranker])
    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker._postprocess_nodes(
        query_nodes, query_bundle=query_bundle)
    for ranked_node in ranked_nodes:
        print('----------------------------------------------------')
        display_jet_source_node(ranked_node, source_length=500)
    # Print the response from the model
    response = query_engine.query(query)
    results["rerank"] = {
        "sentence": {
            "query": query,
            "response": response,
        }
    }
    save_json(results, file_path="generated/crewai/results.json")

    # Start RAG strategies

    # Set query engine for Multi-step and HyDE
    logger.log("Setting query engine for Multi-step and HyDE",
               query, colors=["GRAY", "DEBUG"])
    query_engine = IndexManager.create_query_engine(
        index=index, similarity_top_k=4)
    # Multi-step query
    logger.log("Multi-step query",
               query, colors=["GRAY", "DEBUG"])
    multi_step_response = query_processor.multi_step_query(
        llm=settings_manager.llm,
        query_engine=query_engine,
        query=query
    )
    results["multi_step"] = {
        "query": query,
        "response": multi_step_response,
    }
    save_json(results, file_path="generated/crewai/results.json")

    # HyDE transformation
    logger.log("HyDE transforming on query",
               query, colors=["GRAY", "DEBUG"])
    hyde_response = query_processor.hyde_query_transform(
        query_engine=query_engine, query=query)
    results["hyde"] = {"query": query, **make_serializable(hyde_response)}
    save_json(results, file_path="generated/crewai/results.json")

    # Start LLM generation

    def generate_query_response(query_processor: QueryProcessor, prompt: str | list[str], chat: bool = False):
        query_callback = query_processor.query_chat if chat else query_processor.query_generate
        query_response = query_callback(prompt)
        response = ""
        stream_response = []
        for chunk in query_response:
            response += chunk.delta
            stream_response.append(chunk)
            logger.success(chunk.delta, flush=True)

        result = {
            "contexts": contexts,
            "response": response,
            "stream_response": stream_response
        }
        if chat:
            result["prompt"] = prompt
        else:
            result["messages"] = prompt
        return result

    # Process questions
    for question_idx, question in enumerate(questions):
        logger.log(
            f"Processing question #{question_idx + 1}:", question, colors=["GRAY", "DEBUG"])
        contexts: list[NodeWithScore] = retriever.retrieve(question)
        context_list = [node.get_content() for node in contexts]
        prompt = query_processor.generate_prompt(
            prompt_template, context_list, question)

        logger.debug(f"Generating prompt response #{question_idx + 1}...")
        result = generate_query_response(query_processor, prompt)
        results["questions"].append(result)
        save_json(results, file_path="generated/crewai/results.json")

        # Chat response
        logger.debug(f"Generating chat response #{question_idx + 1}...")
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant."
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=prompt
            ),
        ]
        result = generate_query_response(query_processor, messages, chat=True)
        results["chats"].append(result)
        save_json(results, file_path="generated/crewai/results.json")


if __name__ == "__main__":
    main()
