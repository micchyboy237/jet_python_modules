import os
from typing import Any, Callable, Literal, Optional
from jet.llm.ollama.base import OllamaEmbedding
from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL, OLLAMA_SMALL_LLM_MODEL
from jet.llm.ollama.embeddings import get_ollama_embedding_function
from jet.llm.ollama.models import OLLAMA_MODEL_EMBEDDING_TOKENS, OLLAMA_MODEL_NAMES
from jet.llm.query.splitters import split_heirarchical_nodes, split_sub_nodes
from jet.llm.retrievers.recursive import (
    initialize_summary_nodes_and_retrievers,
    query_nodes as query_nodes_recursive
)
from jet.token import filter_texts
from jet.token.token_utils import get_ollama_tokenizer
from jet.vectors.node_parser.hierarchical import JetHierarchicalNodeParser
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.indices import vector_store
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.node_parser.relational.hierarchical import get_leaf_nodes
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.retrievers.recursive_retriever import RecursiveRetriever
from llama_index.core.schema import Document, NodeWithScore, BaseNode, TextNode, ImageNode
from jet.llm.utils import display_jet_source_nodes
from jet.logger import logger
from jet.llm import call_ollama_chat
from jet.llm.llm_types import OllamaChatOptions
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()


SYSTEM_MESSAGE = "You are a helpful AI Assistant."

PROMPT_TEMPLATE = PromptTemplate(
    """\
Context information are below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \
"""
)

DEFAULT_CHAT_OPTIONS: OllamaChatOptions = {
    "seed": 44,
    "num_ctx": 4096,
    "num_keep": 0,
    "num_predict": -1,
    "temperature": 0,
}


def setup_retrievers(index: VectorStoreIndex, initial_similarity_k: int, final_similarity_k: int) -> list[BaseRetriever]:
    from llama_index.retrievers.bm25 import BM25Retriever

    vector_retriever = index.as_retriever(
        similarity_top_k=initial_similarity_k,
    )

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=final_similarity_k
    )

    return [vector_retriever, bm25_retriever]


def get_fusion_retriever(retrievers: list[BaseRetriever], fusion_mode: FUSION_MODES, final_similarity_k: int):

    retriever = QueryFusionRetriever(
        retrievers,
        retriever_weights=[0.6, 0.4],
        similarity_top_k=final_similarity_k,
        num_queries=1,  # set this to 1 to disable query generation
        mode=fusion_mode,
        use_async=False,
        verbose=True,

    )

    return retriever


def get_recursive_retriever(index: VectorStoreIndex, all_nodes: list[BaseNode], similarity_top_k: Optional[int] = None):
    vector_retriever_chunk = index.as_retriever(
        similarity_top_k=similarity_top_k)

    all_nodes_dict = {n.node_id: n for n in all_nodes}

    retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_chunk},
        node_dict=all_nodes_dict,
        verbose=False,
    )

    return retriever

# Use in a Query Engine!
#
# Now, we can plug our retriever into a query engine to synthesize natural language responses.


def load_documents(data_dir: str, extensions: Optional[list[str]] = None):
    documents = SimpleDirectoryReader(
        data_dir, required_exts=extensions, recursive=True).load_data()
    return documents


def setup_index(
    path_or_docs: str | list[Document],
    *,
    extensions: Optional[list[str]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    sub_chunk_sizes: Optional[list[int]] = None,
    with_heirarchy: Optional[bool] = None,
    embed_model: Optional[str] = OLLAMA_SMALL_EMBED_MODEL,
    mode: Optional[Literal["fusion", "heirarchy", "deeplake"]] = "fusion",
    **kwargs
):
    search_func: Callable[..., dict[str, Any]]

    final_chunk_size: int = chunk_size if isinstance(
        chunk_size, int) else OLLAMA_MODEL_EMBEDDING_TOKENS[embed_model]

    documents: list[Document]
    if type(path_or_docs) == str:
        documents = load_documents(path_or_docs, extensions)
    elif isinstance(path_or_docs, list):
        documents = path_or_docs
    else:
        raise ValueError(f"'data_dir' must be of type str | list[Document]")

    splitter = SentenceSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=get_ollama_tokenizer(embed_model).encode
    )
    all_nodes = splitter.get_nodes_from_documents(
        documents, show_progress=True)

    # if mode == "summary":
    # all_nodes = split_sub_nodes(all_nodes, [128, 256, 512], chunk_overlap)
    if mode == "heirarchy":
        index = VectorStoreIndex(
            all_nodes,
            show_progress=True,
        )

        if not sub_chunk_sizes:
            sub_chunk_sizes = [final_chunk_size]
        # sub_chunk_sizes = [chunk_size, *sub_chunk_sizes]
        sub_nodes = split_heirarchical_nodes(
            all_nodes, sub_chunk_sizes, chunk_overlap)
        jet_node_parser = JetHierarchicalNodeParser(sub_nodes, sub_chunk_sizes)
        all_nodes = jet_node_parser.all_nodes
        # leaf_nodes = get_leaf_nodes(sub_nodes)
        # leaf_nodes = get_leaf_nodes(jet_node_parser.all_nodes)
        # all_nodes = leaf_nodes

        def search_heirarchy_func(
            query: str,
            fusion_mode: FUSION_MODES = FUSION_MODES.RELATIVE_SCORE,
            threshold: float = 0.0,
            top_k: int = 10,
        ):
            # First, we create our retrievers. Each will retrieve the top-10 most similar nodes.

            combined_retriever = get_recursive_retriever(
                index, all_nodes, top_k)
            # else:
            # initial_similarity_k = len(documents)
            # final_similarity_k = len(all_nodes)

            retrieved_nodes: list[NodeWithScore] = combined_retriever.retrieve(
                query)

            filtered_nodes: list[NodeWithScore] = [
                node for node in retrieved_nodes if node.score > threshold]

            texts = [node.text for node in filtered_nodes]

            result = {
                "nodes": filtered_nodes,
                "texts": texts,
            }

            return result

        search_func = search_heirarchy_func

    elif mode == "deeplake":
        store_path = kwargs["store_path"]
        texts = [node.text for node in all_nodes]
        metadata = [node.metadata for node in all_nodes]
        embedding_function = get_ollama_embedding_function(embed_model)
        args = {
            "store_path": store_path,
            "texts": texts,
            "metadata": metadata,
            "embedding_function": embedding_function,
            "overwrite": kwargs.get("overwrite", True),
            "verbose": kwargs.get("verbose", False),
        }
        vector_store = load_or_create_deeplake_vector_store(**args)

        def search_deeplake_func(
            query: str,
            threshold: float = 0.0,
            top_k: int = 4,
        ):
            results = vector_store.search(
                embedding_data=query,
                k=top_k,
            )
            results["text"] = [str(text) for text in results["text"]]

            # Process search results into NodeWithScore format
            nodes_with_scores = [
                NodeWithScore(
                    node=TextNode(text=str(text), metadata=metadata),
                    score=extract_score(score)
                )
                for text, metadata, score in zip(results["text"], results["metadata"], results["score"])
            ]

            result = {
                "nodes": nodes_with_scores,
                "texts": results["text"],
            }

            return result

        search_func = search_deeplake_func

    else:
        index = VectorStoreIndex(
            all_nodes,
            show_progress=True,
        )

        def search_fusion_func(
            query: str,
            fusion_mode: FUSION_MODES = FUSION_MODES.RELATIVE_SCORE,
            threshold: float = 0.0,
            top_k: int = 10,
        ):

            initial_similarity_k = top_k if top_k < len(
                all_nodes) else len(all_nodes)
            final_similarity_k = top_k if top_k < len(
                all_nodes) else len(all_nodes)
            retrievers = setup_retrievers(
                index, initial_similarity_k, final_similarity_k)
            combined_retriever = get_fusion_retriever(
                retrievers, fusion_mode, top_k)

            retrieved_nodes: list[NodeWithScore] = combined_retriever.retrieve(
                query)

            filtered_nodes: list[NodeWithScore] = [
                node for node in retrieved_nodes if node.score > threshold]
            if top_k:
                filtered_nodes = filtered_nodes[:top_k]

            texts = [node.text for node in filtered_nodes]

            result = {
                "nodes": filtered_nodes,
                "texts": texts,
            }

            return result

        search_func = search_fusion_func

    return search_func


def get_relative_path(abs_path: str, partial_path: str) -> str:
    """
    Extracts the relative path from the absolute path using the given partial path as the starting point.

    :param abs_path: The absolute path to process.
    :param partial_path: The partial path to locate within the absolute path.
    :return: The resulting relative path starting from the partial path.
    """
    if partial_path in abs_path:
        start_index = abs_path.index(partial_path)
        return abs_path[start_index:]
    else:
        raise ValueError(f"Partial path '{
                         partial_path}' not found in the absolute path '{abs_path}'.")


def query_llm(
    query: str,
    contexts: list[str],
    model: Optional[OLLAMA_MODEL_NAMES] = OLLAMA_SMALL_LLM_MODEL,
    options: OllamaChatOptions = {},
    system: str = SYSTEM_MESSAGE,
    template: PromptTemplate = PROMPT_TEMPLATE,
    max_tokens: Optional[int | float] = None,
    # retriever: QueryFusionRetriever,
):
    # query_engine = RetrieverQueryEngine.from_args(retriever, text_qa_template=)
    # response = query_engine.query(query)
    # return response

    filtered_texts = filter_texts(
        contexts, model, max_tokens=max_tokens)
    context = "\n\n".join(filtered_texts)
    prompt = template.format(
        context_str=context, query_str=query
    )
    options = {**options, **DEFAULT_CHAT_OPTIONS}

    response = ""
    for chunk in call_ollama_chat(
        prompt,
        stream=True,
        model=model,
        system=system,
        options=options,
        # track={
        #     "repo": "~/aim-logs",
        #     "experiment": "RAG Retriever Test",
        #     "run_name": "Run Fusion Relative Score",
        #     "metadata": {
        #         "type": "rag_retriever",
        #     }
        # }
    ):
        response += chunk
    return response


def read_file(file_path, start_index=None, end_index=None):
    with open(file_path, 'r') as file:
        if not any([start_index, end_index]):
            content = file.read()
        else:
            file.seek(start_index)  # Move to the start index
            # Read only up to the end index
            content = file.read(end_index - start_index)
    return content


def setup_recursive_query(
    *args,
    chunk_size: int = 256,
    chunk_overlap: int = 20,
    **kwargs,
):
    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    summary_nodes, vector_retrievers = initialize_summary_nodes_and_retrievers(
        *args,  transformations=[splitter], **kwargs)

    def query_nodes_func(
        query: str,
        threshold: float = 0.0,
        top_k: int = 4,
    ):
        retrieved_nodes = query_nodes_recursive(
            query,
            summary_nodes,
            vector_retrievers,
            transformations=[splitter],
            similarity_top_k=top_k,
        )

        filtered_nodes: list[NodeWithScore] = [
            node for node in retrieved_nodes if node.score > threshold]

        texts = [node.text for node in filtered_nodes]

        result = {
            "nodes": filtered_nodes,
            "texts": texts,
        }

        return result

    return query_nodes_func


def setup_deeplake_query(
    data_dir: str | list[Document],
    store_path: str,
    *,
    embed_model: str = OLLAMA_SMALL_EMBED_MODEL,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 40,
    overwrite=True,
    **kwargs,
):
    final_chunk_size: int = chunk_size if isinstance(
        chunk_size, int) else OLLAMA_MODEL_EMBEDDING_TOKENS[embed_model]
    # Define file and vector store paths
    documents: list[Document]
    if type(data_dir) == str:
        documents = SimpleDirectoryReader(data_dir).load_data()
    elif isinstance(data_dir, list):
        documents = data_dir
    else:
        raise ValueError(f"'data_dir' must be of type str | list[Document]")
    splitter = SentenceSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=get_ollama_tokenizer(embed_model).encode
    )
    all_nodes = splitter.get_nodes_from_documents(
        documents, show_progress=True)

    texts = [node.text for node in all_nodes]
    metadata = [node.metadata for node in all_nodes]

    embedding_function = get_ollama_embedding_function(embed_model)

    # Create a VectorStore instance
    vector_store = load_or_create_deeplake_vector_store(
        store_path,
        texts=texts,
        metadata=metadata,
        embedding_function=embedding_function,
        overwrite=overwrite,
        verbose=True,
    )

    def query_nodes_func(
        query: str,
        threshold: float = 0.0,
        top_k: int = 4,
    ):
        search_results = search_deeplake_store(
            query, store_path, top_k=top_k, embedding_function=embedding_function)

        texts = [node.text for node in search_results]

        result = {
            "nodes": search_results,
            "texts": texts,
        }

        return result

    return query_nodes_func


def load_or_create_deeplake_vector_store(
    store_path: str,
    texts: list[str],
    metadata: Optional[list[str]] = None,
    embedding_function: Optional[Callable] = None,
    overwrite: bool = False,
    **kwargs
):
    from deeplake.core.vectorstore import VectorStore
    logger.log("Vector store path:", os.path.realpath(store_path),
               colors=["GRAY", "BRIGHT_DEBUG"])

    os.makedirs(store_path, exist_ok=True)

    # Create a VectorStore instance
    vector_store = VectorStore(
        path=store_path,
        embedding_function=embedding_function,
        overwrite=overwrite,
        **kwargs
    )

    if overwrite:
        # Add text chunks to the vector store with embeddings
        vector_store.add(
            text=texts,
            embedding_function=embedding_function,
            embedding_data=texts,
            metadata=metadata,
        )

    return vector_store


def search_deeplake_store(
    prompt: str,
    store_path: str,
    top_k=4,
    embedding_function: Optional[Callable] = None,
) -> list[NodeWithScore]:
    from deeplake.core.vectorstore import VectorStore
    vector_store = VectorStore(
        path=store_path, embedding_function=embedding_function)

    results = vector_store.search(
        embedding_data=prompt,
        k=top_k,
    )
    results["text"] = [str(text) for text in results["text"]]

    # Process search results into NodeWithScore format
    nodes_with_scores = [
        NodeWithScore(
            node=TextNode(text=str(text), metadata=metadata),
            score=extract_score(score)
        )
        for text, metadata, score in zip(results["text"], results["metadata"], results["score"])
    ]
    return nodes_with_scores


def extract_score(score) -> float:
    # Check if score is a tensor and convert it to float
    if hasattr(score, 'item'):  # e.g., if it's a tensor
        return score.item()
    elif isinstance(score, list) and len(score) == 1:
        # If it's a list with a single score, convert to float
        return float(score[0])
    return float(score)  # If it's already a float, return it directly


if __name__ == "__main__":
    system = (
        "You are a job applicant providing tailored responses during an interview.\n"
        "Always answer questions using the provided context as if it is your resume, "
        "and avoid referencing the context directly.\n"
        "Some rules to follow:\n"
        "1. Never directly mention the context or say 'According to my resume' or similar phrases.\n"
        "2. Provide responses as if you are the individual described in the context, focusing on professionalism and relevance."
    )

    prompt_template = PromptTemplate(
        """\
    Resume details are below.
    ---------------------
    {context_str}
    ---------------------
    Given the resume details and not prior knowledge, respond to the question.
    Question: {query_str}
    Response: \
    """
    )

    data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    rag_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    extensions = [".md"]

    sample_query = "Tell me about yourself."

    documents = SimpleDirectoryReader(
        rag_dir, required_exts=extensions).load_data()

    query_nodes = setup_index(documents)

    # logger.newline()
    # logger.info("RECIPROCAL_RANK: query...")
    # response = query_nodes(sample_query, FUSION_MODES.RECIPROCAL_RANK)

    # logger.newline()
    # logger.info("DIST_BASED_SCORE: query...")
    # response = query_nodes(sample_query, FUSION_MODES.DIST_BASED_SCORE)

    logger.newline()
    logger.info("RELATIVE_SCORE: sample query...")
    result = query_nodes(
        sample_query, FUSION_MODES.RELATIVE_SCORE, threshold=0.2)
    logger.info(f"RETRIEVED NODES ({len(result["nodes"])})")
    display_jet_source_nodes(sample_query, result["nodes"])

    response = query_llm(sample_query, result['texts'])
    # logger.info("QUERY RESPONSE:")
    # logger.success(response)

    # Run app
    while True:
        # Continuously ask user for queries
        try:
            query = input("Enter your query (type 'exit' to quit): ").strip()
            if query.lower() == "exit":
                print("Exiting query loop.")
                break

            result = query_nodes(query, FUSION_MODES.RELATIVE_SCORE)
            logger.info(f"RETRIEVED NODES ({len(result["nodes"])})")
            display_jet_source_nodes(query, result["nodes"])

            response = query_llm(query, result['texts'])
            # logger.info("QUERY RESPONSE:")
            # logger.success(response)

        except KeyboardInterrupt:
            print("\nExiting query loop.")
            break
        except Exception as e:
            logger.error(f"Error while processing query: {e}")

    logger.info("\n\n[DONE]", bright=True)
