import os
from typing import Generator
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import Document, NodeWithScore, BaseNode, TextNode, ImageNode
from script_utils import display_source_nodes
from jet.logger import logger
from jet.llm import call_ollama_chat
from jet.llm.llm_types import OllamaChatOptions
from jet.llm.ollama import initialize_ollama_settings, large_llm_model
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
        similarity_top_k=initial_similarity_k)

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
# Use in a Query Engine!
#
# Now, we can plug our retriever into a query engine to synthesize natural language responses.


def setup_index(
    documents: list[Document],
    data_dir: str,
    *,
    chunk_size: int = 256,
    chunk_overlap: int = 20,
):
    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_nodes = splitter.get_nodes_from_documents(documents)
    # Next, we will setup a vector index over the documentation.
    index = VectorStoreIndex.from_documents(
        documents, transformations=[splitter], show_progress=True
    )

    initial_similarity_k = len(documents)
    final_similarity_k = len(all_nodes)

    def query_nodes(
        query: str,
        fusion_mode: FUSION_MODES = FUSION_MODES.RELATIVE_SCORE,
        threshold: float = 0.0,
    ):
        # First, we create our retrievers. Each will retrieve the top-10 most similar nodes.
        retrievers = setup_retrievers(
            index, initial_similarity_k, final_similarity_k)

        fusion_retriever = get_fusion_retriever(
            retrievers, fusion_mode, final_similarity_k)

        retrieved_nodes: list[NodeWithScore] = fusion_retriever.retrieve(query)

        filtered_nodes: list[NodeWithScore] = [
            node for node in retrieved_nodes if node.score > threshold]

        unique_files = set()

        if "metadata" in filtered_nodes[0]:
            texts = [
                read_file(file_path)
                for node in filtered_nodes
                if not (file_path := os.path.join(data_dir, node.metadata['file_name'])) in unique_files
                and not unique_files.add(file_path)
            ]
        else:
            texts = [node.text for node in filtered_nodes]

        result = {
            "nodes": filtered_nodes,
            "texts": texts,
        }

        return result

    return query_nodes


def query_llm(
    query: str,
    contexts: list[str],
    model: str = "llama3.1",
    options: OllamaChatOptions = {},
    system: str = SYSTEM_MESSAGE,
    template: str = PROMPT_TEMPLATE,
    stream: bool = True,
) -> str | Generator[str, None, None]:
    context = "\n\n".join(contexts)
    prompt = template.format(
        context_str=context, query_str=query
    )
    options = {**options, **DEFAULT_CHAT_OPTIONS}

    response = ""
    for chunk in call_ollama_chat(
        prompt,
        stream=stream,
        model=model,
        system=system,
        options=options,
        track={
            "repo": "~/aim-logs",
            "experiment": "RAG Retriever Test",
            "run_name": "Run Fusion Relative Score",
            "metadata": {
                "type": "rag_retriever",
            }
        }
    ):
        response += chunk

        if stream:
            yield chunk

    if not stream:
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


def load_documents(rag_dir, extensions):
    documents = SimpleDirectoryReader(
        rag_dir, required_exts=extensions).load_data()
    return documents


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

    data_dir = "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume"
    rag_dir = "/Users/jethroestrada/Desktop/External_Projects/jet_python_modules/jet/server/rag/generated/summaries"
    extensions = [".md"]

    sample_query = "Tell me about yourself."

    documents = load_documents(rag_dir, extensions)

    query_nodes = setup_index(documents, data_dir)

    # logger.newline()
    # logger.info("RECIPROCAL_RANK: query...")
    # response = query_nodes(sample_query, FUSION_MODES.RECIPROCAL_RANK)

    # logger.newline()
    # logger.info("DIST_BASED_SCORE: query...")
    # response = query_nodes(sample_query, FUSION_MODES.DIST_BASED_SCORE)

    logger.newline()
    logger.info("RELATIVE_SCORE: sample query...")
    result = query_nodes(
        sample_query, FUSION_MODES.RELATIVE_SCORE)
    logger.info(f"RETRIEVED NODES ({len(result["nodes"])})")
    display_source_nodes(sample_query, result["nodes"])

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
            display_source_nodes(query, result["nodes"])

            response = query_llm(query, result["texts"])
            # logger.info("QUERY RESPONSE:")
            # logger.success(response)

        except KeyboardInterrupt:
            print("\nExiting query loop.")
            break
        except Exception as e:
            logger.error(f"Error while processing query: {e}")

    logger.info("\n\n[DONE]", bright=True)
