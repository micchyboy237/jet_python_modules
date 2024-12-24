import os
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, TextNode, ImageNode, NodeWithScore
from script_utils import display_source_nodes
from jet.logger import logger
from jet.llm import call_ollama_chat
from jet.llm.llm_types import OllamaChatOptions
from jet.llm.ollama import initialize_ollama_settings, large_llm_model
initialize_ollama_settings()

data_dir = "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume"

documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/summaries/jet-resume", required_exts=[".md"]).load_data()

qa_system_prompt = (
    "You are an expert Q&A system that is trusted around the world.\n"
    "Always answer the query using the provided context information, "
    "and not prior knowledge.\n"
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or "
    "'The context information ...' or anything along "
    "those lines."
)
qa_prompt = PromptTemplate(
    """\
Context information is below.
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


splitter = SentenceSplitter(chunk_size=256)
total_nodes = len(splitter.get_nodes_from_documents(documents))
initial_similarity_k = len(documents)
final_similarity_k = total_nodes

# Next, we will setup a vector index over the documentation.
index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter], show_progress=True
)


def setup_retrievers() -> list[BaseRetriever]:
    from llama_index.retrievers.bm25 import BM25Retriever

    vector_retriever = index.as_retriever(
        similarity_top_k=initial_similarity_k)

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=final_similarity_k
    )

    return [vector_retriever, bm25_retriever]


def get_fusion_retriever(retrievers: list[BaseRetriever], fusion_mode: FUSION_MODES):

    retriever = QueryFusionRetriever(
        retrievers,
        retriever_weights=[0.6, 0.4],
        similarity_top_k=final_similarity_k,
        num_queries=1,  # set this to 1 to disable query generation
        mode=fusion_mode,
        use_async=True,
        verbose=True,
    )

    return retriever
# Use in a Query Engine!
#
# Now, we can plug our retriever into a query engine to synthesize natural language responses.


def query_nodes(
    query: str,
    fusion_mode: FUSION_MODES = FUSION_MODES.RELATIVE_SCORE,
    threshold: float = 0.0
):
    # First, we create our retrievers. Each will retrieve the top-10 most similar nodes.
    retrievers = setup_retrievers()

    fusion_retriever = get_fusion_retriever(retrievers, fusion_mode)

    retrieved_nodes: list[NodeWithScore] = fusion_retriever.retrieve(query)

    filtered_nodes: list[NodeWithScore] = [
        node for node in retrieved_nodes if node.score > threshold]

    unique_files = set()
    texts = [
        read_file(file_path)
        for node in filtered_nodes
        if not (file_path := os.path.join(data_dir, node.metadata['file_name'])) in unique_files
        and not unique_files.add(file_path)
    ]

    return {
        "nodes": filtered_nodes,
        "retriever": fusion_retriever,
        "texts": texts,
        "files": list(unique_files),
    }


def query_llm(
    query: str,
    contexts: list[str],
    model: str = "llama3.1",
    options: OllamaChatOptions = {},
    # retriever: QueryFusionRetriever,
):
    # query_engine = RetrieverQueryEngine.from_args(retriever, text_qa_template=)
    # response = query_engine.query(query)
    # return response

    context = "\n\n".join(contexts)
    prompt = qa_prompt.format(
        context_str=context, query_str=query
    )
    options = {**options, **DEFAULT_CHAT_OPTIONS}

    response = ""
    for chunk in call_ollama_chat(
        prompt,
        stream=True,
        model=model,
        system=qa_system_prompt,
        options=options,
        track={
            "repo": "./aim-logs",
            "experiment": "RAG Retriever Test",
            "run_name": "Run Fusion Relative Score",
            "metadata": {
                "type": "rag_retriever",
            }
        }
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


if __name__ == "__main__":
    sample_query = "Tell me about yourself."

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
    logger.info("QUERY RESPONSE:")
    logger.success(response)

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

            response = query_llm(query, result['texts'])
            logger.info("QUERY RESPONSE:")
            logger.success(response)

        except KeyboardInterrupt:
            print("\nExiting query loop.")
            break
        except Exception as e:
            logger.error(f"Error while processing query: {e}")

    logger.info("\n\n[DONE]", bright=True)
