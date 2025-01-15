import os
from pathlib import Path
from typing import Optional

from jet.cache.joblib.utils import load_from_cache_or_compute
from jet.llm.ollama.base import Ollama
from jet.logger.timer import time_it
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.retrievers.recursive_retriever import RecursiveRetriever
from llama_index.core.schema import IndexNode, TransformComponent
from tqdm import tqdm
from jet.llm.utils import display_jet_source_nodes
from jet.logger import logger


def get_rag_data(data_dir: str):
    rag_titles = []
    rag_metadatas = {}

    rag_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    for file in rag_files:
        file_name = os.path.basename(file)

        title = file_name
        rag_titles.append(title)
        rag_metadatas[title] = {
            "file_name": file_name,
            "file_path": file
        }

    docs_dict = {}
    for rag_title in rag_titles:
        file_path = Path(rag_metadatas[rag_title]['file_path']).resolve()
        if not str(data_dir) in rag_metadatas[rag_title]['file_path']:
            continue

        doc = SimpleDirectoryReader(
            input_files=[file_path],
        ).load_data()[0]
        doc.metadata.update(rag_metadatas[rag_title])
        docs_dict[rag_title] = doc

    return {
        "titles": rag_titles,
        "data": docs_dict,
    }


def build_recursive_retriever_over_document_summaries(
    data_dir: str,
    similarity_top_k=4,
    output_dir: str = "generated",
    transformations: Optional[list[TransformComponent]] = None,
    llm: Optional[Ollama] = None
):
    if not llm:
        llm = Ollama(model="mistral", request_timeout=300.0,
                     context_window=4096)

    rag_data = get_rag_data(data_dir)

    out_dir = Path(output_dir)
    # Build Recursive Retriever over Document Summaries
    summary_nodes = []
    vector_query_engines = {}
    vector_retrievers = {}

    # Filter rag_titles to only those without existing summaries
    existing_summaries = set(
        p.stem for p in Path(out_dir).rglob("*.md")
    )
    titles_to_process = [
        title for title in rag_data["titles"] if title not in existing_summaries]

    with tqdm(titles_to_process, total=len(titles_to_process)) as pbar:
        for rag_title in pbar:
            # Update the description with the current wiki title
            pbar.set_description(f"Processing: {rag_title}")

            # Build vector index and retriever for the title
            vector_index = VectorStoreIndex.from_documents(
                [rag_data["data"][rag_title]],
                transformations=transformations,
                # callback_manager=callback_manager,
            )
            vector_query_engine = vector_index.as_query_engine(llm=llm)
            vector_query_engines[rag_title] = vector_query_engine
            vector_retrievers[rag_title] = vector_index.as_retriever(
                similarity_top_k=similarity_top_k
            )

            # Generate summary
            summary_index = SummaryIndex.from_documents(
                documents=[rag_data["data"][rag_title]],
                # callback_manager=callback_manager,
                show_progress=True,
            )
            summarizer = summary_index.as_query_engine(
                response_mode="tree_summarize", llm=llm
            )

            logger.newline()
            logger.log("Summary for", rag_title,
                       "...", colors=["WHITE", "INFO"])
            response = summarizer.query(
                f"Summarize the contents of this document.")

            logger.log(
                "Summary nodes (tree_summarize):",
                f"({len(response.source_nodes)})",
                colors=["WHITE", "SUCCESS"]
            )
            display_jet_source_nodes(rag_title, response.source_nodes)

            rag_summary = response.response
            with open(out_dir / rag_title, "a") as fp:
                fp.write(rag_summary)

            node = IndexNode(text=rag_summary, index_id=rag_title)
            summary_nodes.append(node)

    logger.log("Summary nodes:", len(summary_nodes),
               colors=["WHITE", "SUCCESS"])

    return summary_nodes, vector_retrievers


def initialize_summary_nodes_and_retrievers(
    data_dir: str,
    use_cache=False,
    similarity_top_k: int = 4,
    output_dir: str = "generated",
    transformations: Optional[list[TransformComponent]] = None,
    llm: Optional[Ollama] = None
):
    """Load cached data or compute if not available."""

    summary_nodes, vector_retrievers = load_from_cache_or_compute(
        build_recursive_retriever_over_document_summaries,
        file_path=os.path.join(output_dir, "summaries_and_retrievers.pkl"),
        use_cache=use_cache,
        data_dir=data_dir,
        similarity_top_k=similarity_top_k,
        output_dir=output_dir,
        transformations=transformations,
        llm=llm,
    )

    return summary_nodes, vector_retrievers


@time_it
def query_nodes(
    query,
    nodes,
    vector_retrievers,
    transformations: Optional[list[TransformComponent]] = None,
    similarity_top_k: int = 3,
):
    logger.debug(f"Querying ({len(nodes)}) nodes...")
    top_vector_index = VectorStoreIndex(
        nodes, transformations=transformations
    )
    top_vector_retriever = top_vector_index.as_retriever(
        similarity_top_k=similarity_top_k)

    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": top_vector_retriever, **vector_retrievers},
        verbose=False,
    )

    retrieved_nodes = recursive_retriever.retrieve(query)

    # Sort retrieved_nodes by item.score in reverse order
    retrieved_nodes_sorted = sorted(
        retrieved_nodes, key=lambda item: item.score, reverse=True)

    logger.log("Query:", query, colors=["WHITE", "INFO"])
    logger.log(
        "Retrieved summary nodes (RecursiveRetriever):",
        f"({len(retrieved_nodes_sorted)})",
        colors=["WHITE", "SUCCESS"]
    )

    return retrieved_nodes_sorted
