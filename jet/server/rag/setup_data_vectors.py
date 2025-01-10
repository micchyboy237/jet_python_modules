import joblib
import json
import os
import shutil

import requests
from pathlib import Path
from tqdm import tqdm
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode, NodeWithScore
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core import StorageContext
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core import SummaryIndex
from llama_index.core import SimpleDirectoryReader
import sys
import logging
from jet.llm.utils import display_jet_source_nodes
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings, Ollama
initialize_ollama_settings()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

GENERATED_DIR = Path("generated")
DEFAULT_BASE_DIR = os.path.basename(__file__).split(".")[0]
DEFAULT_DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"

DEFAULT_MODEL = "mistral"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50


def setup_paths(
    data_dir: str = DEFAULT_DATA_DIR,
    generated_dir: str = GENERATED_DIR,
    model: str = DEFAULT_MODEL,
    chunk_size: str = DEFAULT_CHUNK_SIZE,
    chunk_overlap: str = DEFAULT_CHUNK_OVERLAP,
):
    cache_dir = Path(generated_dir) / "cache"
    summary_nodes_cache = cache_dir / "summary_nodes.pkl"
    vector_retrievers_cache = cache_dir / "vector_retrievers.pkl"

    # Ensure the cache directory exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    llm = Ollama(model=model, request_timeout=300.0, context_window=4096)
    callback_manager = CallbackManager([LlamaDebugHandler()])
    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Setup output dir
    out_dir = generated_dir / "summaries"
    # Reset out_dir if it exists
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # Create a new empty out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    return llm, data_dir, out_dir, summary_nodes_cache, vector_retrievers_cache, splitter, callback_manager


def create_dataset(data_dir):
    # Read rag files
    documents = SimpleDirectoryReader(
        data_dir, required_exts=[".md"], recursive=True).load_data()
    texts = [doc.text for doc in documents]

    combined_file_path = os.path.join(data_dir, "combined.txt")
    with open(combined_file_path, "w") as f:
        f.write("\n\n\n".join(texts))

    include_files = [
        ".md",
        # "combined.txt",
    ]  # Add filenames to include here
    exclude_files = []  # Add patterns or filenames to exclude here

    rag_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    # Apply include_files filter
    if include_files:
        rag_files = [
            file for file in rag_files
            if any(include in file for include in include_files)
        ]

    # Apply exclude_files filter
    if exclude_files:
        rag_files = [
            file for file in rag_files
            if not any(exclude in file for exclude in exclude_files)
        ]

    # Print the filtered list
    logger.log("rag_files:", len(rag_files), colors=["WHITE", "DEBUG"])
    logger.debug(json.dumps(rag_files, indent=2))

    data_path = Path(data_dir)

    wiki_titles = []
    wiki_metadatas = {}

    for file in rag_files:
        with open(file) as f:
            content = f.read()
            file_name = os.path.basename(file)

            title = file_name
            wiki_titles.append(title)
            wiki_metadatas[title] = {
                "file_name": file_name,
                "file_path": file
            }

    docs_dict = {}
    for wiki_title in wiki_titles:
        file_path = Path(wiki_metadatas[wiki_title]['file_path']).resolve()
        if not str(data_dir) in wiki_metadatas[wiki_title]['file_path']:
            continue

        doc = SimpleDirectoryReader(
            input_files=[file_path],
        ).load_data()[0]
        doc.metadata.update(wiki_metadatas[wiki_title])
        docs_dict[wiki_title] = doc

    return docs_dict, wiki_titles


def build_recursive_retriever_over_document_summaries(
    similarity_top_k=3,
    out_dir: Path = "",
    wiki_titles: list[str] = [],
    transformations: list = [],
    callback_manager: CallbackManager = None,
    llm: Ollama = None,
):
    # Build Recursive Retriever over Document Summaries
    summary_nodes = []
    vector_query_engines = {}
    vector_retrievers = {}

    # Filter wiki_titles to only those without existing summaries
    existing_summaries = set(
        p.stem for p in Path(out_dir).rglob("*.txt")
    )
    titles_to_process = [
        title for title in wiki_titles if title not in existing_summaries]

    with tqdm(titles_to_process, total=len(titles_to_process)) as pbar:
        for wiki_title in pbar:
            # Update the description with the current wiki title
            pbar.set_description(f"Processing: {wiki_title}")

            summary_file = out_dir / wiki_title

            if summary_file.exists():
                with open(summary_file, "r") as f:
                    wiki_summary = f.read()
            else:
                # Build vector index and retriever for the title
                vector_index = VectorStoreIndex.from_documents(
                    [docs_dict[wiki_title]],
                    transformations=transformations,
                    callback_manager=callback_manager,
                )
                vector_query_engine = vector_index.as_query_engine(llm=llm)
                vector_query_engines[wiki_title] = vector_query_engine
                vector_retrievers[wiki_title] = vector_index.as_retriever(
                    similarity_top_k=similarity_top_k
                )

                # Generate summary
                summary_index = SummaryIndex.from_documents(
                    documents=[docs_dict[wiki_title]],
                    callback_manager=callback_manager,
                    show_progress=True,
                )
                summarizer = summary_index.as_query_engine(
                    response_mode="tree_summarize", llm=llm
                )

                logger.newline()
                logger.log("Summary for", wiki_title,
                           "...", colors=["WHITE", "INFO"])
                response = summarizer.query(
                    f"Summarize the contents of this document.")

                logger.log(
                    "Summary nodes (tree_summarize):",
                    f"({len(response.source_nodes)})",
                    colors=["WHITE", "SUCCESS"]
                )
                display_jet_source_nodes(wiki_title, response.source_nodes)

                wiki_summary = response.response
                with open(summary_file, "a") as fp:
                    fp.write(wiki_summary)

            node = IndexNode(text=wiki_summary, index_id=wiki_title)
            summary_nodes.append(node)

    logger.log("Summary nodes:", len(summary_nodes),
               colors=["WHITE", "SUCCESS"])

    return summary_nodes, vector_retrievers


@time_it
def query_nodes(query, nodes, vector_retrievers, similarity_top_k=3) -> list[NodeWithScore]:
    logger.debug(f"Querying ({len(nodes)}) nodes...")
    top_vector_index = VectorStoreIndex(
        nodes, transformations=[splitter], callback_manager=callback_manager
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


def load_from_cache_or_compute(
    llm: Ollama = None,
    wiki_titles: list[str] = [],
    similarity_top_k=3,
    use_cache=False,
    out_dir: Path = Path(""),
    summary_nodes_cache: Path = Path(""),
    vector_retrievers_cache: Path = Path(""),
    transformations: list = [],
    callback_manager: CallbackManager = None,
):
    """Load cached data or compute if not available."""
    if use_cache and summary_nodes_cache.exists() and vector_retrievers_cache.exists():
        summary_nodes = joblib.load(summary_nodes_cache)
        vector_retrievers = joblib.load(vector_retrievers_cache)
        logger.success("Cache hit! Loaded data.")
    else:
        logger.debug("Cache not found. Building data...")
        summary_nodes, vector_retrievers = build_recursive_retriever_over_document_summaries(
            similarity_top_k=similarity_top_k,
            out_dir=out_dir,
            wiki_titles=wiki_titles,
            transformations=transformations,
            callback_manager=callback_manager,
            llm=llm,
        )
        joblib.dump(summary_nodes, summary_nodes_cache)
        joblib.dump(vector_retrievers, vector_retrievers_cache)
        logger.success("Data cached successfully.")

    return summary_nodes, vector_retrievers


if __name__ == "__main__":
    use_cache = True
    similarity_top_k = 4

    llm, data_dir, out_dir, summary_nodes_cache, vector_retrievers_cache, splitter, callback_manager = setup_paths()
    docs_dict, wiki_titles = create_dataset(data_dir)

    # main_metadata_filters_and_auto_retrieval()
    logger.debug("Building recursive retriever over document summaries...")
    summary_nodes, vector_retrievers = load_from_cache_or_compute(
        llm=llm,
        wiki_titles=wiki_titles,
        similarity_top_k=similarity_top_k,
        use_cache=use_cache,
        out_dir=out_dir,
        summary_nodes_cache=summary_nodes_cache,
        vector_retrievers_cache=vector_retrievers_cache,
        transformations=[splitter],
        callback_manager=callback_manager,
    )

    # Sample usage
    query_top_k = len(summary_nodes)

    query = "Tell me about yourself."
    retrieved_nodes = query_nodes(
        query,
        summary_nodes,
        vector_retrievers,
        similarity_top_k=query_top_k,
    )
    display_jet_source_nodes(query, retrieved_nodes)
    for node_idx, node in enumerate(retrieved_nodes):
        logger.log(
            f"{node_idx + 1}", f"{node.metadata['file_name']}:",
            f"{node.score * 100:.2f}%",
            colors=["INFO", "DEBUG", "SUCCESS"],
        )

    retrieved_contents = []
    for node in retrieved_nodes:
        retrieved_contents.append(node.node.get_content())

    result = "\n\n".join(retrieved_contents)
    # logger.newline()
    # logger.log("Final result:")
    # logger.success(result)

    # Run app
    while True:
        # Continuously ask user for queries
        try:
            query = input("Enter your query (type 'exit' to quit): ").strip()
            if query.lower() == "exit":
                print("Exiting query loop.")
                break

            retrieved_nodes = query_nodes(
                query,
                summary_nodes,
                vector_retrievers,
                similarity_top_k=query_top_k,
            )
            display_jet_source_nodes(query, retrieved_nodes)
            for node_idx, node in enumerate(retrieved_nodes):
                logger.log(
                    f"{node_idx + 1}", f"{node.metadata['file_name']}:",
                    f"{node.score * 100:.2f}%",
                    colors=["INFO", "DEBUG", "SUCCESS"],
                )

            result = "\n\n".join(retrieved_contents)
            # logger.newline()
            # logger.log("Final result:")
            # logger.success(result)

        except KeyboardInterrupt:
            print("\nExiting query loop.")
            break
        except Exception as e:
            logger.error(f"Error while processing query: {e}")
