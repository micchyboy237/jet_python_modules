"""
Title: Shared Themes Discovery in Job Postings using BERTopic

Definition Summary:
This script performs topic modeling on job postings to discover shared themes and clusters.
It loads job data from a PostgreSQL database, generates embeddings using a local LLM model,
and applies the BERTopic pipeline (UMAP + HDBSCAN + TF-IDF) to identify common topics.
Results are saved to a dedicated output directory for analysis.

Usage Examples:
    # Run with default settings (last 14 days)
    python example_shared_themes_jobs.py

    # Modify days_back in the script to change the time window
    # Ensure PHOENIX_BASE_URL is set in environment or config for telemetry

Span Hierarchy:
📦 shared-themes-discovery (CHAIN)
│
├── 📦 load-job-documents (CHAIN)
│   └── 🔍 SQL Query Execution
│       ├── SELECT * FROM public.jobs WHERE posted_date >= %s
│       └── Returns: list[str] of formatted job texts
│
├── 🧬 generate-embeddings (EMBEDDING)
│   ├── attr: embedding.model_name = "nomic-embed:2-moe" (or configured EMBED_MODEL)
│   ├── attr: embedding.text_count = <number_of_documents>
│   └── attr: embedding.vector_dimension = 768
│
└── 📦 bertopic-modeling (CHAIN)
    ├── 📦 umap-dimensionality-reduction (TOOL)
    │   └── attr: umap.n_components = 5
    │
    ├── 📦 hdbscan-clustering (TOOL)
    │   └── attr: hdbscan.min_cluster_size = 2
    │
    └── 📦 tfidf-vectorization (TOOL)
        └── attr: vectorizer.max_features = 10000
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Initialize patches before importing BERTopic
from jet.libs.bertopic.monkey_patches.add_check_array import init_patch

init_patch()

from bertopic import BERTopic
from hdbscan import HDBSCAN
from jet.adapters.llama_cpp.chunking_utils import truncate_texts
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.db.postgres.pgvector import PgVectorClient
from jet.logger import logger
from jet_telemetry import chain, embedding, get_trace_url, initialize_telemetry
from shared.job_helpers import DEFAULT_JOBS_DB_NAME, load_jobs_list
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

# --- Telemetry Setup ---
# Ensure telemetry is initialized. In a real app, this might be done at entry point.
# We use a generic service name here if not already set by the environment.
try:
    initialize_telemetry(service_name="shared-themes-demo")
except Exception:
    logger.warning("Telemetry already initialized or failed to initialize.")

# --- Output Directory Configuration ---
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Output directory created at: {OUTPUT_DIR}")


@chain(name="load-job-documents")
def load_documents(
    days_back: int = 14,
    dbname: str = DEFAULT_JOBS_DB_NAME,
    max_tokens=500,
) -> list[str]:
    """
    Loads job titles and details from the database for the specified number of days back.

    Args:
        days_back: Number of days to look back for jobs.
        dbname: Name of the PostgreSQL database.

    Returns:
        A list of strings, each containing the title and details of a job.
    """
    logger.info(f"Loading jobs from last {days_back} days...")
    db_client = PgVectorClient(dbname=dbname)
    days_ago = datetime.now() - timedelta(days=days_back)

    # Load jobs with basic metadata. We don't need entities for topic modeling.
    jobs = load_jobs_list(
        db_client=db_client, posted_after=days_ago, include_entities=False
    )

    logger.info(f"Loaded {len(jobs)} jobs from database.")

    # Format documents for BERTopic: Combine Title and Details
    docs = []
    for j in jobs:
        title = j.get("title", "")
        details = j.get("details", "")
        if title and details:
            # Limit details length to avoid excessive noise in topic modeling
            clean_details = details[:2000] if len(details) > 2000 else details
            docs.append(f"{title}\n{clean_details}")

    logger.info(f"Prepared {len(docs)} documents for embedding.")

    truncated_docs = truncate_texts(
        texts=docs,
        model=EMBED_MODEL,
        max_tokens=max_tokens,
        strict_sentences=True,
        show_progress=False,
    )
    return truncated_docs


@embedding(model_name=EMBED_MODEL)
def generate_embeddings(documents: list[str], model: str) -> any:
    """
    Generates embeddings for a list of documents using the specified model.

    Args:
        documents: List of text documents.
        model: The embedding model key.

    Returns:
        Numpy array of embeddings.
    """
    logger.info(f"Encoding {len(documents)} documents using local model: {model}")
    embeddings = embed(
        text=documents, model=model, return_format="numpy", show_progress=True
    )
    logger.info(f"Generated embedding matrix shape: {embeddings.shape}")
    return embeddings


@chain(name="bertopic-modeling")
def run_bertopic_pipeline(documents: list[str], embeddings: any):
    """
    Runs the BERTopic pipeline: UMAP -> HDBSCAN -> TF-IDF -> Topic Extraction.

    Args:
        documents: List of text documents.
        embeddings: Numpy array of document embeddings.
    """
    logger.info("Configuring BERTopic Pipeline components...")

    # Dimensionality Reduction
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )

    # Clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Vectorization for keyword extraction
    vectorizer_model = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=1,
        max_df=0.9,
    )

    logger.info("Initializing BERTopic model...")
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,  # Enable BERTopic's internal logging
    )

    logger.info("Fitting topic model to documents...")
    topics, probs = topic_model.fit_transform(documents, embeddings)

    return topic_model, topics, probs


def save_results(topic_model, topics, documents):
    """
    Saves topic information and sample documents to the output directory.
    """
    logger.info("Saving results to output directory...")

    # 1. Save Topic Info
    topic_info = topic_model.get_topic_info()
    topic_info_csv_path = OUTPUT_DIR / "topic_info.csv"
    topic_info.to_csv(topic_info_csv_path, index=False)
    logger.info(f"Saved topic info to: {topic_info_csv_path}")

    # 2. Save Detailed Topic Breakdown
    detailed_output_path = OUTPUT_DIR / "detailed_topics.txt"
    with open(detailed_output_path, "w", encoding="utf-8") as f:
        f.write("=== DISCOVERED THEMES ===\n\n")

        unique_topics = set(topics)
        for topic_id in sorted(unique_topics):
            if topic_id == -1:
                f.write("\n❌ Outliers / Unclustered Docs:\n")
            else:
                f.write(f"\n⚡ Theme/Topic Cluster {topic_id}:\n")
                keywords = [word for word, score in topic_model.get_topic(topic_id)]
                f.write(f"   Keywords: {', '.join(keywords)}\n")

                # Get documents belonging to this topic
                cluster_doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
                f.write(f"   Count: {len(cluster_doc_indices)}\n")

                # Write first 3 examples
                for idx in cluster_doc_indices[:3]:
                    doc_preview = documents[idx][:200].replace("\n", " ")
                    f.write(f"   - Example: {doc_preview}...\n")

    logger.info(f"Saved detailed topics to: {detailed_output_path}")


@chain(name="shared-themes-discovery")
def main():
    """
    Main execution flow for discovering shared themes in job postings.
    """
    days_back = 14
    try:
        documents = load_documents(days_back=days_back)
        if not documents:
            logger.warning("No documents found for the specified period. Exiting.")
            return

        target_model = EMBED_MODEL
        embeddings = generate_embeddings(documents, model=target_model)
        topic_model, topics, probs = run_bertopic_pipeline(documents, embeddings)

        topic_info = topic_model.get_topic_info()
        print("\n--- Top 10 Topics by Frequency ---")
        print(topic_info[["Topic", "Count", "Name"]].head(10))

        save_results(topic_model, topics, documents)

        # ✅ Now this will automatically use the remote IP if PHOENIX_COLLECTOR_ENDPOINT is set
        trace_url = get_trace_url()

        if trace_url:
            logger.info(f"View complete trace: {trace_url}")
        logger.success("Shared themes discovery completed successfully.")

    except Exception as e:
        logger.error(f"Failed to run shared themes discovery: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
