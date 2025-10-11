# jet/libs/bertopic/examples/usage/conf_usage_examples.py
import logging
import os
import shutil
from typing import Callable, Dict, List, Optional, Tuple, Any

from sklearn.datasets import fetch_20newsgroups
from umap import UMAP
from hdbscan import HDBSCAN
from jet.adapters.bertopic import BERTopic
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from bertopic.vectorizers import OnlineCountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression

from jet.adapters.bertopic.utils import get_vectorizer
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

EMBED_MODEL = "embeddinggemma"


def load_sample_data() -> Tuple[List[str], List[int]]:
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    documents = newsgroups.data[:1000]
    targets = newsgroups.target[:1000]
    return documents, targets


def example_base_topic_model() -> BERTopic:
    """Demonstrate basic topic modeling with BERTopic."""
    logging.info("Starting base topic model example...")
    documents, _ = load_sample_data()
    embedding_model = EMBED_MODEL
    model = BERTopic(
        vectorizer_model=get_vectorizer(len(documents)),
        verbose=True,
        embedding_model=embedding_model,
        calculate_probabilities=True
    )
    model.umap_model.random_state = 42
    model.hdbscan_model.min_cluster_size = 3
    logging.info("Fitting base topic model...")
    model.fit(documents)
    return model


def example_zeroshot_topic_model() -> BERTopic:
    """Demonstrate zeroshot topic modeling with BERTopic."""
    logging.info("Starting zeroshot topic model example...")
    documents, _ = load_sample_data()
    embedding_model = EMBED_MODEL
    zeroshot_topic_list = ["religion", "cars", "electronics"]
    model = BERTopic(
        vectorizer_model=get_vectorizer(len(documents)),
        verbose=True,
        embedding_model=embedding_model,
        calculate_probabilities=True,
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=0.3,
    )
    model.umap_model.random_state = 42
    model.hdbscan_model.min_cluster_size = 2
    logging.info("Fitting zeroshot topic model...")
    model.fit(documents)
    return model


def example_custom_topic_model() -> BERTopic:
    """Demonstrate custom topic modeling with BERTopic."""
    logging.info("Starting custom topic model example...")
    documents, _ = load_sample_data()
    embedding_model = EMBED_MODEL
    umap_model = UMAP(n_neighbors=15, n_components=6, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=3, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    model = BERTopic(
        vectorizer_model=get_vectorizer(len(documents)),
        verbose=True,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        calculate_probabilities=True,
    )
    logging.info("Fitting custom topic model...")
    model.fit(documents)
    return model


def example_representation_topic_model() -> BERTopic:
    """Demonstrate topic modeling with custom representation models."""
    logging.info("Starting representation topic model example...")
    documents, _ = load_sample_data()
    embedding_model = EMBED_MODEL
    umap_model = UMAP(n_neighbors=15, n_components=6, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=3, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    representation_model = {
        "Main": KeyBERTInspired(),
        "MMR": [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance()],
    }
    model = BERTopic(
        vectorizer_model=get_vectorizer(len(documents)),
        verbose=True,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        representation_model=representation_model,
        calculate_probabilities=True,
    )
    logging.info("Fitting representation topic model...")
    model.fit(documents)
    return model


def example_reduced_topic_model() -> BERTopic:
    """Demonstrate reduced topic modeling with BERTopic."""
    logging.info("Starting reduced topic model example...")
    documents, _ = load_sample_data()
    embedding_model = EMBED_MODEL
    umap_model = UMAP(n_neighbors=15, n_components=6, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=3, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    model = BERTopic(
        vectorizer_model=get_vectorizer(len(documents)),
        verbose=True,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        calculate_probabilities=True,
    )
    logging.info("Fitting initial topic model...")
    model.fit(documents)
    logging.info("Reducing topics...")
    model.reduce_topics(documents, nr_topics="auto")
    return model


def example_merged_topic_model() -> BERTopic:
    """Demonstrate merged topic modeling with BERTopic."""
    logging.info("Starting merged topic model example...")
    documents, _ = load_sample_data()
    embedding_model = EMBED_MODEL
    umap_model = UMAP(n_neighbors=15, n_components=6, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=3, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    model = BERTopic(
        vectorizer_model=get_vectorizer(len(documents)),
        verbose=True,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        calculate_probabilities=True,
    )
    logging.info("Fitting initial topic model...")
    model.fit(documents)
    logging.info("Merging topics...")
    topics_to_merge = [[1, 2], [3, 4]]
    model.merge_topics(documents, topics_to_merge)
    topics_to_merge = [[5, 6, 7]]
    model.merge_topics(documents, topics_to_merge)
    return model


def example_kmeans_pca_topic_model() -> BERTopic:
    """Demonstrate topic modeling with KMeans and PCA."""
    logging.info("Starting KMeans PCA topic model example...")
    documents, _ = load_sample_data()
    embedding_model = EMBED_MODEL
    hdbscan_model = KMeans(n_clusters=15, random_state=42)
    dim_model = PCA(n_components=5)
    model = BERTopic(
        vectorizer_model=get_vectorizer(len(documents)),
        verbose=True,
        hdbscan_model=hdbscan_model,
        umap_model=dim_model,
        embedding_model=embedding_model,
    )
    logging.info("Fitting KMeans PCA topic model...")
    model.fit(documents)
    return model


def example_supervised_topic_model() -> BERTopic:
    """Demonstrate supervised topic modeling with BERTopic."""
    logging.info("Starting supervised topic model example...")
    documents, targets = load_sample_data()
    embedding_model = EMBED_MODEL
    empty_dimensionality_model = BaseDimensionalityReduction()
    clf = LogisticRegression()
    model = BERTopic(
        vectorizer_model=get_vectorizer(len(documents)),
        verbose=True,
        embedding_model=embedding_model,
        umap_model=empty_dimensionality_model,
        hdbscan_model=clf
    )
    logging.info("Fitting supervised topic model...")
    model.fit(documents, y=targets)
    return model


def example_online_topic_model() -> BERTopic:
    """Demonstrate online topic modeling with BERTopic."""
    logging.info("Starting online topic model example...")
    documents, _ = load_sample_data()
    embedding_model = EMBED_MODEL
    umap_model = PCA(n_components=5)
    cluster_model = MiniBatchKMeans(n_clusters=50, random_state=0)
    vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=0.01)
    model = BERTopic(
        verbose=True,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        embedding_model=embedding_model,
    )
    logging.info("Fitting online topic model with partial fit...")
    embeddings = embedding_model.encode(documents)
    for index in range(0, len(documents), 50):
        model.partial_fit(documents[index : index + 50], embeddings[index : index + 50])
    return model


def example_cuml_base_topic_model() -> Optional[BERTopic]:
    """Demonstrate topic modeling with cuML-based BERTopic."""
    logging.info("Starting cuML base topic model example...")
    try:
        from cuml.cluster import HDBSCAN as cuml_hdbscan
        from cuml.manifold import UMAP as cuml_umap
    except ImportError:
        logging.warning("cuML not available, skipping example...")
        return None
    documents, _ = load_sample_data()
    embedding_model = EMBED_MODEL
    model = BERTopic(
        vectorizer_model=get_vectorizer(len(documents)),
        verbose=True,
        embedding_model=embedding_model,
        calculate_probabilities=True,
        umap_model=cuml_umap(n_components=5, n_neighbors=5, random_state=42),
        hdbscan_model=cuml_hdbscan(min_cluster_size=3, prediction_data=True),
    )
    logging.info("Fitting cuML base topic model...")
    model.fit(documents)
    return model


def extract_topic_data(model: BERTopic, documents: List[str], topics: Optional[List[int]] = None, probs: Optional[List[List[float]]] = None, top_n_words: int = 10) -> Dict[str, Any]:
    """
    Convert BERTopic model outputs into a JSON-serializable summary.
    """
    topic_info = model.get_topic_info()
    topics_data: Dict[int, Dict[str, Any]] = {}

    for topic_id in topic_info.Topic:
        if topic_id == -1:
            continue  # skip outliers
        words = model.get_topic(int(topic_id))
        topics_data[int(topic_id)] = {
            "name": topic_info.loc[topic_info.Topic == topic_id, "Name"].values[0],
            "count": int(topic_info.loc[topic_info.Topic == topic_id, "Count"].values[0]),
            "words": [{"word": w, "weight": float(wt)} for w, wt in words[:top_n_words]],
        }

    # attach document-topic assignments and probabilities
    doc_data = []
    if topics is not None and probs is not None:
        for doc, topic, prob in zip(documents, topics, probs):
            doc_data.append({"text": doc[:300], "topic": int(topic), "probabilities": [float(p) for p in prob] if prob is not None else None})

    return {"summary": topics_data, "documents": doc_data}


def process_and_save_model(model_fn: Callable[[], Any], name: str, output_dir: Optional[str] = None) -> None:
    """
    Run an example function that returns either:
      - an unfitted model (so we call fit_transform), or
      - an already-fitted model (so we call transform)
    Extract readable topic data and save both the model + JSON summary.
    """
    logging.info(f"=== Running {name} ===")
    documents, _ = load_sample_data()

    model = model_fn()

    # Safe choice: if model already has topics_ set, use transform() to avoid double-fit issues.
    if hasattr(model, "topics_") and getattr(model, "topics_") is not None:
        logging.info("Model already fitted — using transform() instead of fit_transform()")
        topics, probs = model.transform(documents)
    else:
        topics, probs = model.fit_transform(documents)

    # Extract readable topic info
    topic_data = extract_topic_data(model, documents, topics, probs)

    # Paths
    base_dir = output_dir or os.path.join(os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    model_dir = os.path.join(base_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{name}.bertopic")
    json_path = os.path.join(model_dir, f"{name}_results.json")

    # Save both model + summary JSON
    model.save(model_path)
    save_file(topic_data, json_path)

    logging.info(f"Saved model to {model_path}")
    logging.info(f"Saved readable topic summary to {json_path}")
    logging.info(f"=== Completed {name} ===\n")


if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    # --- Run all examples ---
    process_and_save_model(example_base_topic_model, "base_topic", output_dir=OUTPUT_DIR)
    process_and_save_model(example_zeroshot_topic_model, "zeroshot_topic", output_dir=OUTPUT_DIR)
    process_and_save_model(example_custom_topic_model, "custom_topic", output_dir=OUTPUT_DIR)
    process_and_save_model(example_representation_topic_model, "representation_topic", output_dir=OUTPUT_DIR)
    process_and_save_model(example_reduced_topic_model, "reduced_topic", output_dir=OUTPUT_DIR)
    process_and_save_model(example_merged_topic_model, "merged_topic", output_dir=OUTPUT_DIR)
    process_and_save_model(example_kmeans_pca_topic_model, "kmeans_pca_topic", output_dir=OUTPUT_DIR)
    process_and_save_model(example_supervised_topic_model, "supervised_topic", output_dir=OUTPUT_DIR)
    process_and_save_model(example_online_topic_model, "online_topic", output_dir=OUTPUT_DIR)

    cuml_model = example_cuml_base_topic_model()
    if cuml_model is not None:
        process_and_save_model(example_cuml_base_topic_model, "cuml_base_topic", output_dir=OUTPUT_DIR)

    logging.info("✅ All BERTopic configuration examples completed successfully.")
