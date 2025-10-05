import logging
from sklearn.datasets import fetch_20newsgroups
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import OnlineCountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    targets = newsgroups.target[:1000]
    return documents, targets

def example_base_topic_model():
    """Demonstrate basic topic modeling with BERTopic."""
    logging.info("Starting base topic model example...")
    documents, _ = load_sample_data()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    model = BERTopic(embedding_model=embedding_model, calculate_probabilities=True)
    model.umap_model.random_state = 42
    model.hdbscan_model.min_cluster_size = 3
    logging.info("Fitting base topic model...")
    model.fit(documents)
    logging.info(f"Number of topics: {len(set(model.topics_))}")
    return model

def example_zeroshot_topic_model():
    """Demonstrate zeroshot topic modeling with BERTopic."""
    logging.info("Starting zeroshot topic model example...")
    documents, _ = load_sample_data()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    zeroshot_topic_list = ["religion", "cars", "electronics"]
    model = BERTopic(
        embedding_model=embedding_model,
        calculate_probabilities=True,
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=0.3,
    )
    model.umap_model.random_state = 42
    model.hdbscan_model.min_cluster_size = 2
    logging.info("Fitting zeroshot topic model...")
    model.fit(documents)
    logging.info(f"Number of topics: {len(set(model.topics_))}")
    return model

def example_custom_topic_model():
    """Demonstrate custom topic modeling with BERTopic."""
    logging.info("Starting custom topic model example...")
    documents, _ = load_sample_data()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=15, n_components=6, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        calculate_probabilities=True,
    )
    logging.info("Fitting custom topic model...")
    model.fit(documents)
    logging.info(f"Number of topics: {len(set(model.topics_))}")
    return model

def example_representation_topic_model():
    """Demonstrate topic modeling with custom representation models."""
    logging.info("Starting representation topic model example...")
    documents, _ = load_sample_data()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=15, n_components=6, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    representation_model = {
        "Main": KeyBERTInspired(),
        "MMR": [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance()],
    }
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        representation_model=representation_model,
        calculate_probabilities=True,
    )
    logging.info("Fitting representation topic model...")
    model.fit(documents)
    logging.info(f"Number of topics: {len(set(model.topics_))}")
    return model

def example_reduced_topic_model():
    """Demonstrate reduced topic modeling with BERTopic."""
    logging.info("Starting reduced topic model example...")
    documents, _ = load_sample_data()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=15, n_components=6, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        calculate_probabilities=True,
    )
    logging.info("Fitting initial topic model...")
    model.fit(documents)
    logging.info("Reducing topics...")
    model.reduce_topics(documents, nr_topics="auto")
    logging.info(f"Number of topics after reduction: {len(set(model.topics_))}")
    return model

def example_merged_topic_model():
    """Demonstrate merged topic modeling with BERTopic."""
    logging.info("Starting merged topic model example...")
    documents, _ = load_sample_data()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=15, n_components=6, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    model = BERTopic(
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
    logging.info(f"Number of topics after merging: {len(set(model.topics_))}")
    return model

def example_kmeans_pca_topic_model():
    """Demonstrate topic modeling with KMeans and PCA."""
    logging.info("Starting KMeans PCA topic model example...")
    documents, _ = load_sample_data()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    hdbscan_model = KMeans(n_clusters=15, random_state=42)
    dim_model = PCA(n_components=5)
    model = BERTopic(
        hdbscan_model=hdbscan_model,
        umap_model=dim_model,
        embedding_model=embedding_model,
    )
    logging.info("Fitting KMeans PCA topic model...")
    model.fit(documents)
    logging.info(f"Number of topics: {len(set(model.topics_))}")
    return model

def example_supervised_topic_model():
    """Demonstrate supervised topic modeling with BERTopic."""
    logging.info("Starting supervised topic model example...")
    documents, targets = load_sample_data()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    empty_dimensionality_model = BaseDimensionalityReduction()
    clf = LogisticRegression()
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=empty_dimensionality_model,
        hdbscan_model=clf,
    )
    logging.info("Fitting supervised topic model...")
    model.fit(documents, y=targets)
    logging.info(f"Number of topics: {len(set(model.topics_))}")
    return model

def example_online_topic_model():
    """Demonstrate online topic modeling with BERTopic."""
    logging.info("Starting online topic model example...")
    documents, _ = load_sample_data()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = PCA(n_components=5)
    cluster_model = MiniBatchKMeans(n_clusters=50, random_state=0)
    vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=0.01)
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        embedding_model=embedding_model,
    )
    logging.info("Fitting online topic model with partial fit...")
    embeddings = embedding_model.encode(documents)
    for index in range(0, len(documents), 50):
        model.partial_fit(documents[index : index + 50], embeddings[index : index + 50])
    logging.info(f"Number of topics: {len(set(model.topics_))}")
    return model

def example_cuml_base_topic_model():
    """Demonstrate topic modeling with cuML-based BERTopic."""
    logging.info("Starting cuML base topic model example...")
    try:
        from cuml.cluster import HDBSCAN as cuml_hdbscan
        from cuml.manifold import UMAP as cuml_umap
    except ImportError:
        logging.warning("cuML not available, skipping example...")
        return None
    documents, _ = load_sample_data()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    model = BERTopic(
        embedding_model=embedding_model,
        calculate_probabilities=True,
        umap_model=cuml_umap(n_components=5, n_neighbors=5, random_state=42),
        hdbscan_model=cuml_hdbscan(min_cluster_size=3, prediction_data=True),
    )
    logging.info("Fitting cuML base topic model...")
    model.fit(documents)
    logging.info(f"Number of topics: {len(set(model.topics_))}")
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_base_topic_model()
    example_zeroshot_topic_model()
    example_custom_topic_model()
    example_representation_topic_model()
    example_reduced_topic_model()
    example_merged_topic_model()
    example_kmeans_pca_topic_model()
    example_supervised_topic_model()
    example_online_topic_model()
    example_cuml_base_topic_model()
    logging.info("Configuration usage examples completed successfully.")