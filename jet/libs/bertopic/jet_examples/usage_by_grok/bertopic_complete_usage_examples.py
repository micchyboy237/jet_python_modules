from typing import List, Tuple, Union, Mapping, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from bertopic._bertopic import TopicMapper
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic.representation import BaseRepresentation
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.graph_objects as go

from jet.adapters.bertopic import BERTopic
from jet.file.utils import save_file
from jet.logger import logger
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

# Sample data for demonstration
SAMPLE_DOCS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming natural language processing",
    "BERTopic is a powerful topic modeling tool",
    "Natural language models are improving rapidly",
    "The dog sleeps while the fox runs"
]
SAMPLE_TIMESTAMPS = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
SAMPLE_CLASSES = ["positive", "neutral", "positive", "neutral", "negative"]
SAMPLE_IMAGES = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]

def example_init_bertopic() -> BERTopic:
    """
    Demonstrates initialization of BERTopic with all parameters.
    - Configures a multilingual model with custom settings for all arguments.
    - Uses SAMPLE_DOCS indirectly to ensure compatibility.
    """
    vectorizer = CountVectorizer(stop_words="english")
    ctfidf = ClassTfidfTransformer()
    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
    hdbscan = HDBSCAN(min_cluster_size=2, metric="euclidean", cluster_selection_method="eom")
    model = BERTopic(
        language="multilingual",
        top_n_words=5,
        n_gram_range=(1, 2),
        min_topic_size=2,
        nr_topics="auto",
        low_memory=True,
        calculate_probabilities=True,
        seed_topic_list=[["fox", "dog"], ["machine", "learning"]],
        zeroshot_topic_list=["animals", "technology"],
        zeroshot_min_similarity=0.8,
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        umap_model=umap,
        hdbscan_model=hdbscan,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        representation_model=None,
        verbose=True
    )
    logger.success(f"Initialized BERTopic with language: {model.language}, top_n_words: {model.top_n_words}")
    return model

def example_n_gram_range() -> Tuple[int, int]:
    """Demonstrates accessing the n_gram_range property."""
    model = example_init_bertopic()
    logger.success(f"n_gram_range: {model.n_gram_range}")
    return model.n_gram_range

def example_calculate_probabilities() -> bool:
    """Demonstrates accessing the calculate_probabilities property."""
    model = example_init_bertopic()
    logger.success(f"calculate_probabilities: {model.calculate_probabilities}")
    return model.calculate_probabilities

def example_verbose() -> bool:
    """Demonstrates accessing the verbose property."""
    model = example_init_bertopic()
    logger.success(f"verbose: {model.verbose}")
    return model.verbose

def example_seed_topic_list() -> List[List[str]]:
    """Demonstrates accessing the seed_topic_list property."""
    model = example_init_bertopic()
    logger.success(f"seed_topic_list: {model.seed_topic_list}")
    return model.seed_topic_list

def example_zeroshot_topic_list() -> List[str]:
    """Demonstrates accessing the zeroshot_topic_list property."""
    model = example_init_bertopic()
    logger.success(f"zeroshot_topic_list: {model.zeroshot_topic_list}")
    return model.zeroshot_topic_list

def example_zeroshot_min_similarity() -> float:
    """Demonstrates accessing the zeroshot_min_similarity property."""
    model = example_init_bertopic()
    logger.success(f"zeroshot_min_similarity: {model.zeroshot_min_similarity}")
    return model.zeroshot_min_similarity

def example_language() -> str:
    """Demonstrates accessing the language property."""
    model = example_init_bertopic()
    logger.success(f"language: {model.language}")
    return model.language

def example_embedding_model() -> Any:
    """Demonstrates accessing the embedding_model property."""
    model = example_init_bertopic()
    logger.success(f"embedding_model: {model.embedding_model}")
    return model.embedding_model

def example_vectorizer_model() -> CountVectorizer:
    """Demonstrates accessing the vectorizer_model property."""
    model = example_init_bertopic()
    logger.success(f"vectorizer_model: {model.vectorizer_model}")
    return model.vectorizer_model

def example_ctfidf_model() -> ClassTfidfTransformer:
    """Demonstrates accessing the ctfidf_model property."""
    model = example_init_bertopic()
    logger.success(f"ctfidf_model: {model.ctfidf_model}")
    return model.ctfidf_model

def example_tfidfvectorizer() -> pd.DataFrame:
    """
    Demonstrates using TfidfVectorizer as the vectorizer_model in BERTopic.
    - Configures BERTopic with TfidfVectorizer to generate the term-document matrix.
    - Uses SAMPLE_DOCS to fit the model and display topic information.
    """
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=1,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False
    )
    model = example_init_bertopic()
    model.vectorizer_model = tfidf_vectorizer  # Override default CountVectorizer with TfidfVectorizer
    model.fit(documents=SAMPLE_DOCS)
    topic_info = model.get_topic_info()
    logger.success(f"Topic information with TfidfVectorizer:\n{topic_info}")
    return topic_info

def example_representation_model() -> BaseRepresentation:
    """Demonstrates accessing the representation_model property."""
    model = example_init_bertopic()
    logger.success(f"representation_model: {model.representation_model}")
    return model.representation_model

def example_umap_model() -> UMAP:
    """Demonstrates accessing the umap_model property."""
    model = example_init_bertopic()
    logger.success(f"umap_model: {model.umap_model}")
    return model.umap_model

def example_hdbscan_model() -> HDBSCAN:
    """Demonstrates accessing the hdbscan_model property."""
    model = example_init_bertopic()
    logger.success(f"hdbscan_model: {model.hdbscan_model}")
    return model.hdbscan_model

def example_top_n_words() -> int:
    """Demonstrates accessing the top_n_words property."""
    model = example_init_bertopic()
    logger.success(f"top_n_words: {model.top_n_words}")
    return model.top_n_words

def example_min_topic_size() -> int:
    """Demonstrates accessing the min_topic_size property."""
    model = example_init_bertopic()
    logger.success(f"min_topic_size: {model.min_topic_size}")
    return model.min_topic_size

def example_nr_topics() -> Union[int, str]:
    """Demonstrates accessing the nr_topics property."""
    model = example_init_bertopic()
    logger.success(f"nr_topics: {model.nr_topics}")
    return model.nr_topics

def example_low_memory() -> bool:
    """Demonstrates accessing the low_memory property."""
    model = example_init_bertopic()
    logger.success(f"low_memory: {model.low_memory}")
    return model.low_memory

def example_topics_() -> List[int]:
    """Demonstrates accessing the topics_ property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success(f"topics_: {model.topics_}")
    return model.topics_

def example_probabilities_() -> np.ndarray:
    """Demonstrates accessing the probabilities_ property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success(f"probabilities_ shape: {model.probabilities_.shape if model.probabilities_ is not None else None}")
    return model.probabilities_

def example_topic_sizes_() -> Mapping[int, int]:
    """Demonstrates accessing the topic_sizes_ property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success(f"topic_sizes_: {model.topic_sizes_}")
    return model.topic_sizes_

def example_topic_mapper_() -> Any:
    """Demonstrates accessing the topic_mapper_ property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success(f"topic_mapper_: {model.topic_mapper_}")
    return model.topic_mapper_

def example_topic_representations_() -> Mapping[str, List[Tuple[str, float]]]:
    """Demonstrates accessing the topic_representations_ property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success(f"topic_representations_: {model.topic_representations_}")
    return model.topic_representations_

def example_topic_embeddings_() -> np.ndarray:
    """Demonstrates accessing the topic_embeddings_ property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success(f"topic_embeddings_ shape: {model.topic_embeddings_.shape if model.topic_embeddings_ is not None else None}")
    return model.topic_embeddings_

def example__topic_id_to_zeroshot_topic_idx() -> Mapping[int, int]:
    """Demonstrates accessing the _topic_id_to_zeroshot_topic_idx property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success(f"_topic_id_to_zeroshot_topic_idx: {model._topic_id_to_zeroshot_topic_idx}")
    return model._topic_id_to_zeroshot_topic_idx

def example_custom_labels_() -> List[str]:
    """Demonstrates accessing the custom_labels_ property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    model.set_topic_labels(["animal_topic", "tech_topic"])
    logger.success(f"custom_labels_: {model.custom_labels_}")
    return model.custom_labels_

def example_c_tf_idf_() -> csr_matrix:
    """Demonstrates accessing the c_tf_idf_ property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success(f"c_tf_idf_ shape: {model.c_tf_idf_.shape}")
    return model.c_tf_idf_

def example_representative_images_() -> Any:
    """Demonstrates accessing the representative_images_ property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS, images=SAMPLE_IMAGES)
    logger.success(f"representative_images_: {model.representative_images_}")
    return model.representative_images_

def example_representative_docs_() -> Mapping[int, List[str]]:
    """Demonstrates accessing the representative_docs_ property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success(f"representative_docs_: {model.representative_docs_}")
    return model.representative_docs_

def example_topic_aspects_() -> Mapping[str, Any]:
    """Demonstrates accessing the topic_aspects_ property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success(f"topic_aspects_: {model.topic_aspects_}")
    return model.topic_aspects_

def example__merged_topics() -> Any:
    """Demonstrates accessing the _merged_topics property."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    model.merge_topics(docs=SAMPLE_DOCS, topics_to_merge=[[0, 1]])
    logger.success(f"_merged_topics: {model._merged_topics}")
    return model._merged_topics

def example__outliers():
    """Demonstrates the _outliers method (internal, not directly accessible)."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success("Internal method _outliers not directly accessible.")

def example_topic_labels_():
    """Demonstrates the topic_labels_ method (internal, not directly accessible)."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    logger.success("Internal method topic_labels_ not directly accessible.")

def example_fit() -> None:
    """Demonstrates the fit method with all arguments."""
    model = example_init_bertopic()
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    model.fit(documents=SAMPLE_DOCS, embeddings=embeddings, images=SAMPLE_IMAGES, y=[0, 1, 0, 1, 0])
    logger.success(f"Model fitted with topics: {model.topics_}")

def example_fit_transform() -> Tuple[List[int], np.ndarray]:
    """Demonstrates the fit_transform method with all arguments."""
    model = example_init_bertopic()
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    topics, probabilities = model.fit_transform(documents=SAMPLE_DOCS, embeddings=embeddings, images=SAMPLE_IMAGES, y=[0, 1, 0, 1, 0])
    logger.success(f"fit_transform topics: {topics}, probabilities shape: {probabilities.shape if probabilities is not None else None}")
    return topics, probabilities

def example_transform() -> Tuple[List[int], np.ndarray]:
    """Demonstrates the transform method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS[:3])
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS[3:])
    topics, probabilities = model.transform(documents=SAMPLE_DOCS[3:], embeddings=embeddings, images=SAMPLE_IMAGES[3:])
    logger.success(f"transform topics: {topics}, probabilities shape: {probabilities.shape if probabilities is not None else None}")
    return topics, probabilities

def example_partial_fit() -> None:
    """Demonstrates the partial_fit method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS[:3])
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS[3:])
    model.partial_fit(documents=SAMPLE_DOCS[3:], embeddings=embeddings, y=[1, 0])
    logger.success(f"partial_fit updated topics: {model.topics_}")

def example_topics_over_time() -> pd.DataFrame:
    """Demonstrates the topics_over_time method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topics_over_time = model.topics_over_time(
        docs=SAMPLE_DOCS,
        timestamps=SAMPLE_TIMESTAMPS,
        topics=model.topics_,
        nr_bins=3,
        datetime_format="%Y-%m-%d",
        evolution_tuning=True,
        global_tuning=True
    )
    logger.success(f"topics_over_time:\n{topics_over_time}")
    return topics_over_time

def example_topics_per_class() -> pd.DataFrame:
    """Demonstrates the topics_per_class method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topics_per_class = model.topics_per_class(docs=SAMPLE_DOCS, classes=SAMPLE_CLASSES, global_tuning=True)
    logger.success(f"topics_per_class:\n{topics_per_class}")
    return topics_per_class

def example_hierarchical_topics() -> pd.DataFrame:
    """Demonstrates the hierarchical_topics method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    from scipy.cluster.hierarchy import linkage, euclidean_distances
    hier_topics = model.hierarchical_topics(
        docs=SAMPLE_DOCS,
        use_ctfidf=True,
        linkage_function=lambda x: linkage(x, method="ward"),
        distance_function=euclidean_distances
    )
    logger.success(f"hierarchical_topics:\n{hier_topics}")
    return hier_topics

def example_approximate_distribution() -> Tuple[np.ndarray, Union[List[np.ndarray], None]]:
    """Demonstrates the approximate_distribution method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    distribution, token_dist = model.approximate_distribution(
        documents=SAMPLE_DOCS,
        window=4,
        stride=1,
        min_similarity=0.1,
        batch_size=1000,
        padding=False,
        use_embedding_model=False,
        calculate_tokens=True,
        separator=" "
    )
    logger.success(f"approximate_distribution shape: {distribution.shape}, token_dist: {token_dist is not None}")
    return distribution, token_dist

def example_find_topics() -> Tuple[List[int], List[float]]:
    """Demonstrates the find_topics method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topics, similarities = model.find_topics(search_term="machine learning", image=SAMPLE_IMAGES[0], top_n=3)
    logger.success(f"find_topics for 'machine learning': {topics}, similarities: {similarities}")
    return topics, similarities

def example_update_topics() -> None:
    """Demonstrates the update_topics method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    new_vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words="english")
    model.update_topics(
        docs=SAMPLE_DOCS,
        images=SAMPLE_IMAGES,
        topics=model.topics_,
        top_n_words=3,
        n_gram_range=(1, 3),
        vectorizer_model=new_vectorizer,
        ctfidf_model=ClassTfidfTransformer(),
        representation_model=None
    )
    logger.success(f"Updated topic info:\n{model.get_topic_info()}")

def example_get_topics() -> Mapping[str, List[Tuple[str, float]]]:
    """Demonstrates the get_topics method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topics = model.get_topics(full=True)
    logger.success(f"get_topics (full=True):\n{topics}")
    return topics

def example_get_topic() -> Union[Mapping[str, Tuple[str, float]], bool]:
    """Demonstrates the get_topic method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topic_info = model.get_topic(topic=0, full=True)
    logger.success(f"get_topic for topic 0 (full=True): {topic_info}")
    return topic_info

def example_get_topic_info() -> pd.DataFrame:
    """Demonstrates the get_topic_info method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topic_info = model.get_topic_info(topic=0)
    logger.success(f"get_topic_info for topic 0:\n{topic_info}")
    return topic_info

def example_get_topic_freq() -> Union[pd.DataFrame, int]:
    """Demonstrates the get_topic_freq method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    freq = model.get_topic_freq(topic=0)
    logger.success(f"get_topic_freq for topic 0: {freq}")
    return freq

def example_get_document_info() -> pd.DataFrame:
    """Demonstrates the get_document_info method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    metadata = {"source": ["doc1", "doc2", "doc3", "doc4", "doc5"]}
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    doc_info = model.get_document_info(docs=SAMPLE_DOCS, df=df, metadata=metadata)
    logger.success(f"get_document_info:\n{doc_info}")
    return doc_info

def example_get_representative_docs() -> List[str]:
    """Demonstrates the get_representative_docs method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    docs = model.get_representative_docs(topic=0)
    logger.success(f"get_representative_docs for topic 0: {docs}")
    return docs

def example_get_topic_tree() -> str:
    """Demonstrates the get_topic_tree static method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    hier_topics = model.hierarchical_topics(docs=SAMPLE_DOCS)
    tree = BERTopic.get_topic_tree(hier_topics=hier_topics, max_distance=1.0, tight_layout=True)
    logger.success(f"get_topic_tree:\n{tree}")
    return tree

def example_set_topic_labels() -> None:
    """Demonstrates the set_topic_labels method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    model.set_topic_labels(topic_labels=["animal_topic", "tech_topic"])
    logger.success(f"set_topic_labels: {model.custom_labels_}")

def example_generate_topic_labels() -> List[str]:
    """Demonstrates the generate_topic_labels method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    labels = model.generate_topic_labels(nr_words=2, topic_prefix=False, word_length=10, separator="-", aspect=None)
    logger.success(f"generate_topic_labels: {labels}")
    return labels

def example_merge_topics() -> None:
    """Demonstrates the merge_topics method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    model.merge_topics(docs=SAMPLE_DOCS, topics_to_merge=[[0, 1]], images=SAMPLE_IMAGES)
    logger.success(f"merge_topics updated topics: {model.topics_}")

def example_delete_topics() -> None:
    """Demonstrates the delete_topics method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    model.delete_topics(topics_to_delete=[0])
    logger.success(f"delete_topics updated topics: {model.topics_}")

def example_reduce_topics() -> None:
    """Demonstrates the reduce_topics method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    model.reduce_topics(docs=SAMPLE_DOCS, nr_topics=2, images=SAMPLE_IMAGES, use_ctfidf=True)
    logger.success(f"reduce_topics updated topic info:\n{model.get_topic_info()}")

def example_reduce_outliers() -> List[int]:
    """Demonstrates the reduce_outliers method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    topics = model.reduce_outliers(
        documents=SAMPLE_DOCS,
        topics=model.topics_,
        images=SAMPLE_IMAGES,
        strategy="distributions",
        probabilities=model.probabilities_,
        threshold=0.1,
        embeddings=embeddings,
        distributions_params={"min_similarity": 0.2}
    )
    logger.success(f"reduce_outliers topics: {topics}")
    return topics

def example_visualize_topics() -> go.Figure:
    """Demonstrates the visualize_topics method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_topics(
        topics=[0, 1],
        top_n_topics=2,
        use_ctfidf=True,
        custom_labels=True,
        title="Custom Topic Distance Map",
        width=700,
        height=700
    )
    logger.success("Visualizing topics (figure displayed)")
    fig.show()
    return fig

def example_visualize_documents() -> go.Figure:
    """Demonstrates the visualize_documents method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    reduced_embeddings = model._reduce_dimensionality(embeddings=embeddings)
    fig = model.visualize_documents(
        docs=SAMPLE_DOCS,
        topics=[0, 1],
        embeddings=embeddings,
        reduced_embeddings=reduced_embeddings,
        sample=0.5,
        hide_annotations=True,
        hide_document_hover=True,
        custom_labels=True,
        title="Custom Document Visualization",
        width=1000,
        height=800
    )
    logger.success("Visualizing documents (figure displayed)")
    fig.show()
    return fig

def example_visualize_document_datamap() -> go.Figure:
    """Demonstrates the visualize_document_datamap method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    reduced_embeddings = model._reduce_dimensionality(embeddings=embeddings)
    fig = model.visualize_document_datamap(
        docs=SAMPLE_DOCS,
        topics=[0, 1],
        embeddings=embeddings,
        reduced_embeddings=reduced_embeddings,
        custom_labels=True,
        title="Custom Document Datamap",
        sub_title="Sample Data",
        width=1000,
        height=800,
        interactive=True,
        enable_search=True,
        topic_prefix=False,
        datamap_kwds={"n_neighbors": 10},
        int_datamap_kwds={"show_topics": True}
    )
    logger.success("Visualizing document datamap (figure displayed)")
    fig.show()
    return fig

def example_visualize_hierarchical_documents() -> go.Figure:
    """Demonstrates the visualize_hierarchical_documents method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    hier_topics = model.hierarchical_topics(docs=SAMPLE_DOCS)
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    reduced_embeddings = model._reduce_dimensionality(embeddings=embeddings)
    fig = model.visualize_hierarchical_documents(
        docs=SAMPLE_DOCS,
        hierarchical_topics=hier_topics,
        topics=[0, 1],
        embeddings=embeddings,
        reduced_embeddings=reduced_embeddings,
        sample=0.5,
        hide_annotations=True,
        hide_document_hover=True,
        nr_levels=5,
        level_scale="linear",
        custom_labels=True,
        title="Custom Hierarchical Documents",
        width=1000,
        height=800
    )
    logger.success("Visualizing hierarchical documents (figure displayed)")
    fig.show()
    return fig

def example_visualize_term_rank() -> go.Figure:
    """Demonstrates the visualize_term_rank method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_term_rank(
        topics=[0, 1],
        log_scale=True,
        custom_labels=True,
        title="Custom Term Rank",
        width=600,
        height=400
    )
    logger.success("Visualizing term rank (figure displayed)")
    fig.show()
    return fig

def example_visualize_topics_over_time() -> go.Figure:
    """Demonstrates the visualize_topics_over_time method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topics_over_time = model.topics_over_time(docs=SAMPLE_DOCS, timestamps=SAMPLE_TIMESTAMPS)
    fig = model.visualize_topics_over_time(
        topics_over_time=topics_over_time,
        top_n_topics=2,
        topics=[0, 1],
        normalize_frequency=True,
        custom_labels=True,
        title="Custom Topics Over Time",
        width=1000,
        height=500
    )
    logger.success("Visualizing topics over time (figure displayed)")
    fig.show()
    return fig

def example_visualize_topics_per_class() -> go.Figure:
    """Demonstrates the visualize_topics_per_class method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topics_per_class = model.topics_per_class(docs=SAMPLE_DOCS, classes=SAMPLE_CLASSES)
    fig = model.visualize_topics_per_class(
        topics_per_class=topics_per_class,
        top_n_topics=2,
        topics=[0, 1],
        normalize_frequency=True,
        custom_labels=True,
        title="Custom Topics Per Class",
        width=1000,
        height=800
    )
    logger.success("Visualizing topics per class (figure displayed)")
    fig.show()
    return fig

def example_visualize_distribution() -> go.Figure:
    """Demonstrates the visualize_distribution method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_distribution(
        probabilities=model.probabilities_,
        min_probability=0.01,
        custom_labels=True,
        title="Custom Topic Probability Distribution",
        width=700,
        height=500
    )
    logger.success("Visualizing distribution (figure displayed)")
    fig.show()
    return fig

def example_visualize_approximate_distribution() -> None:
    """Demonstrates the visualize_approximate_distribution method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    distribution, _ = model.approximate_distribution(documents=[SAMPLE_DOCS[0]])
    model.visualize_approximate_distribution(
        document=SAMPLE_DOCS[0],
        topic_token_distribution=distribution,
        normalize=True
    )
    logger.success("Visualizing approximate distribution (figure displayed)")

def example_visualize_hierarchy() -> go.Figure:
    """Demonstrates the visualize_hierarchy method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    hier_topics = model.hierarchical_topics(docs=SAMPLE_DOCS)
    from scipy.cluster.hierarchy import linkage, euclidean_distances
    fig = model.visualize_hierarchy(
        orientation="left",
        topics=[0, 1],
        top_n_topics=2,
        use_ctfidf=True,
        custom_labels=True,
        title="Custom Hierarchical Clustering",
        width=900,
        height=500,
        hierarchical_topics=hier_topics,
        linkage_function=lambda x: linkage(x, method="ward"),
        distance_function=euclidean_distances,
        color_threshold=1
    )
    logger.success("Visualizing hierarchy (figure displayed)")
    fig.show()
    return fig

def example_visualize_heatmap() -> go.Figure:
    """Demonstrates the visualize_heatmap method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_heatmap(
        topics=[0, 1],
        top_n_topics=2,
        n_clusters=2,
        use_ctfidf=True,
        custom_labels=True,
        title="Custom Similarity Matrix",
        width=700,
        height=700
    )
    logger.success("Visualizing heatmap (figure displayed)")
    fig.show()
    return fig

def example_visualize_barchart() -> go.Figure:
    """Demonstrates the visualize_barchart method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_barchart(
        topics=[0, 1],
        top_n_topics=2,
        n_words=3,
        custom_labels=True,
        title="Custom Topic Word Scores",
        width=300,
        height=300,
        autoscale=True
    )
    logger.success("Visualizing barchart (figure displayed)")
    fig.show()
    return fig

def example_save() -> None:
    """Demonstrates the save method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    model.save(
        path="bertopic_model.pkl",
        serialization="pickle",
        save_embedding_model=True,
        save_ctfidf=True
    )
    logger.success("Model saved to bertopic_model.pkl")

def example_load() -> BERTopic:
    """Demonstrates the load class method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    model.save(path="bertopic_model.pkl", serialization="pickle")
    loaded_model = BERTopic.load(path="bertopic_model.pkl", embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
    logger.success(f"Loaded model topics: {loaded_model.topics_}")
    return loaded_model

def example_merge_models() -> BERTopic:
    """Demonstrates the merge_models class method with all arguments."""
    model1 = example_init_bertopic()
    model1.fit(documents=SAMPLE_DOCS[:3])
    model2 = example_init_bertopic()
    model2.fit(documents=SAMPLE_DOCS[3:])
    merged_model = BERTopic.merge_models(
        models=[model1, model2],
        min_similarity=0.7,
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
    )
    logger.success(f"Merged model topics: {merged_model.topics_}")
    return merged_model

def example_push_to_hf_hub() -> None:
    """Demonstrates the push_to_hf_hub method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    model.push_to_hf_hub(
        repo_id="user/bertopic_model",
        commit_message="Test BERTopic model push",
        token=None,  # Requires valid token for actual use
        revision="main",
        private=False,
        create_pr=False,
        model_card=True,
        serialization="safetensors",
        save_embedding_model=True,
        save_ctfidf=True
    )
    logger.success("Model push to Hugging Face Hub (mock, requires token)")

def example_get_params() -> Mapping[str, Any]:
    """Demonstrates the get_params method with all arguments."""
    model = example_init_bertopic()
    params = model.get_params(deep=True)
    logger.success(f"get_params (deep=True): {params}")
    return params

def example__extract_embeddings() -> np.ndarray:
    """Demonstrates the _extract_embeddings method with all arguments."""
    model = example_init_bertopic()
    embeddings = model._extract_embeddings(
        documents=SAMPLE_DOCS,
        images=SAMPLE_IMAGES,
        method="document",
        verbose=True
    )
    logger.success(f"_extract_embeddings shape: {embeddings.shape}")
    return embeddings

def example__images_to_text() -> pd.DataFrame:
    """Demonstrates the _images_to_text method with all arguments."""
    model = example_init_bertopic()
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    result = model._images_to_text(documents=df, embeddings=embeddings)
    logger.success(f"_images_to_text result:\n{result}")
    return result

def example__map_predictions() -> List[int]:
    """Demonstrates the _map_predictions method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    predictions = model.topics_
    mapped = model._map_predictions(predictions=predictions)
    logger.success(f"_map_predictions: {mapped}")
    return mapped

def example__reduce_dimensionality() -> np.ndarray:
    """Demonstrates the _reduce_dimensionality method with all arguments."""
    model = example_init_bertopic()
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    reduced = model._reduce_dimensionality(embeddings=embeddings, y=[0, 1, 0, 1, 0], partial_fit=False)
    logger.success(f"_reduce_dimensionality shape: {reduced.shape}")
    return reduced

def example__cluster_embeddings() -> Tuple[pd.DataFrame, np.ndarray]:
    """Demonstrates the _cluster_embeddings method with all arguments."""
    model = example_init_bertopic()
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    reduced = model._reduce_dimensionality(embeddings=embeddings)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    result, probs = model._cluster_embeddings(
        umap_embeddings=reduced,
        documents=df,
        partial_fit=False,
        y=np.array([0, 1, 0, 1, 0])
    )
    logger.success(f"_cluster_embeddings result:\n{result}")
    return result, probs

def example__zeroshot_topic_modeling() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Demonstrates the _zeroshot_topic_modeling method with all arguments."""
    model = example_init_bertopic()
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    result = model._zeroshot_topic_modeling(documents=df, embeddings=embeddings)
    logger.success(f"_zeroshot_topic_modeling result: {result}")
    return result

def example__is_zeroshot():
    """Demonstrates the _is_zeroshot method (internal, not directly accessible)."""
    model = example_init_bertopic()
    logger.success("Internal method _is_zeroshot not directly accessible.")

def example__combine_zeroshot_topics() -> Tuple[pd.DataFrame, np.ndarray]:
    """Demonstrates the _combine_zeroshot_topics method with all arguments."""
    model = example_init_bertopic()
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    assigned_df = df.copy()
    assigned_embeddings = embeddings.copy()
    result = model._combine_zeroshot_topics(
        documents=df,
        embeddings=embeddings,
        assigned_documents=assigned_df,
        assigned_embeddings=assigned_embeddings
    )
    logger.success(f"_combine_zeroshot_topics result: {result}")
    return result

def example__guided_topic_modeling() -> Tuple[List[int], np.ndarray]:
    """Demonstrates the _guided_topic_modeling method with all arguments."""
    model = example_init_bertopic()
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    topics, probs = model._guided_topic_modeling(embeddings=embeddings)
    logger.success(f"_guided_topic_modeling topics: {topics}")
    return topics, probs

def example__extract_topics():
    """Demonstrates the _extract_topics method with all arguments."""
    model = example_init_bertopic()
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    model._extract_topics(
        documents=df,
        embeddings=embeddings,
        mappings=None,
        verbose=True,
        fine_tune_representation=True
    )
    logger.success("Internal method _extract_topics not directly accessible.")

def example__save_representative_docs():
    """Demonstrates the _save_representative_docs method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    model._save_representative_docs(documents=df)
    logger.success("Internal method _save_representative_docs not directly accessible.")

def example__extract_representative_docs() -> Union[List[str], List[List[int]]]:
    """Demonstrates the _extract_representative_docs method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    topics = model.get_topics()
    result = model._extract_representative_docs(
        c_tf_idf=model.c_tf_idf_,
        documents=df,
        topics=topics,
        nr_samples=500,
        nr_repr_docs=3,
        diversity=0.5
    )
    logger.success(f"_extract_representative_docs: {result}")
    return result

def example__create_topic_vectors():
    """Demonstrates the _create_topic_vectors method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    model._create_topic_vectors(documents=df, embeddings=embeddings, mappings=None)
    logger.success("Internal method _create_topic_vectors not directly accessible.")

def example__c_tf_idf() -> Tuple[csr_matrix, List[str]]:
    """Demonstrates the _c_tf_idf method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS, "Topic": model.topics_})
    c_tf_idf, words = model._c_tf_idf(documents_per_topic=df, fit=True, partial_fit=False)
    logger.success(f"_c_tf_idf shape: {c_tf_idf.shape}, words: {words[:5]}")
    return c_tf_idf, words

def example__update_topic_size():
    """Demonstrates the _update_topic_size method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS, "Topic": model.topics_})
    model._update_topic_size(documents=df)
    logger.success("Internal method _update_topic_size not directly accessible.")

def example__extract_words_per_topic() -> Mapping[str, List[Tuple[str, float]]]:
    """Demonstrates the _extract_words_per_topic method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS, "Topic": model.topics_})
    words = model.vectorizer_model.get_feature_names_out()
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    result = model._extract_words_per_topic(
        words=words,
        documents=df,
        c_tf_idf=model.c_tf_idf_,
        fine_tune_representation=True,
        calculate_aspects=True,
        embeddings=embeddings
    )
    logger.success(f"_extract_words_per_topic: {result}")
    return result

def example__reduce_topics() -> pd.DataFrame:
    """Demonstrates the _reduce_topics method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    result = model._reduce_topics(documents=df, use_ctfidf=True)
    logger.success(f"_reduce_topics result:\n{result}")
    return result

def example__reduce_to_n_topics() -> pd.DataFrame:
    """Demonstrates the _reduce_to_n_topics method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    result = model._reduce_to_n_topics(documents=df, nr_topics=2, use_ctfidf=True)
    logger.success(f"_reduce_to_n_topics result:\n{result}")
    return result

def example__auto_reduce_topics() -> pd.DataFrame:
    """Demonstrates the _auto_reduce_topics method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    result = model._auto_reduce_topics(documents=df, use_ctfidf=True)
    logger.success(f"_auto_reduce_topics result:\n{result}")
    return result

def example__sort_mappings_by_frequency() -> pd.DataFrame:
    """Demonstrates the _sort_mappings_by_frequency method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS, "Topic": model.topics_})
    result = model._sort_mappings_by_frequency(documents=df)
    logger.success(f"_sort_mappings_by_frequency result:\n{result}")
    return result

def example__map_probabilities() -> Union[np.ndarray, None]:
    """Demonstrates the _map_probabilities method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    result = model._map_probabilities(probabilities=model.probabilities_, original_topics=True)
    logger.success(f"_map_probabilities shape: {result.shape if result is not None else None}")
    return result

def example__preprocess_text() -> List[str]:
    """Demonstrates the _preprocess_text method with all arguments."""
    model = example_init_bertopic()
    documents = np.array(SAMPLE_DOCS)
    result = model._preprocess_text(documents=documents)
    logger.success(f"_preprocess_text: {result}")
    return result

def example__top_n_idx_sparse() -> np.ndarray:
    """Demonstrates the _top_n_idx_sparse static method with all arguments."""
    matrix = csr_matrix([[0, 1, 2], [3, 0, 4], [5, 6, 0]])
    indices = BERTopic._top_n_idx_sparse(matrix=matrix, n=2)
    logger.success(f"_top_n_idx_sparse: {indices}")
    return indices

def example__top_n_values_sparse() -> np.ndarray:
    """Demonstrates the _top_n_values_sparse static method with all arguments."""
    matrix = csr_matrix([[0, 1, 2], [3, 0, 4], [5, 6, 0]])
    indices = BERTopic._top_n_idx_sparse(matrix=matrix, n=2)
    values = BERTopic._top_n_values_sparse(matrix=matrix, indices=indices)
    logger.success(f"_top_n_values_sparse: {values}")
    return values

def example__get_param_names() -> List[str]:
    """Demonstrates the _get_param_names class method."""
    param_names = BERTopic._get_param_names()
    logger.success(f"_get_param_names: {param_names}")
    return param_names

def example__str__() -> str:
    """Demonstrates the __str__ method."""
    model = example_init_bertopic()
    model_str = str(model)
    logger.success(f"__str__: {model_str}")
    return model_str

def example_topicmapper_init() -> TopicMapper:
    """Demonstrates the TopicMapper.__init__ method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topic_mapper = TopicMapper(topics=model.topics_)
    logger.success(f"TopicMapper initialized with topics: {topic_mapper.topics}")
    return topic_mapper

def example_topicmapper_get_mappings() -> Mapping[int, int]:
    """Demonstrates the TopicMapper.get_mappings method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topic_mapper = TopicMapper(topics=model.topics_)
    mappings = topic_mapper.get_mappings(original_topics=True)
    logger.success(f"TopicMapper.get_mappings: {mappings}")
    return mappings

def example_topicmapper_add_mappings() -> None:
    """Demonstrates the TopicMapper.add_mappings method with all arguments."""
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topic_mapper = TopicMapper(topics=model.topics_)
    new_mappings = {0: 1}
    topic_mapper.add_mappings(mappings=new_mappings, topic_model=model)
    logger.success(f"TopicMapper.add_mappings updated topics: {model.topics_}")

def example_topicmapper_add_new_topics() -> None:
    """
    Demonstrates the TopicMapper.add_new_topics method with all arguments.
    - Adds new topic assignments to the TopicMapper and updates the BERTopic model.
    - Uses SAMPLE_DOCS to fit the model and generate initial topics, then adds new topics.
    """
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topic_mapper = TopicMapper(topics=model.topics_)
    new_topics = [2, 2, 1, 1, 2]  # Example new topic assignments for SAMPLE_DOCS
    topic_mapper.add_new_topics(topics=new_topics, topic_model=model)
    logger.success(f"TopicMapper.add_new_topics updated topics: {model.topics_}")

if __name__ == "__main__":
    logger.info("\nRunning example_init_bertopic()...")
    results = example_init_bertopic()
    save_file(results, f"{OUTPUT_DIR}/init_bertopic.json")

    logger.info("\nRunning example_n_gram_range()...")
    results = example_n_gram_range()
    save_file(results, f"{OUTPUT_DIR}/n_gram_range.json")

    logger.info("\nRunning example_calculate_probabilities()...")
    results = example_calculate_probabilities()
    save_file(results, f"{OUTPUT_DIR}/calculate_probabilities.json")

    logger.info("\nRunning example_verbose()...")
    results = example_verbose()
    save_file(results, f"{OUTPUT_DIR}/verbose.json")

    logger.info("\nRunning example_seed_topic_list()...")
    results = example_seed_topic_list()
    save_file(results, f"{OUTPUT_DIR}/seed_topic_list.json")

    logger.info("\nRunning example_zeroshot_topic_list()...")
    results = example_zeroshot_topic_list()
    save_file(results, f"{OUTPUT_DIR}/zeroshot_topic_list.json")

    logger.info("\nRunning example_zeroshot_min_similarity()...")
    results = example_zeroshot_min_similarity()
    save_file(results, f"{OUTPUT_DIR}/zeroshot_min_similarity.json")

    logger.info("\nRunning example_language()...")
    results = example_language()
    save_file(results, f"{OUTPUT_DIR}/language.json")

    logger.info("\nRunning example_embedding_model()...")
    results = example_embedding_model()
    save_file(results, f"{OUTPUT_DIR}/embedding_model.json")

    logger.info("\nRunning example_vectorizer_model()...")
    results = example_vectorizer_model()
    save_file(results, f"{OUTPUT_DIR}/vectorizer_model.json")

    logger.info("\nRunning example_ctfidf_model()...")
    results = example_ctfidf_model()
    save_file(results, f"{OUTPUT_DIR}/ctfidf_model.json")

    logger.info("\nRunning example_tfidfvectorizer()...")
    results = example_tfidfvectorizer()
    save_file(results, f"{OUTPUT_DIR}/tfidfvectorizer.json")

    logger.info("\nRunning example_representation_model()...")
    results = example_representation_model()
    save_file(results, f"{OUTPUT_DIR}/representation_model.json")

    logger.info("\nRunning example_umap_model()...")
    results = example_umap_model()
    save_file(results, f"{OUTPUT_DIR}/umap_model.json")

    logger.info("\nRunning example_hdbscan_model()...")
    results = example_hdbscan_model()
    save_file(results, f"{OUTPUT_DIR}/hdbscan_model.json")

    logger.info("\nRunning example_top_n_words()...")
    results = example_top_n_words()
    save_file(results, f"{OUTPUT_DIR}/top_n_words.json")

    logger.info("\nRunning example_min_topic_size()...")
    results = example_min_topic_size()
    save_file(results, f"{OUTPUT_DIR}/min_topic_size.json")

    logger.info("\nRunning example_nr_topics()...")
    results = example_nr_topics()
    save_file(results, f"{OUTPUT_DIR}/nr_topics.json")

    logger.info("\nRunning example_low_memory()...")
    results = example_low_memory()
    save_file(results, f"{OUTPUT_DIR}/low_memory.json")

    logger.info("\nRunning example_topics_()...")
    results = example_topics_()
    save_file(results, f"{OUTPUT_DIR}/topics_.json")

    logger.info("\nRunning example_probabilities_()...")
    results = example_probabilities_()
    save_file(results, f"{OUTPUT_DIR}/probabilities_.json")

    logger.info("\nRunning example_topic_sizes_()...")
    results = example_topic_sizes_()
    save_file(results, f"{OUTPUT_DIR}/topic_sizes_.json")

    logger.info("\nRunning example_topic_mapper_()...")
    results = example_topic_mapper_()
    save_file(results, f"{OUTPUT_DIR}/topic_mapper_.json")

    logger.info("\nRunning example_topic_representations_()...")
    results = example_topic_representations_()
    save_file(results, f"{OUTPUT_DIR}/topic_representations_.json")

    logger.info("\nRunning example_topic_embeddings_()...")
    results = example_topic_embeddings_()
    save_file(results, f"{OUTPUT_DIR}/topic_embeddings_.json")

    logger.info("\nRunning example__topic_id_to_zeroshot_topic_idx()...")
    results = example__topic_id_to_zeroshot_topic_idx()
    save_file(results, f"{OUTPUT_DIR}/_topic_id_to_zeroshot_topic_idx.json")

    logger.info("\nRunning example_custom_labels_()...")
    results = example_custom_labels_()
    save_file(results, f"{OUTPUT_DIR}/custom_labels_.json")

    logger.info("\nRunning example_c_tf_idf_()...")
    results = example_c_tf_idf_()
    save_file(results, f"{OUTPUT_DIR}/c_tf_idf_.json")

    logger.info("\nRunning example_representative_images_()...")
    results = example_representative_images_()
    save_file(results, f"{OUTPUT_DIR}/representative_images_.json")

    logger.info("\nRunning example_representative_docs_()...")
    results = example_representative_docs_()
    save_file(results, f"{OUTPUT_DIR}/representative_docs_.json")

    logger.info("\nRunning example_topic_aspects_()...")
    results = example_topic_aspects_()
    save_file(results, f"{OUTPUT_DIR}/topic_aspects_.json")

    logger.info("\nRunning example__merged_topics()...")
    results = example__merged_topics()
    save_file(results, f"{OUTPUT_DIR}/_merged_topics.json")

    logger.info("\nRunning example__outliers()...")
    results = example__outliers()
    save_file(results, f"{OUTPUT_DIR}/_outliers.json")

    logger.info("\nRunning example_topic_labels_()...")
    results = example_topic_labels_()
    save_file(results, f"{OUTPUT_DIR}/topic_labels_.json")

    logger.info("\nRunning example_fit()...")
    results = example_fit()
    save_file(results, f"{OUTPUT_DIR}/fit.json")

    logger.info("\nRunning example_fit_transform()...")
    results = example_fit_transform()
    save_file(results, f"{OUTPUT_DIR}/fit_transform.json")

    logger.info("\nRunning example_transform()...")
    results = example_transform()
    save_file(results, f"{OUTPUT_DIR}/transform.json")

    logger.info("\nRunning example_partial_fit()...")
    results = example_partial_fit()
    save_file(results, f"{OUTPUT_DIR}/partial_fit.json")

    logger.info("\nRunning example_topics_over_time()...")
    results = example_topics_over_time()
    save_file(results, f"{OUTPUT_DIR}/topics_over_time.json")

    logger.info("\nRunning example_topics_per_class()...")
    results = example_topics_per_class()
    save_file(results, f"{OUTPUT_DIR}/topics_per_class.json")

    logger.info("\nRunning example_hierarchical_topics()...")
    results = example_hierarchical_topics()
    save_file(results, f"{OUTPUT_DIR}/hierarchical_topics.json")

    logger.info("\nRunning example_approximate_distribution()...")
    results = example_approximate_distribution()
    save_file(results, f"{OUTPUT_DIR}/approximate_distribution.json")

    logger.info("\nRunning example_find_topics()...")
    results = example_find_topics()
    save_file(results, f"{OUTPUT_DIR}/find_topics.json")

    logger.info("\nRunning example_update_topics()...")
    results = example_update_topics()
    save_file(results, f"{OUTPUT_DIR}/update_topics.json")

    logger.info("\nRunning example_get_topics()...")
    results = example_get_topics()
    save_file(results, f"{OUTPUT_DIR}/get_topics.json")

    logger.info("\nRunning example_get_topic()...")
    results = example_get_topic()
    save_file(results, f"{OUTPUT_DIR}/get_topic.json")

    logger.info("\nRunning example_get_topic_info()...")
    results = example_get_topic_info()
    save_file(results, f"{OUTPUT_DIR}/get_topic_info.json")

    logger.info("\nRunning example_get_topic_freq()...")
    results = example_get_topic_freq()
    save_file(results, f"{OUTPUT_DIR}/get_topic_freq.json")

    logger.info("\nRunning example_get_document_info()...")
    results = example_get_document_info()
    save_file(results, f"{OUTPUT_DIR}/get_document_info.json")

    logger.info("\nRunning example_get_representative_docs()...")
    results = example_get_representative_docs()
    save_file(results, f"{OUTPUT_DIR}/get_representative_docs.json")

    logger.info("\nRunning example_get_topic_tree()...")
    results = example_get_topic_tree()
    save_file(results, f"{OUTPUT_DIR}/get_topic_tree.json")

    logger.info("\nRunning example_set_topic_labels()...")
    results = example_set_topic_labels()
    save_file(results, f"{OUTPUT_DIR}/set_topic_labels.json")

    logger.info("\nRunning example_generate_topic_labels()...")
    results = example_generate_topic_labels()
    save_file(results, f"{OUTPUT_DIR}/generate_topic_labels.json")

    logger.info("\nRunning example_merge_topics()...")
    results = example_merge_topics()
    save_file(results, f"{OUTPUT_DIR}/merge_topics.json")

    logger.info("\nRunning example_delete_topics()...")
    results = example_delete_topics()
    save_file(results, f"{OUTPUT_DIR}/delete_topics.json")

    logger.info("\nRunning example_reduce_topics()...")
    results = example_reduce_topics()
    save_file(results, f"{OUTPUT_DIR}/reduce_topics.json")

    logger.info("\nRunning example_reduce_outliers()...")
    results = example_reduce_outliers()
    save_file(results, f"{OUTPUT_DIR}/reduce_outliers.json")

    logger.info("\nRunning example_visualize_topics()...")
    results = example_visualize_topics()
    save_file(results, f"{OUTPUT_DIR}/visualize_topics.json")

    logger.info("\nRunning example_visualize_documents()...")
    results = example_visualize_documents()
    save_file(results, f"{OUTPUT_DIR}/visualize_documents.json")

    logger.info("\nRunning example_visualize_document_datamap()...")
    results = example_visualize_document_datamap()
    save_file(results, f"{OUTPUT_DIR}/visualize_document_datamap.json")

    logger.info("\nRunning example_visualize_hierarchical_documents()...")
    results = example_visualize_hierarchical_documents()
    save_file(results, f"{OUTPUT_DIR}/visualize_hierarchical_documents.json")

    logger.info("\nRunning example_visualize_term_rank()...")
    results = example_visualize_term_rank()
    save_file(results, f"{OUTPUT_DIR}/visualize_term_rank.json")

    logger.info("\nRunning example_visualize_topics_over_time()...")
    results = example_visualize_topics_over_time()
    save_file(results, f"{OUTPUT_DIR}/visualize_topics_over_time.json")

    logger.info("\nRunning example_visualize_topics_per_class()...")
    results = example_visualize_topics_per_class()
    save_file(results, f"{OUTPUT_DIR}/visualize_topics_per_class.json")

    logger.info("\nRunning example_visualize_distribution()...")
    results = example_visualize_distribution()
    save_file(results, f"{OUTPUT_DIR}/visualize_distribution.json")

    logger.info("\nRunning example_visualize_approximate_distribution()...")
    results = example_visualize_approximate_distribution()
    save_file(results, f"{OUTPUT_DIR}/visualize_approximate_distribution.json")

    logger.info("\nRunning example_visualize_hierarchy()...")
    results = example_visualize_hierarchy()
    save_file(results, f"{OUTPUT_DIR}/visualize_hierarchy.json")

    logger.info("\nRunning example_visualize_heatmap()...")
    results = example_visualize_heatmap()
    save_file(results, f"{OUTPUT_DIR}/visualize_heatmap.json")

    logger.info("\nRunning example_visualize_barchart()...")
    results = example_visualize_barchart()
    save_file(results, f"{OUTPUT_DIR}/visualize_barchart.json")

    logger.info("\nRunning example_save()...")
    results = example_save()
    save_file(results, f"{OUTPUT_DIR}/save.json")

    logger.info("\nRunning example_load()...")
    results = example_load()
    save_file(results, f"{OUTPUT_DIR}/load.json")

    logger.info("\nRunning example_merge_models()...")
    results = example_merge_models()
    save_file(results, f"{OUTPUT_DIR}/merge_models.json")

    logger.info("\nRunning example_push_to_hf_hub()...")
    results = example_push_to_hf_hub()
    save_file(results, f"{OUTPUT_DIR}/push_to_hf_hub.json")

    logger.info("\nRunning example_get_params()...")
    results = example_get_params()
    save_file(results, f"{OUTPUT_DIR}/get_params.json")

    logger.info("\nRunning example__extract_embeddings()...")
    results = example__extract_embeddings()
    save_file(results, f"{OUTPUT_DIR}/_extract_embeddings.json")

    logger.info("\nRunning example__images_to_text()...")
    results = example__images_to_text()
    save_file(results, f"{OUTPUT_DIR}/_images_to_text.json")

    logger.info("\nRunning example__map_predictions()...")
    results = example__map_predictions()
    save_file(results, f"{OUTPUT_DIR}/_map_predictions.json")

    logger.info("\nRunning example__reduce_dimensionality()...")
    results = example__reduce_dimensionality()
    save_file(results, f"{OUTPUT_DIR}/_reduce_dimensionality.json")

    logger.info("\nRunning example__cluster_embeddings()...")
    results = example__cluster_embeddings()
    save_file(results, f"{OUTPUT_DIR}/_cluster_embeddings.json")

    logger.info("\nRunning example__zeroshot_topic_modeling()...")
    results = example__zeroshot_topic_modeling()
    save_file(results, f"{OUTPUT_DIR}/_zeroshot_topic_modeling.json")

    logger.info("\nRunning example__is_zeroshot()...")
    results = example__is_zeroshot()
    save_file(results, f"{OUTPUT_DIR}/_is_zeroshot.json")

    logger.info("\nRunning example__combine_zeroshot_topics()...")
    results = example__combine_zeroshot_topics()
    save_file(results, f"{OUTPUT_DIR}/_combine_zeroshot_topics.json")

    logger.info("\nRunning example__guided_topic_modeling()...")
    results = example__guided_topic_modeling()
    save_file(results, f"{OUTPUT_DIR}/_guided_topic_modeling.json")

    logger.info("\nRunning example__extract_topics()...")
    results = example__extract_topics()
    save_file(results, f"{OUTPUT_DIR}/_extract_topics.json")

    logger.info("\nRunning example__save_representative_docs()...")
    results = example__save_representative_docs()
    save_file(results, f"{OUTPUT_DIR}/_save_representative_docs.json")

    logger.info("\nRunning example__extract_representative_docs()...")
    results = example__extract_representative_docs()
    save_file(results, f"{OUTPUT_DIR}/_extract_representative_docs.json")

    logger.info("\nRunning example__create_topic_vectors()...")
    results = example__create_topic_vectors()
    save_file(results, f"{OUTPUT_DIR}/_create_topic_vectors.json")

    logger.info("\nRunning example__c_tf_idf()...")
    results = example__c_tf_idf()
    save_file(results, f"{OUTPUT_DIR}/_c_tf_idf.json")

    logger.info("\nRunning example__update_topic_size()...")
    results = example__update_topic_size()
    save_file(results, f"{OUTPUT_DIR}/_update_topic_size.json")

    logger.info("\nRunning example__extract_words_per_topic()...")
    results = example__extract_words_per_topic()
    save_file(results, f"{OUTPUT_DIR}/_extract_words_per_topic.json")

    logger.info("\nRunning example__reduce_topics()...")
    results = example__reduce_topics()
    save_file(results, f"{OUTPUT_DIR}/_reduce_topics.json")

    logger.info("\nRunning example__reduce_to_n_topics()...")
    results = example__reduce_to_n_topics()
    save_file(results, f"{OUTPUT_DIR}/_reduce_to_n_topics.json")

    logger.info("\nRunning example__auto_reduce_topics()...")
    results = example__auto_reduce_topics()
    save_file(results, f"{OUTPUT_DIR}/_auto_reduce_topics.json")

    logger.info("\nRunning example__sort_mappings_by_frequency()...")
    results = example__sort_mappings_by_frequency()
    save_file(results, f"{OUTPUT_DIR}/_sort_mappings_by_frequency.json")

    logger.info("\nRunning example__map_probabilities()...")
    results = example__map_probabilities()
    save_file(results, f"{OUTPUT_DIR}/_map_probabilities.json")

    logger.info("\nRunning example__preprocess_text()...")
    results = example__preprocess_text()
    save_file(results, f"{OUTPUT_DIR}/_preprocess_text.json")

    logger.info("\nRunning example__top_n_idx_sparse()...")
    results = example__top_n_idx_sparse()
    save_file(results, f"{OUTPUT_DIR}/_top_n_idx_sparse.json")

    logger.info("\nRunning example__top_n_values_sparse()...")
    results = example__top_n_values_sparse()
    save_file(results, f"{OUTPUT_DIR}/_top_n_values_sparse.json")

    logger.info("\nRunning example__get_param_names()...")
    results = example__get_param_names()
    save_file(results, f"{OUTPUT_DIR}/_get_param_names.json")

    logger.info("\nRunning example__str__()...")
    results = example__str__()
    save_file(results, f"{OUTPUT_DIR}/_str__.json")

    logger.info("\nRunning example_topicmapper_init()...")
    results = example_topicmapper_init()
    save_file(results, f"{OUTPUT_DIR}/topicmapper_init.json")

    logger.info("\nRunning example_topicmapper_get_mappings()...")
    results = example_topicmapper_get_mappings()
    save_file(results, f"{OUTPUT_DIR}/topicmapper_get_mappings.json")

    logger.info("\nRunning example_topicmapper_add_mappings()...")
    results = example_topicmapper_add_mappings()
    save_file(results, f"{OUTPUT_DIR}/topicmapper_add_mappings.json")

    logger.info("\nRunning example_topicmapper_add_new_topics()...")
    results = example_topicmapper_add_new_topics()
    save_file(results, f"{OUTPUT_DIR}/topicmapper_add_new_topics.json")

