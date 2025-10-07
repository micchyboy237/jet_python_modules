from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import BaseRepresentation
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.graph_objects as go

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

def example_n_gram_range() -> Tuple[int, int]:
    """
    Demonstrates the n_gram_range property.
    - Shows how to access the n-gram range used in the CountVectorizer.
    """
    model = BERTopic(n_gram_range=(1, 2))
    return model.n_gram_range

def example_calculate_probabilities() -> bool:
    """
    Demonstrates the calculate_probabilities property.
    - Shows how to check if topic probabilities are calculated.
    """
    model = BERTopic(calculate_probabilities=True)
    return model.calculate_probabilities

def example_verbose() -> bool:
    """
    Demonstrates the verbose property.
    - Shows how to check if verbose logging is enabled.
    """
    model = BERTopic(verbose=True)
    return model.verbose

def example_seed_topic_list() -> List[List[str]]:
    """
    Demonstrates the seed_topic_list property.
    - Shows how to access seed topics for guided topic modeling.
    """
    seed_topics = [["fox", "dog"], ["machine", "learning"]]
    model = BERTopic(seed_topic_list=seed_topics)
    return model.seed_topic_list

def example_zeroshot_topic_list() -> List[str]:
    """
    Demonstrates the zeroshot_topic_list property.
    - Shows how to access zero-shot topic names.
    """
    zeroshot_topics = ["animals", "technology"]
    model = BERTopic(zeroshot_topic_list=zeroshot_topics)
    return model.zeroshot_topic_list

def example_zeroshot_min_similarity() -> float:
    """
    Demonstrates the zeroshot_min_similarity property.
    - Shows how to access the minimum similarity threshold for zero-shot topics.
    """
    model = BERTopic(zeroshot_min_similarity=0.8)
    return model.zeroshot_min_similarity

def example_language() -> str:
    """
    Demonstrates the language property.
    - Shows how to access the language setting for the embedding model.
    """
    model = BERTopic(language="multilingual")
    return model.language

def example_embedding_model() -> Any:
    """
    Demonstrates the embedding_model property.
    - Shows how to access the embedding model used.
    """
    model = BERTopic(embedding_model="all-MiniLM-L6-v2")
    return model.embedding_model

def example_vectorizer_model() -> CountVectorizer:
    """
    Demonstrates the vectorizer_model property.
    - Shows how to access the CountVectorizer used for text preprocessing.
    """
    vectorizer = CountVectorizer(stop_words="english")
    model = BERTopic(vectorizer_model=vectorizer)
    return model.vectorizer_model

def example_ctfidf_model() -> ClassTfidfTransformer:
    """
    Demonstrates the ctfidf_model property.
    - Shows how to access the c-TF-IDF transformer.
    """
    ctfidf = ClassTfidfTransformer()
    model = BERTopic(ctfidf_model=ctfidf)
    return model.ctfidf_model

def example_representation_model() -> BaseRepresentation:
    """
    Demonstrates the representation_model property.
    - Shows how to access the representation model for fine-tuning topics.
    """
    model = BERTopic(representation_model=None)
    return model.representation_model

def example_umap_model() -> UMAP:
    """
    Demonstrates the umap_model property.
    - Shows how to access the UMAP model for dimensionality reduction.
    """
    umap = UMAP(n_components=2)
    model = BERTopic(umap_model=umap)
    return model.umap_model

def example_hdbscan_model() -> HDBSCAN:
    """
    Demonstrates the hdbscan_model property.
    - Shows how to access the HDBSCAN model for clustering.
    """
    hdbscan = HDBSCAN(min_cluster_size=2)
    model = BERTopic(hdbscan_model=hdbscan)
    return model.hdbscan_model

def example_topic_sizes_() -> Mapping[int, int]:
    """
    Demonstrates the topic_sizes_ property.
    - Shows how to access the size of each topic after fitting.
    """
    model = BERTopic(min_topic_size=2)
    model.fit(documents=SAMPLE_DOCS)
    return model.topic_sizes_

def example_topic_mapper_() -> Any:
    """
    Demonstrates the topic_mapper_ property.
    - Shows how to access the topic mapper after fitting.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    return model.topic_mapper_

def example_topic_embeddings_() -> np.ndarray:
    """
    Demonstrates the topic_embeddings_ property.
    - Shows how to access topic embeddings after fitting.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    return model.topic_embeddings_

def example__topic_id_to_zeroshot_topic_idx() -> Mapping[int, int]:
    """
    Demonstrates the _topic_id_to_zeroshot_topic_idx property.
    - Shows how to access the mapping of topic IDs to zero-shot topic indices.
    """
    model = BERTopic(zeroshot_topic_list=["animals", "technology"])
    model.fit(documents=SAMPLE_DOCS)
    return model._topic_id_to_zeroshot_topic_idx

def example_custom_labels_() -> List[str]:
    """
    Demonstrates the custom_labels_ property.
    - Shows how to access custom topic labels after setting them.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    model.set_topic_labels(["topic_1", "topic_2"])
    return model.custom_labels_

def example_c_tf_idf_() -> csr_matrix:
    """
    Demonstrates the c_tf_idf_ property.
    - Shows how to access the c-TF-IDF matrix after fitting.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    return model.c_tf_idf_

def example_representative_images_() -> Any:
    """
    Demonstrates the representative_images_ property.
    - Shows how to access representative images for topics (if used).
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS, images=["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"])
    return model.representative_images_

def example_representative_docs_() -> Mapping[int, List[str]]:
    """
    Demonstrates the representative_docs_ property.
    - Shows how to access representative documents for all topics.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    return model.representative_docs_

def example_topic_aspects_() -> Mapping[str, Any]:
    """
    Demonstrates the topic_aspects_ property.
    - Shows how to access topic aspects after fitting.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    return model.topic_aspects_

def example__merged_topics() -> Any:
    """
    Demonstrates the _merged_topics property.
    - Shows how to access merged topic information after merging.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    model.merge_topics(docs=SAMPLE_DOCS, topics_to_merge=[[0, 1]])
    return model._merged_topics

def example__outliers():
    """
    Demonstrates the _outliers method.
    - Internal method, not typically used directly, but shown for completeness.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    # Note: This is an internal method, so direct usage is not recommended.
    print("Internal method _outliers not directly accessible.")

def example_topic_labels_():
    """
    Demonstrates the topic_labels_ method.
    - Internal method to get topic labels, shown for completeness.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    # Note: This is an internal method, so direct usage is not recommended.
    print("Internal method topic_labels_ not directly accessible.")

def example_fit(documents: List[str] = SAMPLE_DOCS) -> None:
    """
    Demonstrates the fit method.
    - Fits the model on documents without returning topics or probabilities.
    """
    model = BERTopic()
    model.fit(documents=documents)
    print(f"Model fitted with {len(model.topics_)} topics.")

def example_transform() -> Tuple[List[int], np.ndarray]:
    """
    Demonstrates the transform method.
    - Transforms new documents into topics using a fitted model.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS[:3])
    topics, probabilities = model.transform(documents=SAMPLE_DOCS[3:])
    print(f"Transformed topics: {topics}")
    return topics, probabilities

def example_partial_fit() -> None:
    """
    Demonstrates the partial_fit method.
    - Updates the model with new documents incrementally.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS[:3])
    model.partial_fit(documents=SAMPLE_DOCS[3:])
    print(f"Model updated with partial fit: {model.topics_}")

def example_topics_over_time() -> pd.DataFrame:
    """
    Demonstrates the topics_over_time method.
    - Analyzes topic evolution over time using timestamps.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    topics_over_time = model.topics_over_time(docs=SAMPLE_DOCS, timestamps=SAMPLE_TIMESTAMPS)
    print(f"Topics over time:\n{topics_over_time}")
    return topics_over_time

def example_topics_per_class() -> pd.DataFrame:
    """
    Demonstrates the topics_per_class method.
    - Analyzes topics across different classes.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    topics_per_class = model.topics_per_class(docs=SAMPLE_DOCS, classes=SAMPLE_CLASSES)
    print(f"Topics per class:\n{topics_per_class}")
    return topics_per_class

def example_hierarchical_topics() -> pd.DataFrame:
    """
    Demonstrates the hierarchical_topics method.
    - Generates a hierarchical structure of topics.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    hier_topics = model.hierarchical_topics(docs=SAMPLE_DOCS)
    print(f"Hierarchical topics:\n{hier_topics}")
    return hier_topics

def example_approximate_distribution() -> Tuple[np.ndarray, Union[List[np.ndarray], None]]:
    """
    Demonstrates the approximate_distribution method.
    - Approximates topic distributions for documents.
    """
    model = BERTopic(calculate_probabilities=True)
    model.fit(documents=SAMPLE_DOCS)
    distribution, _ = model.approximate_distribution(documents=SAMPLE_DOCS)
    print(f"Approximate distribution shape: {distribution.shape}")
    return distribution, None

def example_find_topics() -> Tuple[List[int], List[float]]:
    """
    Demonstrates the find_topics method.
    - Finds topics most similar to a search term.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    topics, similarities = model.find_topics(search_term="machine learning")
    print(f"Topics for 'machine learning': {topics}, Similarities: {similarities}")
    return topics, similarities

def example_get_topics() -> Mapping[str, Tuple[str, float]]:
    """
    Demonstrates the get_topics method.
    - Retrieves all topic representations.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    topics = model.get_topics()
    print(f"All topics:\n{topics}")
    return topics

def example_get_topic(topic: int = 0) -> Union[Mapping[str, Tuple[str, float]], bool]:
    """
    Demonstrates the get_topic method.
    - Retrieves representation for a specific topic.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    topic_info = model.get_topic(topic=topic)
    print(f"Topic {topic} info: {topic_info}")
    return topic_info

def example_get_topic_freq(topic: int = None) -> Union[pd.DataFrame, int]:
    """
    Demonstrates the get_topic_freq method.
    - Retrieves frequency of topics or a specific topic.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    freq = model.get_topic_freq(topic=topic)
    print(f"Topic frequencies:\n{freq}")
    return freq

def example_get_document_info() -> pd.DataFrame:
    """
    Demonstrates the get_document_info method.
    - Retrieves detailed information about documents and their topics.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    doc_info = model.get_document_info(docs=SAMPLE_DOCS)
    print(f"Document info:\n{doc_info}")
    return doc_info

def example_get_topic_tree(hier_topics: pd.DataFrame) -> str:
    """
    Demonstrates the get_topic_tree static method.
    - Generates a textual representation of the topic hierarchy.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    hier_topics = model.hierarchical_topics(docs=SAMPLE_DOCS)
    tree = BERTopic.get_topic_tree(hier_topics=hier_topics)
    print(f"Topic tree:\n{tree}")
    return tree

def example_set_topic_labels() -> None:
    """
    Demonstrates the set_topic_labels method.
    - Sets custom labels for topics.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    model.set_topic_labels(["animal_topic", "tech_topic"])
    print(f"Custom labels: {model.custom_labels_}")

def example_generate_topic_labels() -> List[str]:
    """
    Demonstrates the generate_topic_labels method.
    - Generates automatic topic labels based on top words.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    labels = model.generate_topic_labels(nr_words=2, separator="-")
    print(f"Generated labels: {labels}")
    return labels

def example_merge_topics() -> None:
    """
    Demonstrates the merge_topics method.
    - Merges specified topics into a single topic.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    model.merge_topics(docs=SAMPLE_DOCS, topics_to_merge=[[0, 1]])
    print(f"Topics after merging: {model.topics_}")

def example_delete_topics() -> None:
    """
    Demonstrates the delete_topics method.
    - Deletes specified topics from the model.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    model.delete_topics(topics_to_delete=[0])
    print(f"Topics after deletion: {model.topics_}")

def example_reduce_outliers() -> List[int]:
    """
    Demonstrates the reduce_outliers method.
    - Reassigns outlier documents to topics based on strategy.
    """
    model = BERTopic(calculate_probabilities=True)
    model.fit(documents=SAMPLE_DOCS)
    topics = model.reduce_outliers(documents=SAMPLE_DOCS, topics=model.topics_, strategy="distributions")
    print(f"Topics after outlier reduction: {topics}")
    return topics

def example_visualize_documents() -> go.Figure:
    """
    Demonstrates the visualize_documents method.
    - Visualizes documents in a 2D space with topic assignments.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_documents(docs=SAMPLE_DOCS, title="Document Visualization")
    fig.show()
    return fig

def example_visualize_document_datamap() -> go.Figure:
    """
    Demonstrates the visualize_document_datamap method.
    - Visualizes documents with an interactive datamap.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_document_datamap(docs=SAMPLE_DOCS, title="Document Datamap")
    fig.show()
    return fig

def example_visualize_hierarchical_documents() -> go.Figure:
    """
    Demonstrates the visualize_hierarchical_documents method.
    - Visualizes documents with hierarchical topic structure.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    hier_topics = model.hierarchical_topics(docs=SAMPLE_DOCS)
    fig = model.visualize_hierarchical_documents(docs=SAMPLE_DOCS, hierarchical_topics=hier_topics)
    fig.show()
    return fig

def example_visualize_term_rank() -> go.Figure:
    """
    Demonstrates the visualize_term_rank method.
    - Visualizes the decline in term scores for topics.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_term_rank(title="Term Rank Visualization")
    fig.show()
    return fig

def example_visualize_topics_over_time() -> go.Figure:
    """
    Demonstrates the visualize_topics_over_time method.
    - Visualizes topic frequency changes over time.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    topics_over_time = model.topics_over_time(docs=SAMPLE_DOCS, timestamps=SAMPLE_TIMESTAMPS)
    fig = model.visualize_topics_over_time(topics_over_time=topics_over_time)
    fig.show()
    return fig

def example_visualize_topics_per_class() -> go.Figure:
    """
    Demonstrates the visualize_topics_per_class method.
    - Visualizes topic distributions across classes.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    topics_per_class = model.topics_per_class(docs=SAMPLE_DOCS, classes=SAMPLE_CLASSES)
    fig = model.visualize_topics_per_class(topics_per_class=topics_per_class)
    fig.show()
    return fig

def example_visualize_distribution() -> go.Figure:
    """
    Demonstrates the visualize_distribution method.
    - Visualizes topic probability distributions.
    """
    model = BERTopic(calculate_probabilities=True)
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_distribution(probabilities=model.probabilities_)
    fig.show()
    return fig

def example_visualize_approximate_distribution() -> None:
    """
    Demonstrates the visualize_approximate_distribution method.
    - Visualizes approximate topic distribution for a single document.
    """
    model = BERTopic(calculate_probabilities=True)
    model.fit(documents=SAMPLE_DOCS)
    distribution, _ = model.approximate_distribution(documents=[SAMPLE_DOCS[0]])
    model.visualize_approximate_distribution(document=SAMPLE_DOCS[0], topic_token_distribution=distribution)

def example_visualize_hierarchy() -> go.Figure:
    """
    Demonstrates the visualize_hierarchy method.
    - Visualizes the hierarchical clustering of topics.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    hier_topics = model.hierarchical_topics(docs=SAMPLE_DOCS)
    fig = model.visualize_hierarchy(hierarchical_topics=hier_topics)
    fig.show()
    return fig

def example_visualize_heatmap() -> go.Figure:
    """
    Demonstrates the visualize_heatmap method.
    - Visualizes a similarity matrix of topics.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_heatmap(title="Topic Similarity Heatmap")
    fig.show()
    return fig

def example_visualize_barchart() -> go.Figure:
    """
    Demonstrates the visualize_barchart method.
    - Visualizes word scores for top topics.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_barchart(title="Topic Word Scores")
    fig.show()
    return fig

def example_save() -> None:
    """
    Demonstrates the save method.
    - Saves the model to a file.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    model.save(path="bertopic_model.pkl", serialization="pickle")

def example_load() -> BERTopic:
    """
    Demonstrates the load class method.
    - Loads a saved BERTopic model from a file.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    model.save(path="bertopic_model.pkl", serialization="pickle")
    loaded_model = BERTopic.load(path="bertopic_model.pkl")
    return loaded_model

def example_merge_models() -> BERTopic:
    """
    Demonstrates the merge_models class method.
    - Merges multiple BERTopic models.
    """
    model1 = BERTopic()
    model1.fit(documents=SAMPLE_DOCS[:3])
    model2 = BERTopic()
    model2.fit(documents=SAMPLE_DOCS[3:])
    merged_model = BERTopic.merge_models(models=[model1, model2], min_similarity=0.7)
    return merged_model

def example_push_to_hf_hub() -> None:
    """
    Demonstrates the push_to_hf_hub method.
    - Pushes the model to Hugging Face Hub (mock example, requires token).
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    # Note: Requires a valid token and repo_id for actual use.
    model.push_to_hf_hub(repo_id="user/bertopic_model", commit_message="Test push")

def example_get_params() -> Mapping[str, Any]:
    """
    Demonstrates the get_params method.
    - Retrieves the model's parameters.
    """
    model = BERTopic()
    params = model.get_params()
    print(f"Model parameters: {params}")
    return params

def example__extract_embeddings() -> np.ndarray:
    """
    Demonstrates the _extract_embeddings method.
    - Internal method to extract embeddings, shown for completeness.
    """
    model = BERTopic()
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def example__images_to_text() -> pd.DataFrame:
    """
    Demonstrates the _images_to_text method.
    - Internal method for multimodal processing, shown for completeness.
    """
    model = BERTopic()
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    result = model._images_to_text(documents=df, embeddings=embeddings)
    print(f"Images to text result:\n{result}")
    return result

def example__map_predictions() -> List[int]:
    """
    Demonstrates the _map_predictions method.
    - Internal method to map predictions, shown for completeness.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    predictions = model.topics_
    mapped = model._map_predictions(predictions=predictions)
    print(f"Mapped predictions: {mapped}")
    return mapped

def example__reduce_dimensionality() -> np.ndarray:
    """
    Demonstrates the _reduce_dimensionality method.
    - Internal method for dimensionality reduction, shown for completeness.
    """
    model = BERTopic()
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    reduced = model._reduce_dimensionality(embeddings=embeddings)
    print(f"Reduced embeddings shape: {reduced.shape}")
    return reduced

def example__cluster_embeddings() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Demonstrates the _cluster_embeddings method.
    - Internal method for clustering embeddings, shown for completeness.
    """
    model = BERTopic()
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    reduced = model._reduce_dimensionality(embeddings=embeddings)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    result, probs = model._cluster_embeddings(umap_embeddings=reduced, documents=df)
    print(f"Clustered documents:\n{result}")
    return result, probs

def example__zeroshot_topic_modeling() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Demonstrates the _zeroshot_topic_modeling method.
    - Internal method for zero-shot topic modeling, shown for completeness.
    """
    model = BERTopic(zeroshot_topic_list=["animals", "technology"])
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    result = model._zeroshot_topic_modeling(documents=df, embeddings=embeddings)
    print(f"Zero-shot modeling result: {result}")
    return result

def example__is_zeroshot():
    """
    Demonstrates the _is_zeroshot method.
    - Internal method to check if zero-shot modeling is used.
    """
    model = BERTopic(zeroshot_topic_list=["animals", "technology"])
    print("Internal method _is_zeroshot not directly accessible.")

def example__combine_zeroshot_topics() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Demonstrates the _combine_zeroshot_topics method.
    - Internal method for combining zero-shot topics, shown for completeness.
    """
    model = BERTopic(zeroshot_topic_list=["animals", "technology"])
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    assigned_df = df.copy()
    assigned_embeddings = embeddings.copy()
    result = model._combine_zeroshot_topics(documents=df, embeddings=embeddings, assigned_documents=assigned_df, assigned_embeddings=assigned_embeddings)
    print(f"Combined zero-shot topics result: {result}")
    return result

def example__guided_topic_modeling() -> Tuple[List[int], np.ndarray]:
    """
    Demonstrates the _guided_topic_modeling method.
    - Internal method for guided topic modeling, shown for completeness.
    """
    model = BERTopic(seed_topic_list=[["fox", "dog"], ["machine", "learning"]])
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    topics, probs = model._guided_topic_modeling(embeddings=embeddings)
    print(f"Guided topics: {topics}")
    return topics, probs

def example__extract_topics():
    """
    Demonstrates the _extract_topics method.
    - Internal method for topic extraction, shown for completeness.
    """
    model = BERTopic()
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    model._extract_topics(documents=df)
    print("Internal method _extract_topics not directly accessible.")

def example__save_representative_docs():
    """
    Demonstrates the _save_representative_docs method.
    - Internal method for saving representative docs, shown for completeness.
    """
    model = BERTopic()
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    model.fit(documents=SAMPLE_DOCS)
    model._save_representative_docs(documents=df)
    print("Internal method _save_representative_docs not directly accessible.")

def example__extract_representative_docs() -> Union[List[str], List[List[int]]]:
    """
    Demonstrates the _extract_representative_docs method.
    - Internal method for extracting representative docs, shown for completeness.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    topics = model.get_topics()
    result = model._extract_representative_docs(c_tf_idf=model.c_tf_idf_, documents=df, topics=topics)
    print(f"Extracted representative docs: {result}")
    return result

def example__create_topic_vectors():
    """
    Demonstrates the _create_topic_vectors method.
    - Internal method for creating topic vectors, shown for completeness.
    """
    model = BERTopic()
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    model._create_topic_vectors(documents=df)
    print("Internal method _create_topic_vectors not directly accessible.")

def example__c_tf_idf() -> Tuple[csr_matrix, List[str]]:
    """
    Demonstrates the _c_tf_idf method.
    - Internal method for computing c-TF-IDF, shown for completeness.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS, "Topic": model.topics_})
    c_tf_idf, words = model._c_tf_idf(documents_per_topic=df)
    print(f"c-TF-IDF shape: {c_tf_idf.shape}")
    return c_tf_idf, words

def example__update_topic_size():
    """
    Demonstrates the _update_topic_size method.
    - Internal method for updating topic sizes, shown for completeness.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS, "Topic": model.topics_})
    model._update_topic_size(documents=df)
    print("Internal method _update_topic_size not directly accessible.")

def example__extract_words_per_topic() -> Mapping[str, List[Tuple[str, float]]]:
    """
    Demonstrates the _extract_words_per_topic method.
    - Internal method for extracting words per topic, shown for completeness.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS, "Topic": model.topics_})
    words = model.vectorizer_model.get_feature_names_out()
    result = model._extract_words_per_topic(words=words, documents=df, c_tf_idf=model.c_tf_idf_)
    print(f"Words per topic: {result}")
    return result

def example__reduce_topics() -> pd.DataFrame:
    """
    Demonstrates the _reduce_topics method.
    - Internal method for reducing topics, shown for completeness.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    result = model._reduce_topics(documents=df)
    print(f"Reduced topics result:\n{result}")
    return result

def example__reduce_to_n_topics() -> pd.DataFrame:
    """
    Demonstrates the _reduce_to_n_topics method.
    - Internal method for reducing to a specific number of topics.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    result = model._reduce_to_n_topics(documents=df, nr_topics=2)
    print(f"Reduced to n topics result:\n{result}")
    return result

def example__auto_reduce_topics() -> pd.DataFrame:
    """
    Demonstrates the _auto_reduce_topics method.
    - Internal method for automatic topic reduction, shown for completeness.
    """
    model = BERTopic(nr_topics="auto")
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS})
    result = model._auto_reduce_topics(documents=df)
    print(f"Auto reduced topics result:\n{result}")
    return result

def example__sort_mappings_by_frequency() -> pd.DataFrame:
    """
    Demonstrates the _sort_mappings_by_frequency method.
    - Internal method for sorting topic mappings, shown for completeness.
    """
    model = BERTopic()
    model.fit(documents=SAMPLE_DOCS)
    df = pd.DataFrame({"Document": SAMPLE_DOCS, "Topic": model.topics_})
    result = model._sort_mappings_by_frequency(documents=df)
    print(f"Sorted mappings result:\n{result}")
    return result

def example__map_probabilities() -> Union[np.ndarray, None]:
    """
    Demonstrates the _map_probabilities method.
    - Internal method for mapping probabilities, shown for completeness.
    """
    model = BERTopic(calculate_probabilities=True)
    model.fit(documents=SAMPLE_DOCS)
    result = model._map_probabilities(probabilities=model.probabilities_)
    print(f"Mapped probabilities shape: {result.shape if result is not None else None}")
    return result

def example__preprocess_text() -> List[str]:
    """
    Demonstrates the _preprocess_text method.
    - Internal method for preprocessing text, shown for completeness.
    """
    model = BERTopic()
    documents = np.array(SAMPLE_DOCS)
    result = model._preprocess_text(documents=documents)
    print(f"Preprocessed text: {result}")
    return result

def example__top_n_idx_sparse() -> np.ndarray:
    """
    Demonstrates the _top_n_idx_sparse static method.
    - Retrieves top N indices from a sparse matrix.
    """
    from scipy.sparse import csr_matrix
    matrix = csr_matrix([[0, 1, 2], [3, 0, 4], [5, 6, 0]])
    indices = BERTopic._top_n_idx_sparse(matrix=matrix, n=2)
    print(f"Top N indices: {indices}")
    return indices

def example__top_n_values_sparse() -> np.ndarray:
    """
    Demonstrates the _top_n_values_sparse static method.
    - Retrieves top N values from a sparse matrix given indices.
    """
    from scipy.sparse import csr_matrix
    matrix = csr_matrix([[0, 1, 2], [3, 0, 4], [5, 6, 0]])
    indices = BERTopic._top_n_idx_sparse(matrix=matrix, n=2)
    values = BERTopic._top_n_values_sparse(matrix=matrix, indices=indices)
    print(f"Top N values: {values}")
    return values

def example__get_param_names() -> List[str]:
    """
    Demonstrates the _get_param_names class method.
    - Retrieves parameter names of the BERTopic class.
    """
    param_names = BERTopic._get_param_names()
    print(f"Parameter names: {param_names}")
    return param_names

def example__str__() -> str:
    """
    Demonstrates the __str__ method.
    - Returns a string representation of the BERTopic model.
    """
    model = BERTopic()
    model_str = str(model)
    print(f"Model string representation: {model_str}")
    return model_str