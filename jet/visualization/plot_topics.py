from typing import List, Dict
from collections import Counter
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import umap
import pandas as pd
import os
import json
from jet.logger import logger
from sklearn.cluster import KMeans
from keybert import KeyBERT


def categorize_documents_with_bertopic(documents: List[Dict], min_topic_size: int = 2) -> List[str]:
    texts = [doc["content"] for doc in documents]
    if len(texts) < 3:
        logger.warning(
            "Dataset too small (<3 documents); assigning keyword-based topics")
        kw_model = KeyBERT()
        topic_assignments = []
        for doc in texts:
            keywords = kw_model.extract_keywords(doc, top_n=1)
            topic_assignments.append(
                keywords[0][0] if keywords else f"Topic_{len(topic_assignments)+1}")
        logger.debug(
            f"Keyword-based topic assignments: {dict(zip([doc['id'] for doc in documents], topic_assignments))}")
        return topic_assignments

    # Optimize UMAP for small datasets
    n_components = min(2, len(documents) - 1)
    n_neighbors = min(3, len(documents) - 1)
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        random_state=42
    )
    # Use KMeans instead of HDBSCAN for small datasets
    kmeans_model = KMeans(n_clusters=min(3, len(documents)), random_state=42)
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        representation_model=KeyBERTInspired(),
        min_topic_size=max(min_topic_size, len(documents) // 4),
        nr_topics=None,  # Disable auto-reduction
        hdbscan_model=kmeans_model,  # Use KMeans
        low_memory=True,
        umap_model=umap_model,
        calculate_probabilities=False
    )
    try:
        topics, _ = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()

        # Debug: Log topic information
        logger.debug(f"Topic info:\n{topic_info.to_string()}")
        topic_names = {row["Topic"]: row["Name"]
                       for _, row in topic_info.iterrows()}
        topic_names[-1] = "Outlier"
        topic_assignments = [topic_names[topic] for topic in topics]

        # Debug: Log document assignments
        logger.debug(
            f"Document topic assignments: {dict(zip([doc['id'] for doc in documents], topic_assignments))}")

        # Warn if only outliers
        if len(set(topic_assignments) - {"Outlier"}) < 1:
            logger.warning(
                "Only outlier topic detected; using keyword-based topics")
            kw_model = KeyBERT()
            topic_assignments = []
            for doc in texts:
                keywords = kw_model.extract_keywords(doc, top_n=1)
                topic_assignments.append(
                    keywords[0][0] if keywords else f"Topic_{len(topic_assignments)+1}")
            logger.debug(
                f"Keyword-based topic assignments: {dict(zip([doc['id'] for doc in documents], topic_assignments))}")

        return topic_assignments
    except Exception as e:
        logger.error(f"BERTopic failed: {e}")
        logger.warning("Falling back to keyword-based topic assignments")
        kw_model = KeyBERT()
        topic_assignments = []
        for doc in texts:
            keywords = kw_model.extract_keywords(doc, top_n=1)
            topic_assignments.append(
                keywords[0][0] if keywords else f"Topic_{len(topic_assignments)+1}")
        logger.debug(
            f"Keyword-based topic assignments: {dict(zip([doc['id'] for doc in documents], topic_assignments))}")
        return topic_assignments


def aggregate_by_category(documents: List[Dict], min_topic_size: int = 2) -> Dict[str, int]:
    """
    Aggregate documents by dynamically assigned topics.

    Args:
        documents: List of dictionaries with document metadata (must include 'content').
        min_topic_size: Minimum topic size for BERTopic (default: 2).

    Returns:
        Dictionary with topic names as keys and document counts as values.
    """
    topics = categorize_documents_with_bertopic(documents, min_topic_size)
    return dict(Counter(topics))


def generate_chartjs_config(category_counts: Dict[str, int]) -> Dict:
    """
    Generate a Chart.js bar chart configuration for category counts.

    Args:
        category_counts: Dictionary of topic names and their counts.

    Returns:
        A Chart.js configuration dictionary.
    """
    colors = [
        ("rgba(75, 192, 192, 0.8)", "rgba(75, 192, 192, 1)"),  # Teal
        ("rgba(255, 159, 64, 0.8)", "rgba(255, 159, 64, 1)"),    # Orange
        ("rgba(153, 102, 255, 0.8)", "rgba(153, 102, 255, 1)"),  # Purple
        ("rgba(255, 99, 132, 0.8)", "rgba(255, 99, 132, 1)")     # Red
    ]

    labels = list(category_counts.keys())
    data = list(category_counts.values())
    background_colors = [colors[i % len(colors)][0]
                         for i in range(len(labels))]
    border_colors = [colors[i % len(colors)][1] for i in range(len(labels))]

    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Number of Documents",
                "data": data,
                "backgroundColor": background_colors,
                "borderColor": border_colors,
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {"display": True, "text": "Number of Documents"},
                    "ticks": {"stepSize": 1}
                },
                "x": {
                    "title": {"display": True, "text": "Topic"}
                }
            },
            "plugins": {
                "legend": {"display": False},
                "title": {"display": True, "text": "Document Distribution by Topic"}
            }
        }
    }


def process_documents_for_chart(documents: List[Dict], output_dir: str = ".", min_topic_size: int = 2) -> Dict:
    """
    Process documents to generate a Chart.js configuration and save category counts for R.

    Args:
        documents: List of dictionaries with document metadata (must include 'content').
        output_dir: Directory to save category counts CSV and Chart.js JSON (default: current directory).
        min_topic_size: Minimum topic size for BERTopic (default: 2).

    Returns:
        A Chart.js configuration dictionary.

    Raises:
        ValueError: If documents list is empty or category counts are empty.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    csv_output_path = os.path.join(output_dir, "category_counts.csv")
    json_output_path = os.path.join(output_dir, "chart_config.json")

    category_counts = aggregate_by_category(documents, min_topic_size)
    if not category_counts:
        raise ValueError("No categories generated from documents")

    df = pd.DataFrame(list(category_counts.items()),
                      columns=['Topic', 'Count'])
    df.to_csv(csv_output_path, index=False)

    chart_config = generate_chartjs_config(category_counts)
    with open(json_output_path, "w") as f:
        json.dump(chart_config, f, indent=2)

    return chart_config
