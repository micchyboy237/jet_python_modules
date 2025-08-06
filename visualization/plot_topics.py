from typing import List, Dict, Tuple
from collections import Counter
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import json


def categorize_documents_with_bertopic(documents: List[Dict]) -> List[str]:
    texts = [doc["content"] for doc in documents]
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        representation_model=KeyBERTInspired(),  # Avoid LlamaCPP
        min_topic_size=2,
        calculate_probabilities=False
    )
    topics, _ = topic_model.fit_transform(texts)
    topic_info = topic_model.get_topic_info()
    topic_names = {row["Topic"]: row["Name"]
                   for _, row in topic_info.iterrows()}
    topic_names[-1] = "Outlier"
    return [topic_names[topic] for topic in topics]


def aggregate_by_category(documents: List[Dict]) -> Dict[str, int]:
    """
    Aggregate documents by dynamically assigned topics.

    Args:
        documents: List of dictionaries with document metadata (must include 'content').

    Returns:
        Dictionary with topic names as keys and document counts as values.
    """
    topics = categorize_documents_with_bertopic(documents)
    return dict(Counter(topics))


def generate_chartjs_config(category_counts: Dict[str, int]) -> Dict:
    """
    Generate a Chart.js bar chart configuration for category counts.

    Args:
        category_counts: Dictionary of topic names and their counts.

    Returns:
        A dictionary representing the Chart.js configuration.
    """
    # Define distinct colors for dark/light themes
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


def process_documents_for_chart(documents: List[Dict]) -> Dict:
    """
    Process documents to generate a Chart.js configuration for topic visualization.

    Args:
        documents: List of dictionaries with document metadata (must include 'content').

    Returns:
        A Chart.js configuration dictionary.
    """
    category_counts = aggregate_by_category(documents)
    return generate_chartjs_config(category_counts)
