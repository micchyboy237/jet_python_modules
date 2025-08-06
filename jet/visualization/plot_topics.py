from typing import List, Dict
from collections import Counter
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import umap
import pandas as pd
import os
import json


def categorize_documents_with_bertopic(documents: List[Dict], min_topic_size: int = 2) -> List[str]:
    texts = [doc["content"] for doc in documents]
    umap_model = umap.UMAP(n_components=2, random_state=42)
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        representation_model=KeyBERTInspired(),
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        umap_model=umap_model
    )
    topics, _ = topic_model.fit_transform(texts)
    topic_info = topic_model.get_topic_info()
    topic_names = {row["Topic"]: row["Name"]
                   for _, row in topic_info.iterrows()}
    topic_names[-1] = "Outlier"
    return [topic_names[topic] for topic in topics]


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
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    csv_output_path = os.path.join(output_dir, "category_counts.csv")
    json_output_path = os.path.join(output_dir, "chart_config.json")

    category_counts = aggregate_by_category(documents, min_topic_size)
    df = pd.DataFrame(list(category_counts.items()),
                      columns=['Topic', 'Count'])
    df.to_csv(csv_output_path, index=False)

    chart_config = generate_chartjs_config(category_counts)
    with open(json_output_path, "w") as f:
        json.dump(chart_config, f, indent=2)

    return chart_config
