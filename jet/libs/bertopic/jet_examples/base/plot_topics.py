from typing import List, Dict, Tuple, TypedDict
from collections import Counter
from jet.adapters.bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import umap
import pandas as pd
import os
import json
from jet.logger import logger
from sklearn.cluster import KMeans
from keybert import KeyBERT


class Document(TypedDict):
    id: int
    content: str


class DocumentTopicAssignment(TypedDict):
    id: int
    label: int
    topic: str
    content: str


def categorize_documents_with_bertopic(documents: List[Document], min_topic_size: int = 2) -> List[DocumentTopicAssignment]:
    texts = [doc["content"] for doc in documents]

    if len(texts) < 3:
        logger.warning(
            "Dataset too small (<3 documents); assigning keyword-based topics")
        kw_model = KeyBERT()
        assignments: List[DocumentTopicAssignment] = []
        for i, doc in enumerate(documents):
            keywords = kw_model.extract_keywords(doc["content"], top_n=1)
            topic = keywords[0][0] if keywords else f"Topic_{i+1}"
            assignments.append({
                "id": doc["id"],
                "label": i,  # fallback to index as pseudo-label
                "topic": topic,
                "content": doc["content"]
            })
        logger.debug(f"Keyword-based topic assignments: {assignments}")
        return assignments

    try:
        n_components = min(2, len(documents) - 1)
        n_neighbors = min(3, len(documents) - 1)

        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            random_state=42
        )
        kmeans_model = KMeans(n_clusters=min(
            3, len(documents)), random_state=42)

        topic_model = BERTopic(
            embedding_model="all-MiniLM-L6-v2",
            representation_model=KeyBERTInspired(),
            min_topic_size=max(min_topic_size, len(documents) // 4),
            nr_topics=None,
            hdbscan_model=kmeans_model,
            low_memory=True,
            umap_model=umap_model,
            calculate_probabilities=False
        )

        topics, _ = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()
        logger.debug(f"Topic info:\n{topic_info.to_string()}")

        topic_names = {row["Topic"]: row["Name"]
                       for _, row in topic_info.iterrows()}
        topic_names[-1] = "Outlier"

        topic_assignments: List[DocumentTopicAssignment] = [
            {
                "id": doc["id"],
                "label": label,
                "topic": topic_names.get(label, "Unknown"),
                "content": doc["content"]
            }
            for doc, label in zip(documents, topics)
        ]
        logger.debug(f"Document topic assignments: {topic_assignments}")

        if len(set([a["topic"] for a in topic_assignments]) - {"Outlier"}) < 1:
            logger.warning(
                "Only outlier topic detected; using keyword-based topics")
            kw_model = KeyBERT()
            topic_assignments = []
            for i, doc in enumerate(documents):
                keywords = kw_model.extract_keywords(doc["content"], top_n=1)
                topic = keywords[0][0] if keywords else f"Topic_{i+1}"
                topic_assignments.append({
                    "id": doc["id"],
                    "topic": topic,
                    "label": i,
                    "content": doc["content"]
                })
            logger.debug(
                f"Keyword-based topic assignments: {topic_assignments}")
        return topic_assignments

    except Exception as e:
        logger.error(f"BERTopic failed: {e}")
        logger.warning("Falling back to keyword-based topic assignments")
        kw_model = KeyBERT()
        assignments: List[DocumentTopicAssignment] = []
        for i, doc in enumerate(documents):
            keywords = kw_model.extract_keywords(doc["content"], top_n=1)
            topic = keywords[0][0] if keywords else f"Topic_{i+1}"
            assignments.append({
                "id": doc["id"],
                "topic": topic,
                "label": i,
                "content": doc["content"]
            })
        logger.debug(f"Keyword-based topic assignments: {assignments}")
        return assignments


def aggregate_by_category(documents: List[Document], min_topic_size: int = 2) -> Tuple[Dict[str, int], List[DocumentTopicAssignment]]:
    assignments = categorize_documents_with_bertopic(documents, min_topic_size)
    counts = Counter([entry["topic"] for entry in assignments])
    return dict(counts), assignments


def generate_chartjs_config(category_counts: Dict[str, int]) -> Dict:
    colors = [
        ("rgba(75, 192, 192, 0.8)", "rgba(75, 192, 192, 1)"),
        ("rgba(255, 159, 64, 0.8)", "rgba(255, 159, 64, 1)"),
        ("rgba(153, 102, 255, 0.8)", "rgba(153, 102, 255, 1)"),
        ("rgba(255, 99, 132, 0.8)", "rgba(255, 99, 132, 1)")
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


def process_documents_for_chart(documents: List[Document], output_dir: str = ".", min_topic_size: int = 2) -> Dict:
    if not documents:
        raise ValueError("Documents list cannot be empty")

    os.makedirs(output_dir, exist_ok=True)

    category_counts_csv = os.path.join(output_dir, "category_counts.csv")
    chart_config_json = os.path.join(output_dir, "chart_config.json")
    topic_mapping_json = os.path.join(
        output_dir, "document_topic_mapping.json")

    category_counts, assignments = aggregate_by_category(
        documents, min_topic_size)

    if not category_counts:
        raise ValueError("No categories generated from documents")

    pd.DataFrame(list(category_counts.items()), columns=[
                 "Topic", "Count"]).to_csv(category_counts_csv, index=False)

    with open(topic_mapping_json, "w") as f:
        json.dump(assignments, f, indent=2)

    chart_config = generate_chartjs_config(category_counts)
    with open(chart_config_json, "w") as f:
        json.dump(chart_config, f, indent=2)

    return chart_config


if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    # Real-world example: Analyzing customer feedback from a tech support system
    sample_documents: List[Document] = [
        {
            "id": 1,
            "content": "The software crashes when I try to export large datasets to CSV."
        },
        {
            "id": 2,
            "content": "Export feature is slow and needs optimization for large files."
        },
        {
            "id": 3,
            "content": "Customer support was very responsive and resolved my login issue quickly."
        },
        {
            "id": 4,
            "content": "I experienced a bug where the dashboard doesn't load properly on Chrome."
        },
        {
            "id": 5,
            "content": "The login system keeps rejecting valid credentials; very frustrating."
        }
    ]

    try:
        # Process documents and generate chart configuration
        chart_config = process_documents_for_chart(
            documents=sample_documents,
            output_dir=OUTPUT_DIR,
            min_topic_size=2
        )
        print("Chart configuration generated successfully:")
        print(json.dumps(chart_config, indent=2))

        # Display topic assignments for verification
        with open(f"{OUTPUT_DIR}/document_topic_mapping.json", "r") as f:
            assignments = json.load(f)
        print("\nDocument Topic Assignments:")
        for assignment in assignments:
            print(
                f"Document ID: {assignment['id']}, Topic: {assignment['topic']}, Content: {assignment['content'][:50]}..."
            )

        # Display category counts
        with open(f"{OUTPUT_DIR}/category_counts.csv", "r") as f:
            print("\nCategory Counts:")
            print(f.read())

    except Exception as e:
        logger.error(f"Error processing documents: {e}")
