import pytest
import pandas as pd
import os
import json
from jet.visualization.plot_topics import process_documents_for_chart, aggregate_by_category


@pytest.fixture
def sample_documents():
    return [
        {"id": 1, "content": "Advances in artificial intelligence."},
        {"id": 2, "content": "Stock market trends are bullish."},
        {"id": 3, "content": "Machine learning improves accuracy."}
    ]


@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path)


def test_process_documents_for_chart_saves_files_in_output_dir(sample_documents, temp_dir):
    # Given: A list of documents and a temporary output directory
    # Expected due to small dataset and min_topic_size
    expected_labels = ["Outlier"]
    expected_counts = [3]
    csv_path = os.path.join(temp_dir, "category_counts.csv")
    json_path = os.path.join(temp_dir, "chart_config.json")

    # When: Processing documents with output_dir
    chart_config = process_documents_for_chart(
        sample_documents, temp_dir, min_topic_size=2)

    # Then: Both files are created with correct content
    assert os.path.exists(csv_path)
    assert os.path.exists(json_path)

    df = pd.read_csv(csv_path)
    assert list(df["Topic"]) == expected_labels
    assert list(df["Count"]) == expected_counts

    with open(json_path, "r") as f:
        json_data = json.load(f)
    assert json_data["type"] == "bar"
    assert json_data["data"]["labels"] == expected_labels
    assert json_data["data"]["datasets"][0]["data"] == expected_counts


def test_process_documents_for_chart_fails_with_empty_documents(temp_dir):
    # Given: An empty list of documents
    documents = []

    # When: Processing documents
    # Then: A ValueError is raised
    with pytest.raises(ValueError, match="Documents list cannot be empty"):
        process_documents_for_chart(documents, temp_dir)


def test_process_documents_for_chart_creates_output_dir(sample_documents, tmp_path):
    # Given: A non-existent output directory
    non_existent_dir = str(tmp_path / "output")
    expected_labels = ["Outlier"]
    expected_counts = [3]
    csv_path = os.path.join(non_existent_dir, "category_counts.csv")

    # When: Processing documents with a non-existent output_dir
    chart_config = process_documents_for_chart(
        sample_documents, non_existent_dir, min_topic_size=2)

    # Then: The directory is created and CSV file is saved
    assert os.path.exists(non_existent_dir)
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert list(df["Topic"]) == expected_labels
    assert list(df["Count"]) == expected_counts
