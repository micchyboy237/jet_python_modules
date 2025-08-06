import pytest
from visualization.visualize_embeddings import get_embeddings_and_labels, tag_entities, visualize_embeddings


class TestEntityTaggingAndVisualization:
    def test_tag_entities_basic(self):
        input_text = "Apple is based in California."
        expected = [
            {"text": "Apple", "label": "organization"},
            {"text": "California", "label": "location"}
        ]
        result = tag_entities(input_text)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_get_embeddings_and_labels(self):
        input_text = "Apple is based in California."
        expected_tokens = ["Apple", "is", "based", "in", "California", "."]
        expected_labels = ["organization", "O", "O", "O", "location", "O"]
        embeddings, tokens, labels = get_embeddings_and_labels(input_text)
        assert tokens == expected_tokens, f"Expected tokens {expected_tokens}, but got {tokens}"
        assert labels == expected_labels, f"Expected labels {expected_labels}, but got {labels}"
        assert embeddings.shape[0] == len(
            tokens), f"Expected {len(tokens)} embeddings, but got {embeddings.shape[0]}"

    def test_visualize_embeddings(self):
        input_text = "Apple is based in California."
        expected_entities = [
            {"text": "Apple", "label": "organization"},
            {"text": "California", "label": "location"}
        ]
        chart_config, entities = visualize_embeddings(input_text)
        assert entities == expected_entities, f"Expected entities {expected_entities}, but got {entities}"
        assert chart_config[
            'type'] == 'scatter', f"Expected scatter chart, but got {chart_config['type']}"
        assert len(chart_config['data']['datasets']
                   ) > 0, "Expected non-empty datasets"
