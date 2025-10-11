import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from unittest.mock import patch, MagicMock
from jet.libs.bertopic.examples.base.helpers.bertopic_enhancer import BERTopicEnhancer, MODEL2VEC_AVAILABLE

@pytest.fixture
def enhancer():
    """Fixture for BERTopicEnhancer with default configuration."""
    enhancer = BERTopicEnhancer()
    yield enhancer
    # Cleanup: No persistent state to clean up

@pytest.fixture
def sample_docs():
    """Fixture for sample documents."""
    return fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data'][:10]

@pytest.fixture
def mock_multimodal_data(tmp_path):
    """Fixture for mock image paths and captions."""
    img_dir = tmp_path / "photos" / "Flicker8k_Dataset"
    img_dir.mkdir(parents=True)
    images = [str(img_dir / f"img_{i}.jpg") for i in range(5)]
    captions = [f"Caption for image {i}" for i in range(5)]
    return images, captions

class TestBERTopicEnhancerStopwords:
    """Tests for stopword removal functionality."""

    def test_remove_stopwords_countvectorizer(self, enhancer):
        """Given a BERTopicEnhancer, when stopword removal is configured, then CountVectorizer is set with English stopwords."""
        # Given
        expected_stop_words = "english"

        # When
        enhancer.remove_stopwords_countvectorizer()

        # Then
        result = enhancer.vectorizer_model.stop_words
        assert result == expected_stop_words, f"Expected stop_words={expected_stop_words}, got {result}"

class TestBERTopicEnhancerTopicMatrix:
    """Tests for topic-term matrix extraction."""

    def test_get_topic_term_matrix(self, enhancer, sample_docs):
        """Given a fitted BERTopic model, when topic-term matrix is extracted, then it returns a matrix and feature names."""
        # Given
        enhancer.topic_model.fit(sample_docs)
        expected_words_type = list
        expected_matrix_type = np.ndarray

        # When
        topic_term_matrix, words = enhancer.get_topic_term_matrix()

        # Then
        assert isinstance(words, expected_words_type), f"Expected words to be {expected_words_type}, got {type(words)}"
        assert isinstance(topic_term_matrix, expected_matrix_type), f"Expected matrix to be {expected_matrix_type}, got {type(topic_term_matrix)}"
        assert len(words) > 0, "Expected non-empty feature names"
        assert topic_term_matrix.shape[0] > 0, "Expected non-empty topic-term matrix"

class TestBERTopicEnhancerPrecomputeEmbeddings:
    """Tests for pre-computing embeddings."""

    def test_precompute_embeddings(self, enhancer, sample_docs):
        """Given documents, when embeddings are pre-computed, then a numpy array of correct shape is returned."""
        # Given
        expected_shape = (len(sample_docs), 384)  # all-MiniLM-L6-v2 has 384 dimensions

        # When
        embeddings = enhancer.precompute_embeddings(sample_docs)

        # Then
        result_shape = embeddings.shape
        assert isinstance(embeddings, np.ndarray), f"Expected embeddings to be np.ndarray, got {type(embeddings)}"
        assert result_shape == expected_shape, f"Expected shape {expected_shape}, got {result_shape}"

class TestBERTopicEnhancerDocumentDistribution:
    """Tests for document distribution approximation."""

    def test_approximate_document_distribution(self, enhancer, sample_docs):
        """Given a fitted model and documents, when distributions are approximated, then correct shapes are returned."""
        # Given
        enhancer.topic_model.fit(sample_docs)
        expected_distr_shape = (len(sample_docs), len(enhancer.topic_model.get_topics()) + 1)  # +1 for -1 topic

        # When
        topic_distr, topic_token_distr = enhancer.approximate_document_distribution(sample_docs, calculate_tokens=True)

        # Then
        assert isinstance(topic_distr, np.ndarray), f"Expected topic_distr to be np.ndarray, got {type(topic_distr)}"
        assert topic_distr.shape == expected_distr_shape, f"Expected topic_distr shape {expected_distr_shape}, got {topic_distr.shape}"
        assert isinstance(topic_token_distr, np.ndarray), f"Expected topic_token_distr to be np.ndarray, got {type(topic_token_distr)}"

class TestBERTopicEnhancerCompareModels:
    """Tests for comparing topic models."""

    def test_compare_topic_models(self, enhancer, sample_docs):
        """Given two fitted models, when topic models are compared, then a similarity matrix is returned."""
        # Given
        other_enhancer = BERTopicEnhancer(embedding_model=SentenceTransformer("all-MiniLM-L6-v2"))
        enhancer.topic_model.fit(sample_docs)
        other_enhancer.topic_model.fit(sample_docs)
        expected_shape = (
            len(enhancer.topic_model.get_topics()) + 1,
            len(other_enhancer.topic_model.get_topics()) + 1
        )

        # When
        sim_matrix = enhancer.compare_topic_models(other_enhancer.topic_model)

        # Then
        assert isinstance(sim_matrix, np.ndarray), f"Expected sim_matrix to be np.ndarray, got {type(sim_matrix)}"
        assert sim_matrix.shape == expected_shape, f"Expected shape {expected_shape}, got {sim_matrix.shape}"
        assert np.all((sim_matrix >= -1) & (sim_matrix <= 1)), "Cosine similarity values should be between -1 and 1"

class TestBERTopicEnhancerLightweight:
    """Tests for lightweight embedding with Model2Vec."""

    @pytest.mark.skipif(not MODEL2VEC_AVAILABLE, reason="model2vec not installed")
    def test_enable_lightweight_embedding(self, enhancer):
        """Given a BERTopicEnhancer, when lightweight embedding is enabled, then Model2Vec is set as embedding model."""
        # Given
        from model2vec import StaticModel
        expected_model_type = StaticModel

        # When
        enhancer.enable_lightweight_embedding()

        # Then
        result = enhancer.embedding_model
        assert isinstance(result, expected_model_type), f"Expected embedding_model to be {expected_model_type}, got {type(result)}"

class TestBERTopicEnhancerMultimodal:
    """Tests for multimodal data processing."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_process_multimodal_data(self, mock_sentence_transformer, mock_multimodal_data):
        """Given images and captions, when multimodal data is processed, then topics, probs, and DataFrame are returned."""
        # Given
        images, captions = mock_multimodal_data
        expected_num_samples = len(images)
        expected_columns = ["img_id", "img_caption", "Topic"]
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(5, 512).astype(np.float32)  # Mock CLIP embeddings
        mock_sentence_transformer.return_value = mock_model
        enhancer = BERTopicEnhancer()

        # When
        topics, probs, df = enhancer.process_multimodal_data(images, captions, batch_size=2)

        # Then
        assert len(topics) == expected_num_samples, f"Expected {expected_num_samples} topics, got {len(topics)}"
        assert probs.shape == (expected_num_samples,), f"Expected probs shape ({expected_num_samples},), got {probs.shape}"
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
        assert list(df.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(df.columns)}"
        assert len(df) == expected_num_samples, f"Expected DataFrame length {expected_num_samples}, got {len(df)}"