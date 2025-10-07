import pytest
import numpy as np
from typing import List
from unittest.mock import Mock, patch
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from bertopic import BERTopic
from jet.adapters.bertopic.utils import (
    create_bertopic_model,
    extract_topics_without_query,
    extract_topics_with_query,
    TopicDistribution,
    QueryResult
)


@pytest.fixture
def sample_docs() -> List[str]:
    """Real-world example: Short docs on tech topics (AI, blockchain, cloud)."""
    return [
        "Artificial intelligence is transforming healthcare with machine learning algorithms.",
        "Blockchain technology ensures secure decentralized transactions in finance.",
        "Cloud computing platforms like AWS provide scalable infrastructure for apps.",
        "AI models require large datasets for training deep neural networks.",
        "Blockchain smart contracts automate agreements without intermediaries.",
        "Cloud services enable remote data storage and processing."
    ]


@pytest.fixture
def sample_docs_for_embedding() -> List[str]:
    """Short sentences for embedding tests."""
    return [
        "AI transforms healthcare.",
        "Blockchain secures finance."
    ]


@pytest.fixture
def expected_topics() -> np.ndarray:
    """Expected topic assignments for sample_docs (0=AI, 1=Blockchain, 2=Cloud, -1=outlier)."""
    return np.array([0, 1, 2, 0, 1, 2])


@pytest.fixture
def expected_topics_without_query() -> TopicDistribution:
    """Expected output for sample_docs (mocked for determinism)."""
    return {
        0: [("artificial", 0.8), ("intelligence", 0.7), ("machine", 0.6), ("learning", 0.5)],
        1: [("blockchain", 0.9), ("technology", 0.8), ("secure", 0.7)],
        2: [("cloud", 0.85), ("computing", 0.75), ("platforms", 0.65), ("scalable", 0.55)]
    }


@pytest.fixture
def expected_query_result() -> QueryResult:
    """Expected query result for 'artificial intelligence'."""
    return {
        "topic_ids": [0, 1, 2],
        "probabilities": [0.95, 0.2, 0.15]
    }


@pytest.fixture
def expected_embeddings() -> np.ndarray:
    """Mock embeddings: 2 docs, 384-dim (common for embeddinggemma)."""
    return np.array([
        [0.1] * 384,
        [0.2] * 384
    ], dtype=np.float32)




class TestLlamacppEmbeddingIntegration:
    """BDD-style tests for LlamacppEmbedding with BERTopic."""

    @pytest.fixture
    def llama_embedder(self):
        """Mock LlamacppEmbedding with OpenAI client."""
        with patch("openai.OpenAI") as mock_openai:
            embedder = LlamacppEmbedding(model="embeddinggemma", base_url="http://test:8081/v1")
            embedder.client = mock_openai.return_value
            return embedder

    def test_get_embedding_function_with_bertopic(self, llama_embedder, sample_docs_for_embedding, expected_embeddings):
        # Given: LlamacppEmbedding and sample docs
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 384), Mock(embedding=[0.2] * 384)]
        llama_embedder.client.embeddings.create.return_value = mock_response
        
        # When: Getting embedding function and using with BERTopic
        embedding_fn = llama_embedder.get_embedding_function(return_format="numpy", batch_size=4)
        model = create_bertopic_model(embedding_model=embedding_fn, min_topic_size=2)
        
        with patch.object(model, "fit_transform", return_value=(np.array([0, 1]), None)):
            topics, _ = model.fit_transform(sample_docs_for_embedding)
        
        # Then: Callable works, topics assigned
        np.testing.assert_array_equal(topics, [0, 1])
        llama_embedder.client.embeddings.create.assert_called_once_with(
            model="embeddinggemma",
            input=sample_docs_for_embedding
        )
        result = embedding_fn(sample_docs_for_embedding)
        np.testing.assert_array_equal(result, expected_embeddings)

    def test_streaming_embeddings_with_bertopic(self, llama_embedder, sample_docs_for_embedding, expected_embeddings):
        # Given: Streaming setup
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 384), Mock(embedding=[0.2] * 384)]
        llama_embedder.client.embeddings.create.return_value = mock_response
        
        # When: Streaming embeddings and feeding to BERTopic
        stream = llama_embedder.get_embeddings_stream(
            inputs=sample_docs_for_embedding,
            return_format="numpy",
            batch_size=1,
            show_progress=False
        )
        embeddings = np.vstack(list(stream))
        model = create_bertopic_model(min_topic_size=2)
        
        with patch.object(model, "fit_transform", return_value=(np.array([0, 1]), None)):
            topics, _ = model.fit_transform(sample_docs_for_embedding, embeddings=embeddings)
        
        # Then: Streamed embeddings work
        np.testing.assert_array_equal(topics, [0, 1])
        np.testing.assert_array_equal(embeddings, expected_embeddings)


class TestCreateBERTopicModel:
    """BDD-style tests for model creation."""

    def test_creates_model_with_defaults(self):
        # Given: Default parameters
        embedding_model = "all-MiniLM-L6-v2"
        
        # When: Creating the model
        model = create_bertopic_model(embedding_model=embedding_model)
        
        # Then: Model is instance of BERTopic with expected defaults
        assert isinstance(model, BERTopic)
        assert model.embedding_model._first_module_name == embedding_model
        assert model.min_topic_size == 10

    def test_creates_model_with_callable_embedding(self):
        # Given: Callable embedding_model
        mock_callable = Mock(return_value=np.empty((5, 384)))
        
        # When: Creating
        model = create_bertopic_model(embedding_model=mock_callable)
        
        # Then: BERTopic accepts callable
        assert isinstance(model, BERTopic)
        assert model.embedding_model == mock_callable

    @patch('bertopic.BERTopic')
    def test_passes_kwargs_to_constructor(self, mock_bertopic):
        # Given: Custom kwargs
        kwargs = {"top_k_words": 15, "verbose": True}
        
        # When: Creating with kwargs
        create_bertopic_model(embedding_model="test-model", **kwargs)
        
        # Then: Kwargs are passed to BERTopic
        mock_bertopic.assert_called_once_with(
            embedding_model=Mock(_first_module_name="test-model"),
            top_k_words=15,
            verbose=True,
            min_topic_size=10,
            nr_topics=None
        )

    def test_raises_error_on_invalid_embedding(self):
        # Given: Invalid embedding model
        
        # When/Then: Raises ValueError
        with pytest.raises(ValueError):
            create_bertopic_model(embedding_model="invalid-model")


class TestExtractTopicsWithoutQuery:
    """BDD-style tests for unsupervised topic extraction."""

    def test_extracts_topics_from_valid_docs(self, sample_docs, expected_topics, expected_topics_without_query):
        # Given: Valid sample documents
        
        # When: Extracting topics (mock for determinism)
        with patch('bertopic_utils.BERTopic.fit_transform') as mock_fit, \
             patch('bertopic_utils.BERTopic.get_topics', return_value=expected_topics_without_query):
            mock_fit.return_value = (expected_topics, None)
            topics, topic_info = extract_topics_without_query(sample_docs)
        
        # Then: Returns expected topics array and dict
        np.testing.assert_array_equal(topics, expected_topics)
        assert topic_info == expected_topics_without_query
        assert len(topic_info) > 0

    def test_extracts_with_callable_embedding(self, sample_docs, expected_topics, expected_topics_without_query):
        # Given: Callable embedder
        mock_embedder = Mock(return_value=np.random.rand(len(sample_docs), 384))
        mock_model = Mock()
        mock_model.fit_transform.return_value = (expected_topics, None)
        mock_model.get_topics.return_value = expected_topics_without_query
        
        with patch("bertopic_utils.create_bertopic_model", return_value=mock_model):
            topics, topic_info = extract_topics_without_query(
                sample_docs, embedding_model=mock_embedder
            )
        
        # Then: Uses callable
        np.testing.assert_array_equal(topics, expected_topics)
        assert topic_info == expected_topics_without_query
        mock_model.fit_transform.assert_called_once()

    def test_raises_error_on_empty_docs(self):
        # Given: Empty documents list
        
        # When/Then: Raises ValueError
        with pytest.raises(ValueError, match="Documents list cannot be empty."):
            extract_topics_without_query([])

    @patch('bertopic_utils.create_bertopic_model')
    def test_uses_custom_params(self, mock_create):
        # Given: Custom params
        docs = ["test doc"]
        mock_model = Mock()
        mock_model.fit_transform.return_value = (np.array([0]), None)
        mock_model.get_topics.return_value = {0: [("word", 0.5)]}
        mock_create.return_value = mock_model
        
        # When: Extracting with custom nr_topics
        extract_topics_without_query(docs, nr_topics=5, min_topic_size=5)
        
        # Then: Params passed to model creation
        mock_create.assert_called_once_with(
            embedding_model="all-MiniLM-L6-v2",
            nr_topics=5,
            min_topic_size=5
        )


class TestExtractTopicsWithQuery:
    """BDD-style tests for query-based topic extraction."""

    def test_finds_topics_for_valid_query(self, sample_docs, expected_topics, expected_query_result):
        # Given: Valid docs and query
        query = "artificial intelligence"
        
        # When: Extracting with query (mock for determinism)
        with patch('bertopic_utils.BERTopic.fit_transform') as mock_fit, \
             patch('bertopic_utils.BERTopic.find_topics', return_value=(expected_query_result["topic_ids"], expected_query_result["probabilities"])):
            mock_fit.return_value = (expected_topics, None)
            topics, query_result = extract_topics_with_query(sample_docs, query)
        
        # Then: Returns expected topics and query result
        np.testing.assert_array_equal(topics, expected_topics)
        assert query_result == expected_query_result

    def test_finds_topics_with_callable_embedding(self, sample_docs, expected_topics, expected_query_result):
        # Given: Callable
        mock_embedder = Mock(return_value=np.random.rand(len(sample_docs), 384))
        query = "artificial intelligence"
        mock_model = Mock()
        mock_model.fit_transform.return_value = (expected_topics, None)
        mock_model.find_topics.return_value = (expected_query_result["topic_ids"], expected_query_result["probabilities"])
        
        with patch("bertopic_utils.create_bertopic_model", return_value=mock_model):
            topics, query_result = extract_topics_with_query(
                sample_docs, query, embedding_model=mock_embedder
            )
        
        # Then: Uses callable
        np.testing.assert_array_equal(topics, expected_topics)
        assert query_result == expected_query_result

    def test_raises_error_on_empty_docs(self):
        # Given: Empty docs
        
        # When/Then: Raises ValueError
        with pytest.raises(ValueError, match="Documents list cannot be empty."):
            extract_topics_with_query([], "test query")

    def test_raises_error_on_empty_query(self, sample_docs):
        # Given: Empty query
        
        # When/Then: Raises ValueError
        with pytest.raises(ValueError, match="Query cannot be empty."):
            extract_topics_with_query(sample_docs, "")

    @patch('bertopic_utils.create_bertopic_model')
    def test_uses_top_k_param(self, mock_create):
        # Given: Custom top_k
        docs = ["test"]
        query = "test query"
        mock_model = Mock()
        mock_model.fit_transform.return_value = (np.array([0]), None)
        mock_model.find_topics.return_value = ([0], [0.5])
        mock_create.return_value = mock_model
        
        # When: Extracting with top_k=3
        extract_topics_with_query(docs, query, top_k=3)
        
        # Then: top_k passed to find_topics
        mock_model.find_topics.assert_called_once_with(query, top_k=3)