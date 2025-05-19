import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
from jet.llm.embeddings.sentence_embedding import SentenceEmbedding


@pytest.fixture
def sentence_embedding():
    """Fixture to initialize SentenceEmbedding with a small model."""
    return SentenceEmbedding('all-MiniLM-L6-v2')


def test_initialization(sentence_embedding):
    """Test SentenceEmbedding initialization."""
    assert isinstance(sentence_embedding.model, SentenceTransformer)
    assert sentence_embedding.model_name == 'all-MiniLM-L6-v2'
    assert sentence_embedding.tokenizer is not None


def test_generate_embeddings_single_string(sentence_embedding):
    """Test generate_embeddings with a single string."""
    text = "This is a test sentence."
    embedding = sentence_embedding.generate_embeddings(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # Expected dimension for all-MiniLM-L6-v2
    assert all(isinstance(x, float) for x in embedding)


def test_generate_embeddings_list_strings(sentence_embedding):
    """Test generate_embeddings with a list of strings."""
    texts = ["First sentence.", "Second sentence."]
    embeddings = sentence_embedding.generate_embeddings(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(len(emb) == 384 for emb in embeddings)
    assert all(isinstance(x, float) for emb in embeddings for x in emb)


def test_get_embedding_function(sentence_embedding):
    """Test get_embedding_function returns a callable with correct output."""
    embed_fn = sentence_embedding.get_embedding_function()
    text = "Test sentence."
    embedding = embed_fn(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    texts = ["First.", "Second."]
    embeddings = embed_fn(texts)
    assert len(embeddings) == 2
    assert all(len(emb) == 384 for emb in embeddings)


def test_get_token_counts_single(sentence_embedding):
    """Test get_token_counts for a single string."""
    text = "This is a test."
    count = sentence_embedding.get_token_counts(text)
    assert isinstance(count, int)
    assert count > 0  # Should have at least some tokens


def test_get_token_counts_list(sentence_embedding):
    """Test get_token_counts for a list of strings."""
    texts = ["First sentence.", "Second sentence."]
    counts = sentence_embedding.get_token_counts(texts)
    assert isinstance(counts, list)
    assert len(counts) == 2
    assert all(isinstance(c, int) for c in counts)
    assert all(c > 0 for c in counts)


def test_get_token_counts_alt_single(sentence_embedding):
    """Test get_token_counts_alt for a single string."""
    text = "This is a test."
    count = sentence_embedding.get_token_counts_alt(text)
    assert isinstance(count, int)
    assert count > 0


def test_get_token_counts_alt_list(sentence_embedding):
    """Test get_token_counts_alt for a list of strings."""
    texts = ["First sentence.", "Second sentence."]
    counts = sentence_embedding.get_token_counts_alt(texts)
    assert isinstance(counts, list)
    assert len(counts) == 2
    assert all(isinstance(c, int) for c in counts)
    assert all(c > 0 for c in counts)


def test_tokenize_single(sentence_embedding):
    """Test tokenize for a single string."""
    text = "This is a test."
    token_ids = sentence_embedding.tokenize(text)
    assert isinstance(token_ids, list)
    assert all(isinstance(id_, int) for id_ in token_ids)
    assert len(token_ids) > 0


def test_tokenize_list(sentence_embedding):
    """Test tokenize for a list of strings."""
    texts = ["First.", "Second."]
    token_ids = sentence_embedding.tokenize(texts)
    assert isinstance(token_ids, list)
    assert len(token_ids) == 2
    assert all(isinstance(seq, list) for seq in token_ids)
    assert all(isinstance(id_, int) for seq in token_ids for id_ in seq)


def test_tokenize_strings_single(sentence_embedding):
    """Test tokenize_strings for a single string."""
    text = "This is a test."
    tokens = sentence_embedding.tokenize_strings(text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    assert len(tokens) > 0


def test_tokenize_strings_list(sentence_embedding):
    """Test tokenize_strings for a list of strings."""
    texts = ["First.", "Second."]
    tokens = sentence_embedding.tokenize_strings(texts)
    assert isinstance(tokens, list)
    assert len(tokens) == 2
    assert all(isinstance(seq, list) for seq in tokens)
    assert all(isinstance(t, str) for seq in tokens for t in seq)


def test_get_tokenize_fn(sentence_embedding):
    """Test get_tokenize_fn returns a callable with correct output."""
    token_fn = sentence_embedding.get_tokenize_fn()
    text = "Test sentence."
    token_ids = token_fn(text)
    assert isinstance(token_ids, list)
    assert all(isinstance(id_, int) for id_ in token_ids)
    texts = ["First.", "Second."]
    token_ids_list = token_fn(texts)
    assert len(token_ids_list) == 2
    assert all(isinstance(seq, list) for seq in token_ids_list)
    assert all(isinstance(id_, int) for seq in token_ids_list for id_ in seq)
