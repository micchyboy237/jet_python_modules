import pytest
from typing import List, Tuple, Union
import spacy
from keybert import KeyBERT
from jet.wordnet.keywords.keyword_extraction import setup_keybert, extract_query_candidates, extract_keywords_with_embeddings, KeywordResult


class TestExtractQueryCandidates:
    """Test suite for extract_query_candidates function."""
    @pytest.fixture
    def nlp(self):
        """Fixture to initialize spaCy model."""
        return spacy.load("en_core_web_sm")

    def test_basic_query_extraction(self, nlp):
        """Test candidate extraction from a basic query."""
        query = "Artificial intelligence and machine learning in healthcare"
        result_candidates = extract_query_candidates(query, nlp)
        expected_candidates = [
            "artificial intelligence",
            "machine learning",
            "healthcare",
            "intelligence",
            "learning"
        ]
        assert sorted(result_candidates) == sorted(expected_candidates), (
            f"Expected {expected_candidates}, got {result_candidates}"
        )

    def test_empty_query(self, nlp):
        """Test candidate extraction from an empty query."""
        query = ""
        result_candidates = extract_query_candidates(query, nlp)
        expected_candidates: List[str] = []
        assert result_candidates == expected_candidates, (
            f"Expected {expected_candidates}, got {result_candidates}"
        )

    def test_stop_words_only_query(self, nlp):
        """Test candidate extraction from a query with only stop words."""
        query = "the and is are"
        result_candidates = extract_query_candidates(query, nlp)
        expected_candidates: List[str] = []
        assert result_candidates == expected_candidates, (
            f"Expected {expected_candidates}, got {result_candidates}"
        )

    def test_multi_word_phrase_extraction(self, nlp):
        """Test candidate extraction with multi-word phrases."""
        query = "Renewable energy sources and climate change mitigation"
        result_candidates = extract_query_candidates(query, nlp)
        expected_candidates = [
            "renewable energy sources",
            "climate change mitigation",
            "energy sources",
            "climate change",
            "mitigation",
            "sources",
            "energy",
            "climate",
            "change"
        ]
        assert sorted(result_candidates) == sorted(expected_candidates), (
            f"Expected {expected_candidates}, got {result_candidates}"
        )

    def test_case_insensitivity(self, nlp):
        """Test candidate extraction with mixed case query."""
        query = "MACHINE Learning AND Artificial INTELLIGENCE"
        result_candidates = extract_query_candidates(query, nlp)
        expected_candidates = [
            "machine learning",
            "artificial intelligence",
            "learning",
            "intelligence"
        ]
        assert sorted(result_candidates) == sorted(expected_candidates), (
            f"Expected {expected_candidates}, got {result_candidates}"
        )


class TestExtractKeywordsWithEmbeddings:
    """Test suite for extract_keywords_with_embeddings function."""
    @pytest.fixture
    def model(self) -> KeyBERT:
        """Fixture to initialize KeyBERT model."""
        return setup_keybert(model_name="static-retrieval-mrl-en-v1")

    @pytest.fixture
    def sample_docs(self) -> List[str]:
        """Fixture for sample documents."""
        return [
            "Renewable energy sources like solar are key to sustainability.",
            "Blockchain ensures secure transactions in cryptocurrencies."
        ]

    @pytest.fixture
    def single_doc(self) -> str:
        """Fixture for a single sample document."""
        return "Renewable energy sources like solar are key to sustainability."

    def test_extract_keywords_with_embeddings(self, model: KeyBERT, sample_docs: List[str]):
        """Test keyword extraction with embeddings for multiple documents."""
        # Given: A KeyBERT model and sample documents
        expected_num_docs = 2
        expected_num_keywords = 3
        expected_device = "cpu"
        # When: Extracting keywords with embeddings
        result_keywords = extract_keywords_with_embeddings(
            sample_docs, model, top_n=3)
        result_device = str(model.model.embedding_model.device)
        # Then: Verify the number of documents, keywords per document, and their properties
        assert len(
            result_keywords) == expected_num_docs, f"Expected {expected_num_docs} documents, got {len(result_keywords)}"
        for i, res_kws in enumerate(result_keywords):
            assert len(
                res_kws) == expected_num_keywords, f"Doc {i+1}: Expected {expected_num_keywords} keywords, got {len(res_kws)}"
            for kw in res_kws:
                assert isinstance(
                    kw["text"], str), f"Doc {i+1}: Keyword {kw['text']} must be a string"
                assert 0 <= kw["score"] <= 1, f"Doc {i+1}: Score {kw['score']} for {kw['text']} must be between 0 and 1"
                assert kw["doc_index"] == i, f"Doc {i+1}: Expected doc_index {i}, got {kw['doc_index']}"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"

    def test_extract_keywords_with_embeddings_multi_doc(self, model: KeyBERT, sample_docs: List[str]):
        """Test keyword extraction with embeddings for multiple documents with n-gram range."""
        # Given: A KeyBERT model and sample documents with n-gram range
        expected_num_docs = 2
        expected_num_keywords = 3
        expected_device = "cpu"
        # When: Extracting keywords with embeddings and n-gram range
        result_keywords = extract_keywords_with_embeddings(
            sample_docs, model, top_n=3, keyphrase_ngram_range=(1, 2))
        result_device = str(model.model.embedding_model.device)
        # Then: Verify the number of documents, keywords per document, and their properties
        assert len(
            result_keywords) == expected_num_docs, f"Expected {expected_num_docs} documents, got {len(result_keywords)}"
        for i, res_kws in enumerate(result_keywords):
            assert len(
                res_kws) == expected_num_keywords, f"Doc {i+1}: Expected {expected_num_keywords} keywords, got {len(res_kws)}"
            for kw in res_kws:
                assert isinstance(
                    kw["text"], str), f"Doc {i+1}: Keyword {kw['text']} must be a string"
                assert 0 <= kw["score"] <= 1, f"Doc {i+1}: Score {kw['score']} for {kw['text']} must be between 0 and 1"
                assert kw["doc_index"] == i, f"Doc {i+1}: Expected doc_index {i}, got {kw['doc_index']}"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"

    def test_extract_keywords_with_embeddings_single_doc(self, model: KeyBERT, single_doc: str):
        """Test keyword extraction with embeddings for a single document."""
        # Given: A KeyBERT model and a single document
        expected_num_keywords = 3
        expected_device = "cpu"
        # When: Extracting keywords with embeddings
        result_keywords = extract_keywords_with_embeddings(
            single_doc, model, top_n=3)
        result_device = str(model.model.embedding_model.device)
        # Then: Verify the number of keywords and their properties
        assert len(
            result_keywords) == expected_num_keywords, f"Expected {expected_num_keywords} keywords, got {len(result_keywords)}"
        for kw in result_keywords:
            assert isinstance(
                kw["text"], str), f"Keyword {kw['text']} must be a string"
            assert 0 <= kw["score"] <= 1, f"Score {kw['score']} for {kw['text']} must be between 0 and 1"
            assert kw["doc_index"] == 0, f"Expected doc_index 0, got {kw['doc_index']}"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"

    def test_extract_keywords_with_embeddings_empty_input(self, model: KeyBERT):
        """Test keyword extraction with empty input."""
        # Given: A KeyBERT model and an empty input
        expected_keywords: List[List[KeywordResult]] = []
        # When: Extracting keywords with embeddings
        result_keywords = extract_keywords_with_embeddings([], model, top_n=3)
        # Then: Verify the result is an empty list
        assert result_keywords == expected_keywords, f"Expected {expected_keywords}, got {result_keywords}"

    def test_extract_keywords_with_embeddings_invalid_input(self, model: KeyBERT):
        """Test keyword extraction with invalid input."""
        # Given: A KeyBERT model and invalid input
        invalid_docs = [1, 2, 3]
        # When: Extracting keywords with embeddings
        # Then: Expect a ValueError
        with pytest.raises(ValueError, match="All elements in docs must be strings"):
            extract_keywords_with_embeddings(invalid_docs, model, top_n=3)

    def test_extract_keywords_with_embeddings_top_n_variation(self, model: KeyBERT, sample_docs: List[str]):
        """Test keyword extraction with varying top_n values."""
        # Given: A KeyBERT model and sample documents
        expected_num_docs = 2
        expected_max_keywords = 5
        # When: Extracting keywords with embeddings and top_n=5
        result_keywords = extract_keywords_with_embeddings(
            sample_docs, model, top_n=5)
        # Then: Verify the number of documents, keywords per document, and their properties
        assert len(
            result_keywords) == expected_num_docs, f"Expected {expected_num_docs} documents, got {len(result_keywords)}"
        for i, res_kws in enumerate(result_keywords):
            assert len(
                res_kws) <= expected_max_keywords, f"Doc {i+1}: Expected up to {expected_max_keywords} keywords, got {len(res_kws)}"
            for kw in res_kws:
                assert isinstance(
                    kw["text"], str), f"Doc {i+1}: Keyword {kw['text']} must be a string"
                assert 0 <= kw["score"] <= 1, f"Doc {i+1}: Score {kw['score']} for {kw['text']} must be between 0 and 1"
                assert kw["doc_index"] == i, f"Doc {i+1}: Expected doc_index {i}, got {kw['doc_index']}"


class TestExtractKeywordsWithDiverseDocuments:
    """Test suite for extract_keywords_with_embeddings with diverse real-world documents."""
    @pytest.fixture
    def model(self) -> KeyBERT:
        """Fixture to initialize KeyBERT model."""
        return setup_keybert(model_name="static-retrieval-mrl-en-v1")

    @pytest.fixture
    def short_doc(self) -> str:
        """Fixture for a short real-world document (news snippet)."""
        return (
            "NASA's Perseverance rover successfully landed on Mars, beginning its mission to "
            "search for signs of ancient life and collect rock samples."
        )

    @pytest.fixture
    def medium_doc(self) -> str:
        """Fixture for a medium-length real-world document (tech article excerpt)."""
        return (
            "Artificial intelligence is transforming industries from healthcare to finance. "
            "Machine learning models, trained on vast datasets, enable predictive analytics "
            "and automation. Companies like Google and Amazon leverage AI to optimize operations, "
            "while startups innovate with specialized algorithms. However, ethical concerns around "
            "bias and privacy remain critical challenges for the widespread adoption of AI technologies."
        )

    @pytest.fixture
    def long_doc(self) -> str:
        """Fixture for a long real-world document (science report summary)."""
        return (
            "Climate change poses significant risks to global ecosystems and human societies. "
            "Rising temperatures, driven by greenhouse gas emissions, are causing more frequent "
            "and intense heatwaves, droughts, and wildfires. Coastal regions face threats from "
            "sea-level rise, with projections estimating a 0.3 to 1.2-meter increase by 2100. "
            "Biodiversity loss is accelerating, with species extinction rates 100 times higher than "
            "natural baselines. Mitigation strategies include transitioning to renewable energy, "
            "enhancing carbon capture technologies, and implementing sustainable land-use practices. "
            "International cooperation, such as the Paris Agreement, aims to limit warming to 1.5°C, "
            "but current commitments fall short. Adaptation measures, like building resilient infrastructure "
            "and improving early warning systems, are equally critical to reduce vulnerability. "
            "Addressing climate change requires urgent, collective action across governments, industries, "
            "and communities to safeguard the planet for future generations."
        )

    @pytest.fixture
    def multi_docs(self) -> List[str]:
        """Fixture for multiple diverse real-world documents."""
        return [
            "Tesla's stock surged after announcing record-breaking electric vehicle deliveries in Q4.",
            "Vaccines remain a cornerstone of public health, reducing the spread of infectious diseases. "
            "mRNA technology, used in COVID-19 vaccines, has opened new possibilities for rapid vaccine development.",
            "Urbanization is reshaping global demographics, with 68% of the world’s population expected "
            "to live in cities by 2050. This trend offers opportunities for economic growth but strains "
            "infrastructure, housing, and environmental resources. Sustainable urban planning, including "
            "green spaces and efficient public transport, is essential to balance development and livability."
        ]

    def test_short_doc_extraction(self, model: KeyBERT, short_doc: str):
        """Test keyword extraction for a short real-world document."""
        # Given: A KeyBERT model and a short document
        expected_num_keywords = 3
        expected_device = "cpu"
        # When: Extracting keywords with embeddings
        result_keywords = extract_keywords_with_embeddings(
            short_doc, model, top_n=3, keyphrase_ngram_range=(1, 2))
        result_device = str(model.model.embedding_model.device)
        # Then: Verify the number of keywords and their properties
        assert len(
            result_keywords) == expected_num_keywords, f"Expected {expected_num_keywords} keywords, got {len(result_keywords)}"
        for kw in result_keywords:
            assert isinstance(
                kw["text"], str), f"Keyword {kw['text']} must be a string"
            assert 0 <= kw["score"] <= 1, f"Score {kw['score']} for {kw['text']} must be between 0 and 1"
            assert kw["doc_index"] == 0, f"Expected doc_index 0, got {kw['doc_index']}"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"

    def test_medium_doc_extraction(self, model: KeyBERT, medium_doc: str):
        """Test keyword extraction for a medium-length real-world document."""
        # Given: A KeyBERT model and a medium-length document
        expected_num_keywords = 5
        expected_device = "cpu"
        # When: Extracting keywords with embeddings
        result_keywords = extract_keywords_with_embeddings(
            medium_doc, model, top_n=5, keyphrase_ngram_range=(1, 2))
        result_device = str(model.model.embedding_model.device)
        # Then: Verify the number of keywords and their properties
        assert len(
            result_keywords) == expected_num_keywords, f"Expected {expected_num_keywords} keywords, got {len(result_keywords)}"
        for kw in result_keywords:
            assert isinstance(
                kw["text"], str), f"Keyword {kw['text']} must be a string"
            assert 0 <= kw["score"] <= 1, f"Score {kw['score']} for {kw['text']} must be between 0 and 1"
            assert kw["doc_index"] == 0, f"Expected doc_index 0, got {kw['doc_index']}"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"

    def test_long_doc_extraction(self, model: KeyBERT, long_doc: str):
        """Test keyword extraction for a long real-world document."""
        # Given: A KeyBERT model and a long document
        expected_num_keywords = 7
        expected_device = "cpu"
        # When: Extracting keywords with embeddings
        result_keywords = extract_keywords_with_embeddings(
            long_doc, model, top_n=7, keyphrase_ngram_range=(1, 2))
        result_device = str(model.model.embedding_model.device)
        # Then: Verify the number of keywords and their properties
        assert len(
            result_keywords) == expected_num_keywords, f"Expected {expected_num_keywords} keywords, got {len(result_keywords)}"
        for kw in result_keywords:
            assert isinstance(
                kw["text"], str), f"Keyword {kw['text']} must be a string"
            assert 0 <= kw["score"] <= 1, f"Score {kw['score']} for {kw['text']} must be between 0 and 1"
            assert kw["doc_index"] == 0, f"Expected doc_index 0, got {kw['doc_index']}"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"

    def test_multi_doc_extraction(self, model: KeyBERT, multi_docs: List[str]):
        """Test keyword extraction for multiple diverse real-world documents."""
        # Given: A KeyBERT model and multiple documents
        expected_num_docs = 3
        expected_num_keywords = 4
        expected_device = "cpu"
        # When: Extracting keywords with embeddings
        result_keywords = extract_keywords_with_embeddings(
            multi_docs, model, top_n=4, keyphrase_ngram_range=(1, 2))
        result_device = str(model.model.embedding_model.device)
        # Then: Verify the number of documents, keywords per document, and their properties
        assert len(
            result_keywords) == expected_num_docs, f"Expected {expected_num_docs} documents, got {len(result_keywords)}"
        for i, res_kws in enumerate(result_keywords):
            assert len(
                res_kws) == expected_num_keywords, f"Doc {i+1}: Expected {expected_num_keywords} keywords, got {len(res_kws)}"
            for kw in res_kws:
                assert isinstance(
                    kw["text"], str), f"Doc {i+1}: Keyword {kw['text']} must be a string"
                assert 0 <= kw["score"] <= 1, f"Doc {i+1}: Score {kw['score']} for {kw['text']} must be between 0 and 1"
                assert kw["doc_index"] == i, f"Doc {i+1}: Expected doc_index {i}, got {kw['doc_index']}"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"

    def test_large_top_n_extraction(self, model: KeyBERT, long_doc: str):
        """Test keyword extraction with a large top_n value for a long document."""
        # Given: A KeyBERT model and a long document
        expected_max_keywords = 10
        expected_device = "cpu"
        # When: Extracting keywords with embeddings
        result_keywords = extract_keywords_with_embeddings(
            long_doc, model, top_n=10, keyphrase_ngram_range=(1, 2))
        result_device = str(model.model.embedding_model.device)
        # Then: Verify the number of keywords and their properties
        assert len(
            result_keywords) <= expected_max_keywords, f"Expected up to {expected_max_keywords} keywords, got {len(result_keywords)}"
        for kw in result_keywords:
            assert isinstance(
                kw["text"], str), f"Keyword {kw['text']} must be a string"
            assert 0 <= kw["score"] <= 1, f"Score {kw['score']} for {kw['text']} must be between 0 and 1"
            assert kw["doc_index"] == 0, f"Expected doc_index 0, got {kw['doc_index']}"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"

    def test_small_top_n_extraction(self, model: KeyBERT, medium_doc: str):
        """Test keyword extraction with a small top_n value for a medium document."""
        # Given: A KeyBERT model and a medium-length document
        expected_max_keywords = 2
        expected_device = "cpu"
        # When: Extracting keywords with embeddings
        result_keywords = extract_keywords_with_embeddings(
            medium_doc, model, top_n=2, keyphrase_ngram_range=(1, 2))
        result_device = str(model.model.embedding_model.device)
        # Then: Verify the number of keywords and their properties
        assert len(
            result_keywords) <= expected_max_keywords, f"Expected up to {expected_max_keywords} keywords, got {len(result_keywords)}"
        for kw in result_keywords:
            assert isinstance(
                kw["text"], str), f"Keyword {kw['text']} must be a string"
            assert 0 <= kw["score"] <= 1, f"Score {kw['score']} for {kw['text']} must be between 0 and 1"
            assert kw["doc_index"] == 0, f"Expected doc_index 0, got {kw['doc_index']}"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"

    def test_single_word_ngram_extraction(self, model: KeyBERT, long_doc: str):
        """Test keyword extraction with single-word n-gram range for a long document."""
        # Given: A KeyBERT model and a long document
        expected_num_keywords = 5
        expected_device = "cpu"
        # When: Extracting keywords with embeddings
        result_keywords = extract_keywords_with_embeddings(
            long_doc, model, top_n=5, keyphrase_ngram_range=(1, 1))
        result_device = str(model.model.embedding_model.device)
        # Then: Verify the number of keywords and their properties
        assert len(
            result_keywords) == expected_num_keywords, f"Expected {expected_num_keywords} keywords, got {len(result_keywords)}"
        for kw in result_keywords:
            assert isinstance(
                kw["text"], str), f"Keyword {kw['text']} must be a string"
            assert 0 <= kw["score"] <= 1, f"Score {kw['score']} for {kw['text']} must be between 0 and 1"
            assert kw["doc_index"] == 0, f"Expected doc_index 0, got {kw['doc_index']}"
            assert len(kw["text"].split(
            )) == 1, f"Keyword {kw['text']} must be a single word for ngram_range=(1, 1)"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"

    def test_trigram_ngram_extraction(self, model: KeyBERT, multi_docs: List[str]):
        """Test keyword extraction with trigram n-gram range for multiple documents."""
        # Given: A KeyBERT model and multiple documents
        expected_num_docs = 3
        expected_num_keywords = 3
        expected_device = "cpu"
        # When: Extracting keywords with embeddings
        result_keywords = extract_keywords_with_embeddings(
            multi_docs, model, top_n=3, keyphrase_ngram_range=(1, 3))
        result_device = str(model.model.embedding_model.device)
        # Then: Verify the number of documents, keywords per document, and their properties
        assert len(
            result_keywords) == expected_num_docs, f"Expected {expected_num_docs} documents, got {len(result_keywords)}"
        for i, res_kws in enumerate(result_keywords):
            assert len(
                res_kws) == expected_num_keywords, f"Doc {i+1}: Expected {expected_num_keywords} keywords, got {len(res_kws)}"
            for kw in res_kws:
                assert isinstance(
                    kw["text"], str), f"Doc {i+1}: Keyword {kw['text']} must be a string"
                assert 0 <= kw["score"] <= 1, f"Doc {i+1}: Score {kw['score']} for {kw['text']} must be between 0 and 1"
                assert kw["doc_index"] == i, f"Doc {i+1}: Expected doc_index {i}, got {kw['doc_index']}"
                assert len(kw["text"].split(
                )) <= 3, f"Doc {i+1}: Keyword {kw['text']} must have at most 3 words for ngram_range=(1, 3)"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"
