import pytest
from jet.wordnet.word_cooccurrence import find_cooccurring_words, tokenize_document


class TestTokenizeDocument:
    def test_tokenize_document_with_pos_filtering(self):
        """Given a simple sentence, when tokenizing with default n-gram range, then return single words."""
        document = "The quick fox jumps over the lazy dog."
        expected = ["quick", "fox", "jumps", "lazy", "dog"]
        result = tokenize_document(document, ngram_range=(1, 1))
        assert result == expected

    def test_tokenize_document_with_bigrams(self):
        """Given a sentence, when tokenizing with bigram range, then return single words and bigrams."""
        document = "The quick fox jumps."
        expected = ["quick", "fox", "jumps", "quick fox", "fox jumps"]
        result = tokenize_document(document, ngram_range=(1, 2))
        assert result == expected


class TestFindCooccurringWords:
    def test_words_cooccurring_in_multiple_documents_with_pos(self):
        """Given documents with overlapping words, when finding co-occurrences, then return word pairs meeting min_docs."""
        documents = [
            "The quick fox jumps over hills",
            "The fox runs fast on hills",
            "A quick dog jumps high"
        ]
        expected = [
            ("fox", "hills", 2),
            ("jumps", "quick", 2)
        ]
        result = find_cooccurring_words(
            documents, min_docs=2, ngram_range=(1, 1))
        assert result == expected

    def test_ngrams_cooccurring_in_multiple_documents(self):
        """Given documents with overlapping phrases, when finding co-occurrences with n-grams, then return word and phrase pairs."""
        documents = [
            "The quick fox jumps over hills",
            "The quick fox runs fast on hills",
            "A quick dog jumps high"
        ]
        expected = [
            ("quick fox", "hills", 2),
            ("fox", "hills", 2),
            ("jumps", "quick", 2)
        ]
        result = find_cooccurring_words(
            documents, min_docs=2, ngram_range=(1, 2))
        assert result == expected

    def test_no_pairs_meet_min_docs_threshold(self):
        """Given documents with no frequent pairs, when finding co-occurrences, then return empty list."""
        documents = [
            "The quick fox jumps",
            "The slow dog runs",
            "Fast cat sleeps"
        ]
        expected = []
        result = find_cooccurring_words(
            documents, min_docs=3, ngram_range=(1, 2))
        assert result == expected

    def test_empty_documents_list(self):
        """Given an empty document list, when finding co-occurrences, then return empty list."""
        documents = []
        expected = []
        result = find_cooccurring_words(documents, ngram_range=(1, 2))
        assert result == expected

    def test_single_document_with_pos(self):
        """Given a single document, when finding co-occurrences with min_docs=2, then return empty list."""
        documents = ["The quick fox jumps"]
        expected = []
        result = find_cooccurring_words(
            documents, min_docs=2, ngram_range=(1, 2))
        assert result == expected
