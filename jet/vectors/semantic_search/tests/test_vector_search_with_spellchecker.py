from typing import List, Dict, Any
import pytest
from jet.vectors.semantic_search.vector_search_with_spellchecker import SpellCorrectedSearchEngine


@pytest.fixture
def search_engine():
    """Fixture to initialize SpellCorrectedSearchEngine."""
    engine = SpellCorrectedSearchEngine()
    documents = [
        {"id": 1, "content": "The quick brown foxx jumps over the lazy dog"},
        {"id": 2, "content": "A beautifull garden blooms with collorful flowers"},
        {"id": 3, "content": "Teh sun sets slowly behind the mountan"},
    ]
    keywords = ["beautiful garden", "quick fox", "sunset mountain"]
    engine.add_documents(documents, keywords)
    return engine


def test_search_finds_misspelled_documents(search_engine: SpellCorrectedSearchEngine):
    # Given: A query with correctly spelled keywords
    query = "beautiful garden"
    expected = [
        {"id": 2, "content": "A beautiful garden blooms with colorful flowers"}
    ]

    # When: Performing a search
    results = search_engine.search(query, limit=1)

    # Then: The correct document is returned with corrected content
    assert len(results) == 1, "Expected exactly one result"
    assert results[0]["id"] == expected[0]["id"], "Expected document ID 2"
    assert results[0]["content"] == expected[0][
        "content"], f"Expected '{expected[0]['content']}', but got '{results[0]['content']}'"
    assert results[0]["score"] > 0.5, "Expected high semantic similarity score"


def test_search_handles_multiple_keywords(search_engine: SpellCorrectedSearchEngine):
    # Given: A query for a different keyword
    query = "quick fox"
    expected = [
        {"id": 1, "content": "The quick brown fox jumps over the lazy dog"}
    ]

    # When: Performing a search
    results = search_engine.search(query, limit=1)

    # Then: The correct document is returned with corrected content
    assert len(results) == 1, "Expected exactly one result"
    assert results[0]["id"] == expected[0]["id"], "Expected document ID 1"
    assert results[0]["content"] == expected[0][
        "content"], f"Expected '{expected[0]['content']}', but got '{results[0]['content']}'"
    assert results[0]["score"] > 0.5, "Expected high semantic similarity score"


def test_spell_correction_with_custom_dictionary(search_engine: SpellCorrectedSearchEngine):
    # Given: A text with misspellings and a custom dictionary
    text = "A beautifull garden"
    expected = "A beautiful garden"

    # When: Correcting the text
    corrected = search_engine.correct_text(text)[0]

    # Then: The text is corrected properly using the custom dictionary
    assert corrected == expected, f"Expected '{expected}', but got '{corrected}'"


def test_custom_dictionary_includes_query_words(search_engine: SpellCorrectedSearchEngine):
    # Given: A query word that should be in the custom dictionary
    query_word = "beautiful"
    expected = query_word

    # When: Checking if the word is known (not corrected)
    corrected = search_engine.spell_checker.correction(query_word)

    # Then: The query word is preserved as it is in the custom dictionary
    assert corrected == expected, f"Expected '{expected}' to be recognized, but got '{corrected}'"


def test_correction_uses_both_dictionaries(search_engine: SpellCorrectedSearchEngine):
    # Given: A text with a document-specific word and a common misspelling
    text = "mountan collorful"
    expected = "mountain colorful"

    # When: Correcting the text
    corrected = search_engine.correct_text(text)[0]

    # Then: The text is corrected using both custom (query) and built-in dictionaries
    assert corrected == expected, f"Expected '{expected}', but got '{corrected}'"


def test_spell_correction_before_indexing(search_engine: SpellCorrectedSearchEngine):
    # Given: A document with misspellings
    doc_id = 2
    expected_corrected = "A beautiful garden blooms with colorful flowers"
    expected_original = "A beautifull garden blooms with collorful flowers"

    # When: Checking the corrected document after adding
    corrected_doc = next(
        doc for doc in search_engine.corrected_documents if doc["id"] == doc_id)

    # Then: The document is corrected before indexing
    assert corrected_doc[
        "content"] == expected_corrected, f"Expected corrected '{expected_corrected}', but got '{corrected_doc['content']}'"
    assert search_engine.documents[1][
        "content"] == expected_original, f"Expected original '{expected_original}', but got '{search_engine.documents[1]['content']}'"


def test_corrections_are_tracked(search_engine: SpellCorrectedSearchEngine):
    # Given: A document with known misspellings
    doc_id = 2
    expected_corrections = [
        {"original": "beautifull", "corrected": "beautiful"},
        {"original": "collorful", "corrected": "colorful"}
    ]

    # When: Retrieving corrections for the document
    corrections = search_engine.get_corrections(doc_id)

    # Then: The corrections are tracked correctly
    assert len(corrections) == len(
        expected_corrections), f"Expected {len(expected_corrections)} corrections, but got {len(corrections)}"
    for expected, actual in zip(expected_corrections, corrections):
        assert actual["original"] == expected[
            "original"], f"Expected original '{expected['original']}', but got '{actual['original']}'"
        assert actual["corrected"] == expected[
            "corrected"], f"Expected corrected '{expected['corrected']}', but got '{actual['corrected']}'"
