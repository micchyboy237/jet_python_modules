import pytest
import spacy
from typing import List
from jet.wordnet.keywords.helpers import extract_query_candidates


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


class TestExtractQueryCandidates:
    def test_extracts_noun_phrase_with_adjective(self, nlp):
        # Given a query with a noun phrase containing an adjective and a noun
        query = "beautiful mountain landscape"
        expected = ["beautiful mountain", "mountain", "landscape"]

        # When extracting candidates
        result = extract_query_candidates(query, nlp)

        # Then the result should include the adjective-noun phrase and single nouns
        assert sorted(result) == sorted(expected)

    def test_extracts_noun_phrase_with_adverb(self, nlp):
        # Given a query with a noun phrase containing an adverb and a noun
        query = "quickly running river"
        expected = ["running river", "river"]

        # When extracting candidates
        result = extract_query_candidates(query, nlp)

        # Then the result should include the adverb-noun phrase and single noun
        assert sorted(result) == sorted(expected)

    def test_extracts_noun_phrase_with_number(self, nlp):
        # Given a query with a noun phrase containing a number and a noun
        query = "three tall trees"
        expected = ["tall trees", "trees"]

        # When extracting candidates
        result = extract_query_candidates(query, nlp)

        # Then the result should include the number-noun phrase and single noun
        assert sorted(result) == sorted(expected)

    def test_extracts_single_adverb(self, nlp):
        # Given a query with a single adverb
        query = "run quickly now"
        expected = ["quickly"]

        # When extracting candidates
        result = extract_query_candidates(query, nlp)

        # Then the result should include the single adverb
        assert sorted(result) == sorted(expected)

    def test_extracts_proper_noun_with_adjective_and_adverb(self, nlp):
        # Given a query with a proper noun, adjective, and adverb
        query = "beautifully crafted Paris sculpture"
        expected = ["Paris sculpture", "Paris", "sculpture"]

        # When extracting candidates
        result = extract_query_candidates(query, nlp)

        # Then the result should include the proper noun phrase and single proper noun
        assert sorted(result) == sorted(expected)

    def test_extracts_year_with_noun(self, nlp):
        # Given a query with a year and a noun
        query = "2023 election results"
        expected = ["election results", "2023", "election", "results"]

        # When extracting candidates
        result = extract_query_candidates(query, nlp)

        # Then the result should include the year and noun phrase
        assert sorted(result) == sorted(expected)
