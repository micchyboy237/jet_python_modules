# test_semgrex_searcher.py
"""
Comprehensive pytest suite for SemgrexSearcher.

Follows BDD style:
- Given: setup data / state
- When: action (search)
- Then: assertions on exact outputs

Uses real-world examples, exact dict matching, and fixture cleanup.
Assumes semgrex_searcher.py is importable from the project root.
"""

import pytest
from typing import List, Dict, Any

# Adjust path if running from project root
from jet.libs.stanza.semgrex_searcher import SemgrexSearcher


@pytest.fixture(scope="module")
def searcher() -> SemgrexSearcher:
    """
    Fixture: One SemgrexSearcher per test module.
    Yields for use, then closes Java server to prevent leaks.
    """
    s = SemgrexSearcher(lang="en", use_java=True)
    yield s
    s.close()  # Ensures Java process terminates


# ----------------------------------------------------------------------
# Helper for cleaner assertions (optional, but improves readability)
# ----------------------------------------------------------------------
def _node_text(nodes: Dict[str, Dict], key: str) -> str:
    return nodes.get(key, {}).get("text", "")


# ----------------------------------------------------------------------
# Test Class: Acquisition Pattern (SVO with proper nouns)
# ----------------------------------------------------------------------
class TestAcquisitionPattern:
    # Given: Text containing acquisition events and distractors
    # When: Searching with subject-acquire-object pattern
    # Then: Only valid triples are returned with correct node texts
    def test_finds_two_acquisitions(self, searcher: SemgrexSearcher):
        text = (
            "Apple acquires Beats for $3 billion in 2014. "
            "Google buys DeepMind. "
            "Microsoft partners with OpenAI but does not acquire it."
        )
        pattern = '{pos:/NNP|PROPN/}=subject >nsubj {lemma:acquire}=verb >dobj {pos:/NNP|PROPN/}=object'

        result: List[Dict[str, Any]] = searcher.search_in_doc(text, pattern)
        expected = [
            {
                "sentence_text": "Apple acquires Beats for $3 billion in 2014.",
                "matched_nodes": {
                    "subject": {"text": "Apple"},
                    "verb": {"text": "acquires"},
                    "object": {"text": "Beats"},
                },
                "length": 3,
            },
            {
                "sentence_text": "Google buys DeepMind.",
                "matched_nodes": {
                    "subject": {"text": "Google"},
                    "verb": {"text": "buys"},
                    "object": {"text": "DeepMind"},
                },
                "length": 3,
            },
        ]

        assert len(result) == 2
        for res, exp in zip(result, expected):
            assert res["sentence_text"] == exp["sentence_text"]
            assert res["length"] == exp["length"]
            # Compare only text fields (indices may vary slightly)
            assert {k: v["text"] for k, v in res["matched_nodes"].items()} == \
                   {k: v["text"] for k, v in exp["matched_nodes"].items()}

    # Given: Text with no acquisition
    # When: Same pattern
    # Then: Empty list
    def test_no_acquisition_returns_empty(self, searcher: SemgrexSearcher):
        text = "The cat sleeps peacefully."
        pattern = '{pos:/NNP|PROPN/}=subject >nsubj {lemma:acquire}=verb >dobj {pos:/NNP|PROPN/}=object'

        result = searcher.search_in_doc(text, pattern)
        expected: List[Dict[str, Any]] = []

        assert result == expected


# ----------------------------------------------------------------------
# Test Class: Passive Voice Detection
# ----------------------------------------------------------------------
class TestPassiveVoicePattern:
    # Given: Mixed active/passive sentences
    # When: Querying for verb with passive 'be' auxiliary
    # Then: Only passive sentences match, with correct aux/verb
    def test_finds_two_passives(self, searcher: SemgrexSearcher):
        text = (
            "The movie was praised by critics. "
            "Users loved the ending. "
            "It was hated by some. "
            "I ate the cake."
        )
        pattern = '{}=verb >aux:pass {lemma:be}=aux'

        result = searcher.search_in_doc(text, pattern)
        expected = [
            {
                "sentence_text": "The movie was praised by critics.",
                "matched_nodes": {
                    "verb": {"text": "praised"},
                    "aux": {"text": "was"},
                },
                "length": 2,
            },
            {
                "sentence_text": "It was hated by some.",
                "matched_nodes": {
                    "verb": {"text": "hated"},
                    "aux": {"text": "was"},
                },
                "length": 2,
            },
        ]

        assert len(result) == 2
        for res, exp in zip(result, expected):
            assert res["sentence_text"] == exp["sentence_text"]
            assert res["length"] == exp["length"]
            assert {k: v["text"] for k, v in res["matched_nodes"].items()} == \
                   {k: v["text"] for k, v in exp["matched_nodes"].items()}

    # Given: Only active sentences
    # When: Passive pattern
    # Then: No matches
    def test_no_passive_returns_empty(self, searcher: SemgrexSearcher):
        text = "I love pizza. She runs fast."
        pattern = '{}=verb >aux:pass {lemma:be}=aux'

        result = searcher.search_in_doc(text, pattern)
        expected: List[Dict[str, Any]] = []

        assert result == expected


# ----------------------------------------------------------------------
# Test Class: WH-Subject Questions
# ----------------------------------------------------------------------
class TestWHSubjectQuestionPattern:
    # Given: Questions and statements
    # When: Pattern for WH-word as subject
    # Then: Only subject-extracted WH questions match
    def test_finds_wh_subject_questions(self, searcher: SemgrexSearcher):
        text = (
            "Who invented the telephone? "
            "What is artificial intelligence? "
            "Bell invented it. "
            "Why did he leave?"
        )
        pattern = '{pos:/WP|WDT/}=wh >nsubj {}=verb'

        result = searcher.search_in_doc(text, pattern)
        expected = [
            {
                "sentence_text": "Who invented the telephone?",
                "matched_nodes": {
                    "wh": {"text": "Who"},
                    "verb": {"text": "invented"},
                },
                "length": 2,
            },
        ]  # "What is..." uses copula, not direct nsubj

        assert len(result) == 1
        res = result[0]
        exp = expected[0]
        assert res["sentence_text"] == exp["sentence_text"]
        assert res["length"] == exp["length"]
        assert {k: v["text"] for k, v in res["matched_nodes"].items()} == \
               {k: v["text"] for k, v in exp["matched_nodes"].items()}

    # Given: No WH-questions
    # When: WH-subject pattern
    # Then: Empty
    def test_no_wh_subject_returns_empty(self, searcher: SemgrexSearcher):
        text = "The sky is blue. Dogs bark."
        pattern = '{pos:/WP|WDT/}=wh >nsubj {}=verb'

        result = searcher.search_in_doc(text, pattern)
        expected: List[Dict[str, Any]] = []

        assert result == expected


# ----------------------------------------------------------------------
# Test Class: Edge Cases & Robustness
# ----------------------------------------------------------------------
class TestEdgeCases:
    # Given: Empty text
    # When: Any pattern
    # Then: Empty list, no crash
    def test_empty_text_returns_empty(self, searcher: SemgrexSearcher):
        text = ""
        pattern = '{pos:NN}'

        result = searcher.search_in_doc(text, pattern)
        expected: List[Dict[str, Any]] = []

        assert result == expected

    # Given: Malformed pattern (invalid syntax)
    # When: search_in_doc
    # Then: Raises ValueError from Semgrex (propagated)
    def test_invalid_pattern_raises_error(self, searcher: SemgrexSearcher):
        text = "Test sentence."
        pattern = '{pos:INVALID'  # Unclosed brace

        with pytest.raises(ValueError, match=".*Semgrex.*"):
            searcher.search_in_doc(text, pattern)

    # Given: Multiple matches in one sentence (overlapping)
    # When: Pattern that could match twice
    # Then: Only one match per root (Semgrex limitation)
    def test_single_match_per_root(self, searcher: SemgrexSearcher):
        text = "The cat that sleeps eats."
        pattern = '{pos:NN}=noun >nsubj {pos:VB}=verb'

        result = searcher.search_in_doc(text, pattern)
        # "cat" and "sleeps" could match, but only one per root
        assert len(result) <= 1  # Known Semgrex behavior


# ----------------------------------------------------------------------
# Run instructions
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main(["-v", __file__])