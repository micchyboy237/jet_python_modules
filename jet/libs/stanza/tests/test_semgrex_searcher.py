import pytest
from typing import List, Dict, Any
from jet.libs.stanza.semgrex_searcher import SemgrexSearcher, SemgrexMatch


class DummyClient:
    """
    Fake CoreNLP-like client returning consistent Semgrex structure.
    """
    def semgrex(self, text: str, pattern: str) -> List[Dict[str, Any]]:
        # Simulate different sentence structures per document
        if "Chris wrote" in text:
            return [
                {
                    "sentenceIndex": 0,
                    "matchNumber": 1,
                    "nodes": {
                        "subject": {"text": "Chris", "index": 1, "tag": "NNP"},
                        "verb": {"text": "wrote", "index": 2, "tag": "VBD"}
                    }
                }
            ]
        elif "He gives" in text and "She reads" in text:
            return [
                {
                    "sentenceIndex": 0,
                    "matchNumber": 1,
                    "nodes": {
                        "subject": {"text": "He", "index": 1, "tag": "PRP"},
                        "verb": {"text": "gives", "index": 2, "tag": "VBZ"}
                    }
                },
                {
                    "sentenceIndex": 1,
                    "matchNumber": 1,
                    "nodes": {
                        "subject": {"text": "She", "index": 1, "tag": "PRP"},
                        "verb": {"text": "reads", "index": 2, "tag": "VBZ"}
                    }
                }
            ]
        else:
            return []


@pytest.fixture
def searcher():
    client = DummyClient()
    return SemgrexSearcher(client)


class TestSemgrexSearcherSuccess:
    """
    Test successful search and normalization on single and multiple documents.
    """

    # Given: a single document with one match
    # When: calling search()
    # Then: returns normalized match with doc_index=0
    def test_search_single_document(self, searcher):
        text = "Chris wrote a book."
        pattern = "{pos:NN} >nsubj {}"
        result = searcher.search(text, pattern)

        expected: List[SemgrexMatch] = [
            {
                "doc_index": 0,
                "sentence_index": 0,
                "match_index": 1,
                "nodes": [
                    {"name": "subject", "text": "Chris", "attributes": {"index": 1, "tag": "NNP"}},
                    {"name": "verb", "text": "wrote", "attributes": {"index": 2, "tag": "VBD"}},
                ],
            }
        ]
        assert result == expected

    # Given: multiple documents with varying sentence counts
    # When: calling search_documents()
    # Then: all matches include correct doc_index and are normalized
    def test_search_documents_multiple(self, searcher):
        docs = [
            "Chris wrote a book.",                    # doc 0
            "He gives oranges. She reads books.",     # doc 1
        ]
        pattern = "{pos:NN} >nsubj {}"
        result = searcher.search_documents(docs, pattern)

        expected: List[SemgrexMatch] = [
            {
                "doc_index": 0,
                "sentence_index": 0,
                "match_index": 1,
                "nodes": [
                    {"name": "subject", "text": "Chris", "attributes": {"index": 1, "tag": "NNP"}},
                    {"name": "verb", "text": "wrote", "attributes": {"index": 2, "tag": "VBD"}},
                ],
            },
            {
                "doc_index": 1,
                "sentence_index": 0,
                "match_index": 1,
                "nodes": [
                    {"name": "subject", "text": "He", "attributes": {"index": 1, "tag": "PRP"}},
                    {"name": "verb", "text": "gives", "attributes": {"index": 2, "tag": "VBZ"}},
                ],
            },
            {
                "doc_index": 1,
                "sentence_index": 1,
                "match_index": 1,
                "nodes": [
                    {"name": "subject", "text": "She", "attributes": {"index": 1, "tag": "PRP"}},
                    {"name": "verb", "text": "reads", "attributes": {"index": 2, "tag": "VBZ"}},
                ],
            },
        ]
        assert result == expected

    # Given: search() is called
    # When: underlying search_documents is used
    # Then: search() forwards to search_documents with list of one
    def test_search_forwards_to_search_documents(self, searcher, monkeypatch):
        calls = []

        def mock_search_documents(docs, pattern):
            calls.append((docs, pattern))
            return [{"doc_index": 0, "sentence_index": 0, "match_index": 1, "nodes": []}]

        monkeypatch.setattr(searcher, "search_documents", mock_search_documents)
        searcher.search("single doc", "pattern")
        assert calls == [(["single doc"], "pattern")]


class TestSemgrexSearcherFiltering:
    """
    Test all filter_match scenarios with multi-document results.
    """

    @pytest.fixture(autouse=True)
    def setup_matches(self, searcher):
        self.docs = [
            "Chris wrote a book.",
            "He gives oranges. She reads books.",
        ]
        self.pattern = "{pos:NN} >nsubj {}"
        self.matches = searcher.search_documents(self.docs, self.pattern)

    # Given: matches from multiple docs
    # When: filtering by doc_index
    # Then: only matches from that document are returned
    def test_filter_by_doc_index(self, searcher):
        result = searcher.filter_matches(self.matches, doc_index=0)
        expected = [m for m in self.matches if m["doc_index"] == 0]
        assert result == expected

        result = searcher.filter_matches(self.matches, doc_index=1)
        expected = [m for m in self.matches if m["doc_index"] == 1]
        assert result == expected

    # Given: matches across sentences
    # When: filtering by sentence_index
    # Then: only matches from that sentence (within any doc) are kept
    def test_filter_by_sentence_index(self, searcher):
        result = searcher.filter_matches(self.matches, sentence_index=0)
        expected = [m for m in self.matches if m["sentence_index"] == 0]
        assert result == expected

    # Given: matches with named nodes
    # When: filtering by node_name
    # Then: only matches containing that node name are kept
    def test_filter_by_node_name(self, searcher):
        result = searcher.filter_matches(self.matches, node_name="subject")
        expected = self.matches  # all have subject
        assert result == expected

        result = searcher.filter_matches(self.matches, node_name="object")
        expected = []
        assert result == expected

    # Given: matches with node text
    # When: filtering by node_text_contains (case-insensitive)
    # Then: matches with substring in any node text are kept
    def test_filter_by_node_text_contains(self, searcher):
        result = searcher.filter_matches(self.matches, node_text_contains="chris")
        expected = [m for m in self.matches if m["doc_index"] == 0]
        assert result == expected

        result = searcher.filter_matches(self.matches, node_text_contains="READS")
        expected = [m for m in self.matches if m["doc_index"] == 1 and m["sentence_index"] == 1]
        assert result == expected

    # Given: nodes with attributes
    # When: filtering by node_attr (exact key/value match)
    # Then: only matches with a node having all attributes are kept
    def test_filter_by_node_attr(self, searcher):
        result = searcher.filter_matches(self.matches, node_attr={"tag": "VBD"})
        expected = [m for m in self.matches if m["doc_index"] == 0]
        assert result == expected

        result = searcher.filter_matches(self.matches, node_attr={"tag": "PRP", "index": 1})
        expected = [m for m in self.matches if m["doc_index"] == 1]
        assert result == expected

    # Given: multiple filter criteria
    # When: combining doc_index + node_attr + text
    # Then: all conditions must pass
    def test_combined_filters(self, searcher):
        result = searcher.filter_matches(
            self.matches,
            doc_index=1,
            node_name="verb",
            node_text_contains="gives",
            node_attr={"tag": "VBZ"}
        )
        expected = [m for m in self.matches if m["doc_index"] == 1 and m["sentence_index"] == 0]
        assert result == expected

        result = searcher.filter_matches(
            self.matches,
            doc_index=0,
            node_text_contains="unknown"
        )
        expected = []
        assert result == expected


class TestSemgrexSearcherErrors:
    """
    Test error handling for missing/invalid client.
    """

    def test_no_client_raises(self):
        searcher = SemgrexSearcher(None)
        with pytest.raises(RuntimeError, match="No client configured"):
            searcher.search("text", "pattern")

        with pytest.raises(RuntimeError, match="No client configured"):
            searcher.search_documents(["text"], "pattern")

    def test_client_without_semgrex_raises(self):
        class NoSemgrexClient:
            pass

        searcher = SemgrexSearcher(NoSemgrexClient())
        with pytest.raises(RuntimeError, match="Client does not expose"):
            searcher.search("text", "pattern")


class TestSemgrexSearcherOptionalClient:
    """
    Test behavior when client is optional or set later.
    """

    # Given: searcher with no client
    # When: calling search() with raw data and mode="raw"
    # Then: normalizes correctly without needing a client
    def test_search_raw_without_client(self):
        searcher = SemgrexSearcher()  # no client

        raw_match = {
            "sentenceIndex": 0,
            "matchNumber": 1,
            "nodes": {
                "subject": {"text": "Chris", "tag": "NNP"}
            }
        }

        result = searcher.search(raw_match, mode="raw")
        expected: List[SemgrexMatch] = [
            {
                "doc_index": -1,
                "sentence_index": 0,
                "match_index": 1,
                "nodes": [
                    {"name": "subject", "text": "Chris", "attributes": {"tag": "NNP"}}
                ]
            }
        ]
        assert result == expected

    # Given: searcher with no client
    # When: calling search_documents() with list of raw results
    # Then: assigns correct doc_index and normalizes
    def test_search_documents_raw_list(self):
        searcher = SemgrexSearcher()

        raw_list = [
            [  # doc 0
                {
                    "sentenceIndex": 0,
                    "matchNumber": 1,
                    "nodes": {"subj": {"text": "I", "tag": "PRP"}}
                }
            ],
            [  # doc 1
                {
                    "sentenceIndex": 0,
                    "matchNumber": 1,
                    "nodes": {"subj": {"text": "You", "tag": "PRP"}}
                }
            ]
        ]

        result = searcher.search_documents(raw_list, mode="raw")
        expected: List[SemgrexMatch] = [
            {
                "doc_index": 0,
                "sentence_index": 0,
                "match_index": 1,
                "nodes": [{"name": "subj", "text": "I", "attributes": {"tag": "PRP"}}]
            },
            {
                "doc_index": 1,
                "sentence_index": 0,
                "match_index": 1,
                "nodes": [{"name": "subj", "text": "You", "attributes": {"tag": "PRP"}}]
            }
        ]
        assert result == expected

    # Given: searcher created without client
    # When: set_client() is called later
    # Then: search() works with live client
    def test_set_client_later(self, monkeypatch):
        searcher = SemgrexSearcher()

        class LiveClient:
            def semgrex(self, text, pattern):
                return [{"sentenceIndex": 0, "matchNumber": 1, "nodes": {"n": {"text": "set"}}}]

        searcher.set_client(LiveClient())
        result = searcher.search("test", "pat")
        assert len(result) == 1
        assert result[0]["doc_index"] == 0

    # Given: no client and no mode="raw"
    # When: calling search() with text/pattern
    # Then: raises clear RuntimeError
    def test_search_text_without_client_raises(self):
        searcher = SemgrexSearcher()
        with pytest.raises(RuntimeError, match="No client configured"):
            searcher.search("text", "pattern")


class TestSemgrexSearcherEdgeCases:
    """
    Test empty inputs, malformed raw data, and edge filtering.
    """

    def test_search_documents_empty_list(self, searcher, monkeypatch):
        # Given: empty doc list
        # When: calling search_documents
        # Then: returns empty list
        result = searcher.search_documents([], "pattern")
        assert result == []

    def test_filter_matches_empty_list(self, searcher):
        result = searcher.filter_matches([], doc_index=0)
        assert result == []

    def test_normalize_handles_varied_raw_formats(self, monkeypatch):
        client = DummyClient()
        searcher = SemgrexSearcher(client)

        # Patch client to return non-standard formats
        def mock_semgrex(text, pattern):
            return {"matches": []}
        monkeypatch.setattr(client, "semgrex", mock_semgrex)
        assert searcher.search_documents(["doc"], "pat") == []

        def mock_semgrex2(text, pattern):
            return []
        monkeypatch.setattr(client, "semgrex", mock_semgrex2)
        assert searcher.search_documents(["doc"], "pat") == []