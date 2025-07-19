import pytest
import numpy as np
from typing import List, Dict
from jet.vectors.semantic_search.aggregation import aggregate_doc_scores, ChunkWithScore
from jet.vectors.semantic_search.search_types import Match


class TestAggregation:
    def test_longer_ngram_ranks_higher(self):
        """Test that a document with a longer n-gram match ranks higher than one with multiple shorter matches."""
        query_candidates = ["react native development", "react", "native"]
        chunks: List[ChunkWithScore] = [
            {
                "id": "chunk1",
                "doc_id": "doc1",
                "score": 0.9,
                "header": "React Native Job",
                "content": "Seeking React Native development expertise.",
                "parent_header": "",
                "matches": [Match(text="react native development", start_idx=8, end_idx=32)],
                "metadata": {"query_scores": {"react native development": 0.9, "react": 0.8, "native": 0.8}}
            },
            {
                "id": "chunk2",
                "doc_id": "doc2",
                "score": 0.95,
                "header": "React Developer",
                "content": "Need react and native skills.",
                "parent_header": "",
                "matches": [
                    Match(text="react", start_idx=5, end_idx=10),
                    Match(text="native", start_idx=15, end_idx=21)
                ],
                "metadata": {"query_scores": {"react": 0.95, "native": 0.9}}
            }
        ]
        data_dict = {
            "doc1": {"id": "doc1", "text": "React Native Job\nSeeking React Native development expertise."},
            "doc2": {"id": "doc2", "text": "React Developer\nNeed react and native skills."}
        }
        expected = [
            {
                "rank": 1,
                "score": pytest.approx(0.9 * (2 + 2 * np.log1p(24) / np.log1p(50)) * 1.0 * 1.5, 0.01),
                "id": "doc1",
                "text": "React Native Job\nSeeking React Native development expertise.",
                "posted_date": "",
                "link": "",
                "num_tokens": 0,
                "metadata": {"query_scores": {"react native development": 0.9, "react": 0.8, "native": 0.8}},
                "matches": [Match(text="react native development", start_idx=8, end_idx=32)]
            },
            {
                "rank": 2,
                "score": pytest.approx(0.95 * (2 + 2 * np.log1p(6) / np.log1p(50)) * (1.0 / (1.0 + 0.2 * max(0, 2 - 3))), 0.01),
                "id": "doc2",
                "text": "React Developer\nNeed react and native skills.",
                "posted_date": "",
                "link": "",
                "num_tokens": 0,
                "metadata": {"query_scores": {"react": 0.95, "native": 0.9}},
                "matches": [
                    Match(text="react", start_idx=5, end_idx=10),
                    Match(text="native", start_idx=15, end_idx=21)
                ]
            }
        ]
        result = aggregate_doc_scores(chunks, data_dict, query_candidates)
        assert len(result) == 2, "Should return two documents"
        assert result[0]["rank"] == 1, "First document should have rank 1"
        assert result[1]["rank"] == 2, "Second document should have rank 2"
        assert result[0]["id"] == "doc1", "First document should be doc1"
        assert result[1]["id"] == "doc2", "Second document should be doc2"
        assert result[0]["score"] > result[1]["score"], "Longer n-gram match should have higher score"
        assert pytest.approx(
            result[0]["score"], 0.01) == expected[0]["score"], "Score for doc1 should match"
        assert pytest.approx(
            result[1]["score"], 0.01) == expected[1]["score"], "Score for doc2 should match"
        assert result[0]["matches"] == expected[0]["matches"], "Matches for doc1 should be correct"
        assert result[1]["matches"] == expected[1]["matches"], "Matches for doc2 should be correct"

    def test_multiple_chunks_with_short_matches(self):
        """Test that a document with multiple chunks of short matches doesn't outrank a single long match."""
        query_candidates = ["react native", "react", "native"]
        chunks: List[ChunkWithScore] = [
            {
                "id": "chunk1",
                "doc_id": "doc1",
                "score": 0.9,
                "header": "React Native Job",
                "content": "Seeking react native expertise.",
                "parent_header": "",
                "matches": [Match(text="react native", start_idx=8, end_idx=20)],
                "metadata": {"query_scores": {"react native": 0.9, "react": 0.8, "native": 0.8}}
            },
            {
                "id": "chunk2",
                "doc_id": "doc2",
                "score": 0.95,
                "header": "React Skill",
                "content": "Need react skills.",
                "parent_header": "",
                "matches": [Match(text="react", start_idx=5, end_idx=10)],
                "metadata": {"query_scores": {"react": 0.95}}
            },
            {
                "id": "chunk3",
                "doc_id": "doc2",
                "score": 0.92,
                "header": "Native Skill",
                "content": "Need native skills.",
                "parent_header": "",
                "matches": [Match(text="native", start_idx=5, end_idx=11)],
                "metadata": {"query_scores": {"native": 0.92}}
            }
        ]
        data_dict = {
            "doc1": {"id": "doc1", "text": "React Native Job\nSeeking react native expertise."},
            "doc2": {"id": "doc2", "text": "React Skill\nNeed react skills.\n\nNative Skill\nNeed native skills."}
        }
        expected = [
            {
                "rank": 1,
                "score": pytest.approx(0.9 * (2 + 2 * np.log1p(12) / np.log1p(50)) * 1.0 * 1.5, 0.01),
                "id": "doc1",
                "text": "React Native Job\nSeeking react native expertise.",
                "posted_date": "",
                "link": "",
                "num_tokens": 0,
                "metadata": {"query_scores": {"react native": 0.9, "react": 0.8, "native": 0.8}},
                "matches": [Match(text="react native", start_idx=8, end_idx=20)]
            },
            {
                "rank": 2,
                "score": pytest.approx(((0.95 + 0.92) / 2) * (2 + 2 * np.log1p(6) / np.log1p(50)) * (1.0 / (1.0 + 0.2 * max(0, 2 - 3))), 0.01),
                "id": "doc2",
                "text": "React Skill\nNeed react skills.\n\nNative Skill\nNeed native skills.",
                "posted_date": "",
                "link": "",
                "num_tokens": 0,
                "metadata": {"query_scores": {"react": 0.95, "native": 0.92}},
                "matches": [
                    Match(text="react", start_idx=5, end_idx=10),
                    Match(text="native", start_idx=37, end_idx=43)
                ]
            }
        ]
        result = aggregate_doc_scores(chunks, data_dict, query_candidates)
        assert len(result) == 2, "Should return two documents"
        assert result[0]["rank"] == 1, "First document should have rank 1"
        assert result[1]["rank"] == 2, "Second document should have rank 2"
        assert result[0]["id"] == "doc1", "First document should be doc1"
        assert result[1]["id"] == "doc2", "Second document should be doc2"
        assert result[0]["score"] > result[1]["score"], "Longer n-gram match should have higher score"
        assert pytest.approx(
            result[0]["score"], 0.01) == expected[0]["score"], "Score for doc1 should match"
        assert pytest.approx(
            result[1]["score"], 0.01) == expected[1]["score"], "Score for doc2 should match"
        assert result[0]["matches"] == expected[0]["matches"], "Matches for doc1 should be correct"
        assert result[1]["matches"] == expected[1]["matches"], "Matches for doc2 should be correct"

    def test_empty_chunks(self):
        """Test that empty chunk list returns empty document list."""
        query_candidates = ["react native"]
        chunks: List[ChunkWithScore] = []
        data_dict = {"doc1": {"id": "doc1", "text": "Some text"}}
        result = aggregate_doc_scores(chunks, data_dict, query_candidates)
        expected: List[dict] = []
        assert result == expected, "Empty chunks should return empty list"

    def test_missing_query_scores(self):
        """Test that chunks with missing query scores are handled gracefully."""
        query_candidates = ["react native"]
        chunks: List[ChunkWithScore] = [
            {
                "id": "chunk1",
                "doc_id": "doc1",
                "score": 0.9,
                "header": "React Native Job",
                "content": "Seeking react native expertise.",
                "parent_header": "",
                "matches": [Match(text="react native", start_idx=8, end_idx=20)],
                "metadata": {}
            }
        ]
        data_dict = {
            "doc1": {"id": "doc1", "text": "React Native Job\nSeeking react native expertise."}
        }
        expected = [
            {
                "rank": 1,
                "score": pytest.approx(0.9 * (2 + 2 * np.log1p(12) / np.log1p(50)) * 1.0 * 1.5, 0.01),
                "id": "doc1",
                "text": "React Native Job\nSeeking react native expertise.",
                "posted_date": "",
                "link": "",
                "num_tokens": 0,
                "metadata": {"query_scores": {}},
                "matches": [Match(text="react native", start_idx=8, end_idx=20)]
            }
        ]
        result = aggregate_doc_scores(chunks, data_dict, query_candidates)
        assert len(result) == 1, "Should return one document"
        assert result[0]["rank"] == 1, "Document should have rank 1"
        assert result[0]["id"] == "doc1", "Document should be doc1"
        assert pytest.approx(
            result[0]["score"], 0.01) == expected[0]["score"], "Score should match"
        assert result[0]["matches"] == expected[0]["matches"], "Matches should be correct"
        assert result[0]["metadata"]["query_scores"] == {
        }, "Query scores should be empty"

    def test_many_half_ngram_matches(self):
        """Test that a document with many half n-gram matches ranks lower than one with fewer full matches."""
        query_candidates = ["react native", "react", "native"]
        chunks: List[ChunkWithScore] = [
            {
                "id": "chunk1",
                "doc_id": "doc1",
                "score": 0.9,
                "header": "React Native Job",
                "content": "Seeking react native expertise.",
                "parent_header": "",
                "matches": [Match(text="react native", start_idx=8, end_idx=20)],
                "metadata": {"query_scores": {"react native": 0.9, "react": 0.8, "native": 0.8}}
            },
            {
                "id": "chunk2",
                "doc_id": "doc2",
                "score": 0.95,
                "header": "React Developer",
                "content": "Need react skills.",
                "parent_header": "",
                "matches": [Match(text="react", start_idx=5, end_idx=10)],
                "metadata": {"query_scores": {"react": 0.95}}
            },
            {
                "id": "chunk3",
                "doc_id": "doc2",
                "score": 0.95,
                "header": "React Skill",
                "content": "Need react skills again.",
                "parent_header": "",
                "matches": [Match(text="react", start_idx=5, end_idx=10)],
                "metadata": {"query_scores": {"react": 0.95}}
            },
            {
                "id": "chunk4",
                "doc_id": "doc2",
                "score": 0.95,
                "header": "React Experience",
                "content": "Need react experience.",
                "parent_header": "",
                "matches": [Match(text="react", start_idx=5, end_idx=10)],
                "metadata": {"query_scores": {"react": 0.95}}
            }
        ]
        data_dict = {
            "doc1": {"id": "doc1", "text": "React Native Job\nSeeking react native expertise."},
            "doc2": {"id": "doc2", "text": "React Developer\nNeed react skills.\n\nReact Skill\nNeed react skills again.\n\nReact Experience\nNeed react experience."}
        }
        expected = [
            {
                "rank": 1,
                "score": pytest.approx(0.9 * (2 + 2 * np.log1p(12) / np.log1p(50)) * 1.0 * 1.5, 0.01),
                "id": "doc1",
                "text": "React Native Job\nSeeking react native expertise.",
                "posted_date": "",
                "link": "",
                "num_tokens": 0,
                "metadata": {"query_scores": {"react native": 0.9, "react": 0.8, "native": 0.8}},
                "matches": [Match(text="react native", start_idx=8, end_idx=20)]
            },
            {
                "rank": 2,
                "score": pytest.approx(((0.95 + 0.95 + 0.95) / 3) * (2 + 2 * np.log1p(5) / np.log1p(50)) * (1.0 / (1.0 + 0.2 * max(0, 3 - 3))), 0.01),
                "id": "doc2",
                "text": "React Developer\nNeed react skills.\n\nReact Skill\nNeed react skills again.\n\nReact Experience\nNeed react experience.",
                "posted_date": "",
                "link": "",
                "num_tokens": 0,
                "metadata": {"query_scores": {"react": 0.95}},
                "matches": [
                    Match(text="react", start_idx=5, end_idx=10),
                    Match(text="react", start_idx=41, end_idx=46),
                    Match(text="react", start_idx=79, end_idx=84)
                ]
            }
        ]
        result = aggregate_doc_scores(chunks, data_dict, query_candidates)
        assert len(result) == 2, "Should return two documents"
        assert result[0]["rank"] == 1, "First document should have rank 1"
        assert result[1]["rank"] == 2, "Second document should have rank 2"
        assert result[0]["id"] == "doc1", "First document should be doc1"
        assert result[1]["id"] == "doc2", "Second document should be doc2"
        assert result[0]["score"] > result[1]["score"], "Longer n-gram match should have higher score"
        assert pytest.approx(
            result[0]["score"], 0.01) == expected[0]["score"], "Score for doc1 should match"
        assert pytest.approx(
            result[1]["score"], 0.01) == expected[1]["score"], "Score for doc2 should match"
        assert result[0]["matches"] == expected[0]["matches"], "Matches for doc1 should be correct"
        assert result[1]["matches"] == expected[1]["matches"], "Matches for doc2 should be correct"

    def test_no_matches_vs_full_match(self):
        """Test that a document with a full n-gram match ranks higher than one with no matches."""
        query_candidates = ["react native"]
        chunks: List[ChunkWithScore] = [
            {
                "id": "chunk1",
                "doc_id": "doc1",
                "score": 0.05,
                "header": "React Native Job",
                "content": "Seeking react native expertise.",
                "parent_header": "",
                "matches": [Match(text="react native", start_idx=8, end_idx=20)],
                "metadata": {"query_scores": {"react native": 0.05}}
            },
            {
                "id": "chunk2",
                "doc_id": "doc2",
                "score": 0.1,
                "header": "Web Developer",
                "content": "Need web development skills.",
                "parent_header": "",
                "matches": [],
                "metadata": {"query_scores": {"react native": 0.1}}
            }
        ]
        data_dict = {
            "doc1": {"id": "doc1", "text": "React Native Job\nSeeking react native expertise."},
            "doc2": {"id": "doc2", "text": "Web Developer\nNeed web development skills."}
        }
        expected = [
            {
                "rank": 1,
                "score": pytest.approx(0.05 * (2 + 2 * np.log1p(12) / np.log1p(50)) * 1.0 * 1.5, 0.01),
                "id": "doc1",
                "text": "React Native Job\nSeeking react native expertise.",
                "posted_date": "",
                "link": "",
                "num_tokens": 0,
                "metadata": {"query_scores": {"react native": 0.05}},
                "matches": [Match(text="react native", start_idx=8, end_idx=20)]
            },
            {
                "rank": 2,
                "score": pytest.approx(0.1 * (2 + 2 * np.log1p(0) / np.log1p(50)) * 0.3, 0.01),
                "id": "doc2",
                "text": "Web Developer\nNeed web development skills.",
                "posted_date": "",
                "link": "",
                "num_tokens": 0,
                "metadata": {"query_scores": {"react native": 0.1}},
                "matches": []
            }
        ]
        result = aggregate_doc_scores(chunks, data_dict, query_candidates)
        assert len(result) == 2, "Should return two documents"
        assert result[0]["rank"] == 1, "First document should have rank 1"
        assert result[1]["rank"] == 2, "Second document should have rank 2"
        assert result[0]["id"] == "doc1", "First document should be doc1"
        assert result[1]["id"] == "doc2", "Second document should be doc2"
        assert result[0]["score"] > result[1]["score"], "Full n-gram match should have higher score"
        assert pytest.approx(
            result[0]["score"], 0.01) == expected[0]["score"], "Score for doc1 should match"
        assert pytest.approx(
            result[1]["score"], 0.01) == expected[1]["score"], "Score for doc2 should match"
        assert result[0]["matches"] == expected[0]["matches"], "Matches for doc1 should be correct"
        assert result[1]["matches"] == expected[1]["matches"], "Matches for doc2 should be correct"

    def test_half_matches_vs_full_match(self):
        """Test that a document with a full n-gram match ranks higher than one with half matches."""
        query_candidates = ["react native", "react", "native"]
        chunks: List[ChunkWithScore] = [
            {
                "id": "chunk1",
                "doc_id": "doc1",
                "score": 0.534,
                "header": "React Native Job",
                "content": "Seeking react native expertise.",
                "parent_header": "",
                "matches": [Match(text="react native", start_idx=8, end_idx=20)],
                "metadata": {"query_scores": {"react native": 0.534}}
            },
            {
                "id": "chunk2",
                "doc_id": "doc2",
                "score": 0.671,
                "header": "Front-End Developer",
                "content": "Need react skills. Expertise in react frameworks.",
                "parent_header": "",
                "matches": [
                    Match(text="react", start_idx=5, end_idx=10),
                    Match(text="react", start_idx=28, end_idx=33)
                ],
                "metadata": {"query_scores": {"react": 0.671}}
            }
        ]
        data_dict = {
            "doc1": {"id": "doc1", "text": "React Native Job\nSeeking react native expertise."},
            "doc2": {"id": "doc2", "text": "Front-End Developer\nNeed react skills. Expertise in react frameworks."}
        }
        expected = [
            {
                "rank": 1,
                "score": pytest.approx(0.534 * (2 + 2 * np.log1p(12) / np.log1p(50)) * 1.0 * 1.5, 0.01),
                "id": "doc1",
                "text": "React Native Job\nSeeking react native expertise.",
                "posted_date": "",
                "link": "",
                "num_tokens": 0,
                "metadata": {"query_scores": {"react native": 0.534}},
                "matches": [Match(text="react native", start_idx=8, end_idx=20)]
            },
            {
                "rank": 2,
                "score": pytest.approx(0.671 * (2 + 2 * np.log1p(5) / np.log1p(50)) * (1.0 / (1.0 + 0.2 * max(0, 2 - 3))), 0.01),
                "id": "doc2",
                "text": "Front-End Developer\nNeed react skills. Expertise in react frameworks.",
                "posted_date": "",
                "link": "",
                "num_tokens": 0,
                "metadata": {"query_scores": {"react": 0.671}},
                "matches": [
                    Match(text="react", start_idx=5, end_idx=10),
                    Match(text="react", start_idx=28, end_idx=33)
                ]
            }
        ]
        result = aggregate_doc_scores(chunks, data_dict, query_candidates)
        assert len(result) == 2, "Should return two documents"
        assert result[0]["rank"] == 1, "First document should have rank 1"
        assert result[1]["rank"] == 2, "Second document should have rank 2"
        assert result[0]["id"] == "doc1", "First document should be doc1"
        assert result[1]["id"] == "doc2", "Second document should be doc2"
        assert result[0]["score"] > result[1]["score"], "Full n-gram match should have higher score"
        assert pytest.approx(
            result[0]["score"], 0.01) == expected[0]["score"], "Score for doc1 should match"
        assert pytest.approx(
            result[1]["score"], 0.01) == expected[1]["score"], "Score for doc2 should match"
        assert result[0]["matches"] == expected[0]["matches"], "Matches for doc1 should be correct"
        assert result[1]["matches"] == expected[1]["matches"], "Matches for doc2 should be correct"
