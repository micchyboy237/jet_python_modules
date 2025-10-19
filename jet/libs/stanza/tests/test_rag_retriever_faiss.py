import pytest
from jet.libs.stanza.rag_retriever_faiss import RagRetrieverFAISS, ContextChunk


class TestRagRetrieverFAISS:
    def setup_method(self):
        # Given realistic preprocessed chunks (mocked from Stanza)
        self.chunks = [
            ContextChunk(
                text="OpenAI released GPT-5 with enhanced multilingual reasoning capabilities.",
                salience=12.5,
                entities=["OpenAI", "GPT-5"],
                sentence_indices=[0, 1],
            ),
            ContextChunk(
                text="Stanford used Stanza for biomedical text processing in multilingual corpora.",
                salience=10.1,
                entities=["Stanford", "Stanza"],
                sentence_indices=[2],
            ),
            ContextChunk(
                text="McKinsey analyzed RAG pipelines using transformer embeddings for enterprise search.",
                salience=14.8,
                entities=["McKinsey", "RAG"],
                sentence_indices=[3, 4],
            ),
        ]

        self.retriever = RagRetrieverFAISS()

    # ----------------------------------------------------------------------
    def test_build_and_query_index(self):
        # Given
        self.retriever.build_index(self.chunks)

        # When
        results = self.retriever.retrieve("OpenAI model release", top_k=2)

        # Then
        assert len(results) == 2
        assert all(isinstance(r[1], ContextChunk) for r in results)
        assert results[0][1].entities  # entities non-empty

    # ----------------------------------------------------------------------
    def test_describe_result_structure(self):
        # Given
        self.retriever.build_index(self.chunks)

        # When
        summary = self.retriever.describe_result("enterprise RAG analysis", top_k=2)

        # Then
        assert "query" in summary
        assert "results" in summary
        assert len(summary["results"]) == 2
        assert all("entities" in r for r in summary["results"])
        assert all("preview" in r for r in summary["results"])

    # ----------------------------------------------------------------------
    def test_invalid_index_error(self):
        # Given
        retriever = RagRetrieverFAISS()

        # When / Then
        with pytest.raises(RuntimeError):
            retriever.retrieve("test query")

    # ----------------------------------------------------------------------
    def test_empty_chunk_list_error(self):
        with pytest.raises(ValueError):
            self.retriever.build_index([])
