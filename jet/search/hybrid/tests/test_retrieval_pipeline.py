import pytest
from jet.search.hybrid.retrieval_pipeline import Document, RetrievalPipeline


@pytest.fixture
def sample_docs():
    return [
        Document(
            id="1",
            text="JWT authentication enables stateless API security.",
            metadata={"category": "security"},
        ),
        Document(
            id="2",
            text="How to reset a PostgreSQL password safely.",
            metadata={"category": "database"},
        ),
        Document(
            id="3",
            text="Token refresh flow in authentication systems.",
            metadata={"category": "security"},
        ),
    ]


@pytest.fixture
def pipeline(sample_docs):
    pipe = RetrievalPipeline()
    pipe.add_documents(sample_docs)
    return pipe


def test_metadata_filter(pipeline):
    results = pipeline.retrieve(
        query="JWT token refresh", metadata_filters={"category": "security"}, top_k=2
    )

    assert len(results) == 2
    for doc in results:
        assert doc.metadata["category"] == "security"


def test_retrieval_relevance(pipeline):
    results = pipeline.retrieve(query="How does JWT refresh work?", top_k=1)

    assert results[0].id in {"1", "3"}


def test_database_query(pipeline):
    results = pipeline.retrieve(query="reset postgres password", top_k=1)

    assert results[0].id == "2"
