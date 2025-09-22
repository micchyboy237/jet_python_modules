import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from jet.features.web_retrieval_project.src.indexer import index_scraped_docs

@pytest.fixture
def mock_documents():
    return [
        Document(page_content="Python uses reference counting.", metadata={"source": "https://docs.python.org/3.12/"}),
        Document(page_content="Text about lists.", metadata={"source": "https://docs.python.org/3.12/"})
    ]

def test_index_scraped_docs(mock_documents):
    # Given: A list of documents
    documents = mock_documents

    # When: Indexing is called
    with patch('langchain_community.vectorstores.FAISS.from_documents') as mock_faiss:
        mock_vectorstore = Mock()
        mock_faiss.return_value = mock_vectorstore
        vectorstore = index_scraped_docs(documents, chunk_size=500, chunk_overlap=100)

    # Then: Vector store is created
    assert vectorstore == mock_vectorstore
    mock_faiss.assert_called_once()
