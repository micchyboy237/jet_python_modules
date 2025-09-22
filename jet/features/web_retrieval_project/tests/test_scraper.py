import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from jet.features.web_retrieval_project.src.scraper import scrape_recursive_url

@pytest.fixture
def mock_loader():
    with patch('langchain_community.document_loaders.RecursiveUrlLoader.load') as mock:
        mock.return_value = [
            Document(page_content="Python uses reference counting.", metadata={"source": "https://docs.python.org/3.12/reference/datamodel.html"}),
            Document(page_content="Text about lists.", metadata={"source": "https://docs.python.org/3.12/tutorial/"})
        ]
        yield mock

def test_scrape_recursive_url(mock_loader):
    # Given: A root URL and max depth
    root_url = "https://docs.python.org/3.12/"
    max_depth = 1

    # When: Scraping is called
    docs = scrape_recursive_url(root_url, max_depth=max_depth)

    # Then: Exactly 2 docs are scraped with correct metadata
    expected_docs = [
        {"page_content": "Python uses reference counting.", "source": "https://docs.python.org/3.12/reference/datamodel.html"},
        {"page_content": "Text about lists.", "source": "https://docs.python.org/3.12/tutorial/"}
    ]
    assert len(docs) == 2
    for i, doc in enumerate(docs):
        assert doc.page_content == expected_docs[i]["page_content"]
        assert doc.metadata["source"] == expected_docs[i]["source"]
