import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from jet.features.web_retrieval_project.src.retriever import rag_query, RAGInput

@pytest.fixture
def mock_retriever():
    retriever = Mock()
    retriever.invoke.return_value = [Document(page_content="Python uses reference counting.", metadata={})]
    return retriever

def test_rag_query(mock_retriever):
    # Given: A query and retriever
    input_data = RAGInput(query="How does Python handle memory?", retriever=mock_retriever)

    # When: RAG query is executed
    with patch('langchain_openai.ChatOpenAI') as mock_llm:
        mock_llm.return_value.invoke.return_value.content = "Python uses reference counting."
        answer = rag_query(input_data)

    # Then: Answer matches expected
    expected = "Python uses reference counting."
    assert answer == expected
