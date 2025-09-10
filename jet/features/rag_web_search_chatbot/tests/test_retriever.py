import pytest
import os
import shutil
from jet.features.rag_web_search_chatbot.retriever import load_documents, split_documents, create_vectorstore, get_retriever
from jet.features.rag_web_search_chatbot.config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PERSIST_DIRECTORY, RETRIEVER_K

DOCS_DIRECTORY = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/features/rag_web_search_chatbot/tests/sample_docs"


@pytest.fixture
def sample_docs_path():
    return DOCS_DIRECTORY


def test_load_documents(sample_docs_path):
    docs = load_documents(sample_docs_path)
    assert len(docs) > 0
    assert "page_content" in docs[0].dict()


def test_split_documents():
    from langchain_core.documents import Document
    mock_docs = [Document(page_content="Test content " * 1000)]
    splits = split_documents(
        mock_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    assert len(splits) > 0
    assert len(splits[0].page_content) <= CHUNK_SIZE


def test_create_vectorstore(sample_docs_path):
    docs = load_documents(sample_docs_path)
    splits = split_documents(docs)
    vectorstore = create_vectorstore(splits)
    assert vectorstore._collection.count() > 0
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        shutil.rmtree(CHROMA_PERSIST_DIRECTORY)


def test_get_retriever():
    retriever = get_retriever()
    results = retriever.invoke("test query")
    assert len(results) == RETRIEVER_K
    assert "page_content" in results[0].dict()
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
