import os
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from jet.logger import logger
from .config import EMBEDDING_MODEL, OLLAMA_BASE_URL, CHROMA_PERSIST_DIRECTORY, DOCS_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K


def load_documents(folder_path: str = DOCS_DIRECTORY) -> List[Document]:
    """Load documents from folder. Raises ValueError if no documents found."""
    logger.debug(f"Loading documents from {folder_path}")
    if not os.path.exists(folder_path):
        logger.error(f"Directory {folder_path} does not exist")
        raise ValueError(f"Directory {folder_path} does not exist")
    loader = PyPDFDirectoryLoader(folder_path)
    docs = loader.load()
    if not docs:
        logger.error(f"No documents found in {folder_path}")
        raise ValueError(f"No documents found in {folder_path}")
    logger.debug(f"Loaded {len(docs)} documents")
    return docs


def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(docs)


def create_vectorstore(docs):
    """Create and persist vector store."""
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    vectorstore.persist()
    return vectorstore


def get_retriever():
    """Get or create retriever."""
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY, embedding_function=embeddings)
    except:
        docs = load_documents()
        splits = split_documents(docs)
        vectorstore = create_vectorstore(splits)
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
