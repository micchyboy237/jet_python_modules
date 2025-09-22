from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings

def index_scraped_docs(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> FAISS:
    """Split long documents into chunks and index in a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="embeddinggemma")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore
