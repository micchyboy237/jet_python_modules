from typing import TypedDict
from langchain_community.vectorstores import FAISS

class RAGInput(TypedDict):
    query: str
    retriever: FAISS
