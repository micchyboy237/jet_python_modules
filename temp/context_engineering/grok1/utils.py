from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

def split_documents(documents, chunk_size, chunk_overlap):
    """
    Splits documents into smaller chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_vector_store(documents, embedding_model):
    """
    Embeds documents and creates an in-memory vector store.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return InMemoryVectorStore.from_documents(documents, embeddings)