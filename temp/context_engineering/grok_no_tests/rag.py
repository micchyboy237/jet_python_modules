from langchain import hub
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from utils import split_documents, create_vector_store

from config import EMBEDDING_MODEL, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K

def rag_generation(query: str, documents: list[str]):
    """
    Performs RAG: Embeds documents, retrieves relevant chunks, and generates a response.
    """
    # Convert strings to Document objects
    docs = [Document(page_content=doc) for doc in documents]
    
    # Split documents into chunks
    all_splits = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Create vector store with embeddings
    vector_store = create_vector_store(all_splits, EMBEDDING_MODEL)
    
    # Retrieve relevant chunks
    retrieved_docs = vector_store.similarity_search(query, k=RETRIEVAL_K)
    
    # Prepare context from retrieved documents
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Load RAG prompt from LangChain hub
    prompt = hub.pull("rlm/rag-prompt")
    
    # Initialize LLM
    llm = init_chat_model(LLM_MODEL, model_provider="openai")  # Adapt for other providers if needed
    
    # Generate response
    messages = prompt.invoke({"question": query, "context": context})
    response = llm.invoke(messages)
    
    return response.content