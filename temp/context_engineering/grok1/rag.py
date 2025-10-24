from langchain import hub
from langchain_core.documents import Document
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from utils import split_documents, create_vector_store

from config import EMBEDDING_MODEL, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.llm import LlamacppLLM

def create_vector_store(documents, embedding_model):
    embedding_client = LlamacppEmbedding(model=embedding_model, base_url="http://shawn-pc.local:8081/v1")
    texts = [doc.page_content for doc in documents]
    embs = embedding_client.get_embeddings(texts, return_format="numpy")
    # Simple in-memory store using the same logic as InMemoryVectorStore
    store = InMemoryVectorStore.from_documents(documents, embedding=None)  # placeholder
    store.embeddings = embs.tolist()
    store.doc_texts = texts
    return store

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
    llm = LlamacppLLM(model=LLM_MODEL, base_url="http://shawn-pc.local:8080/v1", verbose=True)

    # Generate response
    messages = prompt.invoke({"question": query, "context": context})
    response = llm.chat([{"role": "user", "content": messages[0].content}], temperature=0.0)
    
    return response.content