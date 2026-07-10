# jet.adapters.langchain.factory


from jet.adapters.llama_cpp.factory import get_embedding_client
from langchain_openai import OpenAIEmbeddings


def get_openai_embeddings() -> OpenAIEmbeddings:
    client = get_embedding_client()
    return OpenAIEmbeddings(
        client=client.embeddings,  # Fix: Use .embeddings sub-client
        model="text-embedding-ada-002",  # Specify model for compatibility
        check_embedding_ctx_length=False,  # Disable token checking for non-OpenAI APIs
    )
