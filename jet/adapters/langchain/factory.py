# jet.adapters.langchain.factory


from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.factory import get_embedding_client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_chat_openai(**kwargs) -> ChatOpenAI:
    llm = ChatLlamaCpp(**kwargs)
    return llm


def get_openai_embeddings() -> OpenAIEmbeddings:
    client = get_embedding_client()
    return OpenAIEmbeddings(
        client=client.embeddings,  # Fix: Use .embeddings sub-client
        model=EMBED_MODEL,  # Specify model for compatibility
        check_embedding_ctx_length=False,  # Disable token checking for non-OpenAI APIs
    )
