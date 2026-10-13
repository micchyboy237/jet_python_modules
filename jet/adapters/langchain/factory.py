from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp
from jet.adapters.langchain.embed_llama_cpp import EmbedLlamaCpp
from jet.adapters.llama_cpp.config import EMBED_MODEL


def get_chat_openai(**kwargs) -> ChatLlamaCpp:
    """Get a ChatLlamaCpp instance configured with jet defaults.

    Args:
        **kwargs: Additional arguments to pass to ChatLlamaCpp constructor.

    Returns:
        ChatLlamaCpp instance.
    """
    llm = ChatLlamaCpp(**kwargs)
    return llm


def get_openai_embeddings(embed_model: str = EMBED_MODEL) -> EmbedLlamaCpp:
    """Get an EmbedLlamaCpp instance configured with jet defaults.

    Args:
        embed_model: The name of the embedding model to use. Defaults to EMBED_MODEL from config.

    Returns:
        EmbedLlamaCpp instance.
    """
    return EmbedLlamaCpp(model=embed_model)
