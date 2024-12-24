import requests
from typing import Optional, Callable, Union, List, TypedDict
from chromadb import Documents, EmbeddingFunction, Embeddings
from jet.llm.ollama import (
    large_embed_model,
    base_url,

)
from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()


class MemoryEmbedding(TypedDict):
    vector: List[float]  # Embedding vector
    metadata: str        # Any associated metadata (if applicable)


GenerateEmbeddingsReturnType = Union[MemoryEmbedding, List[MemoryEmbedding]]

GenerateMultipleReturnType = Callable[[
    Union[str, List[str]]], List[MemoryEmbedding]]


class SFEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )

        # embed the documents somehow
        embeddings = sentence_transformer_ef(input)
        return embeddings


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(
            self,
            model_name: str = large_embed_model,
            batch_size: int = 32,
            key: str = "",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.key = key

    def __call__(self, input: Documents) -> Embeddings:
        def func(query): return generate_embeddings(
            model=self.model_name,
            text=query,
            url=base_url,
            key=self.key,
        )

        batch_embeddings = generate_multiple(input, func, self.batch_size)
        return batch_embeddings


def get_embedding_function(
    embedding_model=large_embed_model,
    url=base_url,
    batch_size=32,
    key="",
) -> EmbeddingFunction:
    def func(query): return generate_embeddings(
        model=embedding_model,
        text=query,
        url=url,
        key=key,
    )

    return lambda query: generate_multiple(query, func, batch_size)


def generate_multiple(
    query: Union[str, list[str]],
    func: GenerateMultipleReturnType,
    batch_size: int = 32,
) -> list[MemoryEmbedding]:
    if isinstance(query, list):
        embeddings = []
        for i in range(0, len(query), batch_size):
            embeddings.extend(func(query[i: i + batch_size]))
        return embeddings
    else:
        return func(query)


def generate_embeddings(
    model: str,
    text: Union[str, list[str]],
    **kwargs
) -> GenerateEmbeddingsReturnType:
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")

    if isinstance(text, list):
        embeddings = generate_ollama_batch_embeddings(
            **{"model": model, "texts": text, "url": url, "key": key}
        )
    else:
        embeddings = generate_ollama_batch_embeddings(
            **{"model": model, "texts": [text], "url": url, "key": key}
        )
    return embeddings[0] if isinstance(text, str) else embeddings


def generate_ollama_batch_embeddings(
    model: str, texts: list[str], url: str, key: str = ""
) -> Optional[list[list[float]]]:
    try:
        r = requests.post(
            f"{url}/api/embed",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            json={"input": texts, "model": model},
        )
        r.raise_for_status()
        data = r.json()

        if "embeddings" in data:
            return data["embeddings"]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        print(e)
        return None
