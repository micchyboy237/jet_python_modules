import numpy as np
from openai import OpenAI
from tqdm import tqdm
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.logger import logger


def main():
    """Example usage of EmbeddingClient."""
    client = OpenAI(
        base_url="http://shawn-pc.local:8081/v1",
        api_key="no-key-required",
        max_retries=3,
    )

    # Example inputs
    texts = [
        "This is a sample text to generate embeddings.",
        "Another text for embedding.",
    ]

    model: LLAMACPP_EMBED_KEYS = "nomic-embed-text"
    batch_size = 32
    embeddings_list = []
    embeddings_numpy = []
    progress_bar = tqdm(range(0, len(texts), batch_size), desc="Processing batches")

    for i in progress_bar:
        batch = texts[i : i + batch_size]
        try:
            response = client.embeddings.create(model=model, input=batch)
            batch_embeddings = [d.embedding for d in response.data]
            batch_numpy_embeddings = [np.array(emb) for emb in batch_embeddings]

            embeddings_list.extend(batch_embeddings)
            embeddings_numpy.extend(batch_numpy_embeddings)
        except Exception as e:
            logger.error(
                f"Error generating embeddings for batch {i // batch_size + 1}: {e}"
            )
            raise

    for text, emb_list, emb_np in zip(texts, embeddings_list, embeddings_numpy):
        print(f"Text: {text}")
        print(f"List Embedding (first 5 values): {emb_list[:5]}")
        print(f"List Embedding Type: {type(emb_list)}")
        print(f"Numpy Embedding (first 5 values): {emb_np[:5]}")
        print(f"Numpy Embedding Type: {type(emb_np)}")
        print(f"Length: {len(emb_list)}\n")


if __name__ == "__main__":
    main()
