

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding

def main():
    """Example usage of EmbeddingClient."""
    model = "embeddinggemma-300M-Q8_0.gguf"
    client = LlamacppEmbedding(model=model)
    
    # Example inputs
    texts = ["This is a sample text to generate embeddings.", "Another text for embedding."]
    embeddings_list = client.get_embeddings(texts, return_format="list", show_progress=True)
    embeddings_numpy = client.get_embeddings(texts, return_format="numpy", show_progress=True)
    
    for text, emb_list, emb_np in zip(texts, embeddings_list, embeddings_numpy):
        print(f"Text: {text}")
        print(f"List Embedding (first 5 values): {emb_list[:5]}")
        print(f"List Embedding Type: {type(emb_list)}")
        print(f"Numpy Embedding (first 5 values): {emb_np[:5]}")
        print(f"Numpy Embedding Type: {type(emb_np)}")
        print(f"Length: {len(emb_list)}\n")

if __name__ == "__main__":
    main()