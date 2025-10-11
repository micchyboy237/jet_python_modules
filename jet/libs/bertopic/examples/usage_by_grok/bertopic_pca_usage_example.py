from sklearn.decomposition import PCA

def example_pca_in_bertopic() -> np.ndarray:
    """
    Demonstrates using PCA as the dimensionality reduction method in BERTopic.
    - Configures BERTopic with PCA instead of UMAP for the umap_model.
    - Uses SAMPLE_DOCS to fit the model and reduce embeddings.
    """
    pca = PCA(n_components=2)
    model = example_init_bertopic()
    model.umap_model = pca  # Override default UMAP with PCA
    embeddings = model._extract_embeddings(documents=SAMPLE_DOCS)
    reduced_embeddings = model._reduce_dimensionality(embeddings=embeddings, y=[0, 1, 0, 1, 0], partial_fit=False)
    print(f"PCA reduced embeddings shape: {reduced_embeddings.shape}")
    return reduced_embeddings