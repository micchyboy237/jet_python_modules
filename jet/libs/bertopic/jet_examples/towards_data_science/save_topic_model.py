from bertopic import BERTopic
import os
from typing import Optional
import json
from datetime import datetime


def save_topic_model(
    topic_model: BERTopic,
    path: str,
    serialization: str = "safetensors",
    save_embedding_model: bool = True,
    save_ctfidf: bool = True,
    save_metadata: bool = True
) -> dict:
    """
    Save a BERTopic model (and optional components) to disk.
    
    Args:
        topic_model: The BERTopic model to save
        path: Directory path to save the model
        serialization: Serialization format ("safetensors", "pickle", "pytorch")
        save_embedding_model: Whether to save the embedding model
        save_ctfidf: Whether to save the c-TF-IDF model
        save_metadata: Whether to save model metadata
        
    Returns:
        dict: Information about what was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Save the main model
    model_path = os.path.join(path, "bertopic_model")
    topic_model.save(
        model_path,
        serialization=serialization,
        save_embedding_model=save_embedding_model,
        save_ctfidf=save_ctfidf
    )
    
    saved_info = {
        "model_path": model_path,
        "serialization": serialization,
        "save_embedding_model": save_embedding_model,
        "save_ctfidf": save_ctfidf,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save metadata if requested
    if save_metadata:
        metadata = {
            "model_info": {
                "n_topics": len(topic_model.get_topic_info()),
                "embedding_model": str(topic_model.embedding_model),
                "language": getattr(topic_model, 'language', 'unknown')
            },
            "topic_info": topic_model.get_topic_info().to_dict(),
            "saved_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        saved_info["metadata_path"] = metadata_path
    
    return saved_info


def load_topic_model(
    path: str,
    embedding_model: Optional[str] = None,
    load_metadata: bool = True
) -> tuple[BERTopic, dict]:
    """
    Load a BERTopic model. If embedding_model is not saved within the model, supply it.
    
    Args:
        path: Directory path where the model is saved
        embedding_model: Embedding model to use if not saved with the model
        load_metadata: Whether to load metadata if available
        
    Returns:
        tuple: Loaded BERTopic model and metadata dictionary
    """
    model_path = os.path.join(path, "bertopic_model")
    
    # Load the model
    model = BERTopic.load(model_path, embedding_model=embedding_model)
    
    metadata = {}
    if load_metadata:
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
    
    return model, metadata


def get_model_info(topic_model: BERTopic) -> dict:
    """
    Get comprehensive information about a BERTopic model.
    
    Args:
        topic_model: The BERTopic model to analyze
        
    Returns:
        dict: Model information
    """
    topic_info = topic_model.get_topic_info()
    
    info = {
        "n_topics": len(topic_info),
        "topic_info": topic_info.to_dict(),
        "embedding_model": str(topic_model.embedding_model),
        "language": getattr(topic_model, 'language', 'unknown'),
        "has_umap_model": hasattr(topic_model, 'umap_model') and topic_model.umap_model is not None,
        "has_hdbscan_model": hasattr(topic_model, 'hdbscan_model') and topic_model.hdbscan_model is not None,
        "has_ctfidf_model": hasattr(topic_model, 'ctfidf_model') and topic_model.ctfidf_model is not None
    }
    
    return info


def compare_models(model1: BERTopic, model2: BERTopic) -> dict:
    """
    Compare two BERTopic models and return differences.
    
    Args:
        model1: First BERTopic model
        model2: Second BERTopic model
        
    Returns:
        dict: Comparison results
    """
    info1 = get_model_info(model1)
    info2 = get_model_info(model2)
    
    comparison = {
        "n_topics_diff": info1["n_topics"] - info2["n_topics"],
        "embedding_models_same": info1["embedding_model"] == info2["embedding_model"],
        "languages_same": info1["language"] == info2["language"],
        "model1_info": info1,
        "model2_info": info2
    }
    
    return comparison


if __name__ == "__main__":
    from topic_model_fit_transform import topic_model_fit_transform
    
    # Sample documents
    docs = [
        "Machine learning and artificial intelligence are revolutionizing technology.",
        "Data science involves statistics, programming, and domain expertise.",
        "COVID-19 pandemic has changed global health and economy.",
        "Vaccines and medical research are crucial for public health.",
        "Quantum computing could break current encryption methods.",
        "Cryptocurrency and blockchain technology are emerging trends.",
        "Climate change is affecting weather patterns worldwide.",
        "Renewable energy sources like solar and wind are growing.",
        "Stock market volatility affects investor confidence.",
        "Economic policies influence inflation and employment rates."
    ]
    
    print("Fitting BERTopic model...")
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    # Get model information
    print("Model information:")
    info = get_model_info(model)
    for key, value in info.items():
        if key != "topic_info":  # Skip the large topic_info dict
            print(f"  {key}: {value}")
    
    # Save the model
    print("\nSaving model...")
    save_path = "saved_bertopic_model"
    saved_info = save_topic_model(
        model, 
        save_path, 
        serialization="safetensors",
        save_metadata=True
    )
    
    print("Saved model information:")
    for key, value in saved_info.items():
        print(f"  {key}: {value}")
    
    # Load the model
    print("\nLoading model...")
    loaded_model, metadata = load_topic_model(save_path, load_metadata=True)
    
    print("Loaded model metadata:")
    if metadata:
        print(f"  Saved at: {metadata.get('saved_at', 'unknown')}")
        print(f"  Number of topics: {metadata['model_info']['n_topics']}")
        print(f"  Embedding model: {metadata['model_info']['embedding_model']}")
    
    # Compare original and loaded models
    print("\nComparing models...")
    comparison = compare_models(model, loaded_model)
    print(f"  Number of topics difference: {comparison['n_topics_diff']}")
    print(f"  Embedding models same: {comparison['embedding_models_same']}")
    print(f"  Languages same: {comparison['languages_same']}")
    
    # Test that loaded model works
    print("\nTesting loaded model...")
    print("Topic information from loaded model:")
    print(loaded_model.get_topic_info())
    
    # Clean up
    import shutil
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f"\nCleaned up saved model directory: {save_path}")
