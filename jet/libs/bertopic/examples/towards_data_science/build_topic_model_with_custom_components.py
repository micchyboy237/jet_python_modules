from jet.adapters.bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from typing import Optional, Dict, Any, Tuple, List
import numpy as np


def build_topic_model_with_custom_components(
    docs: list[str],
    embedding_model: str = "embeddinggemma",
    umap_params: dict = None,
    hdbscan_params: dict = None,
    calculate_probabilities: bool = False,
    clustering_model: str = "hdbscan",
    kmeans_params: dict = None
) -> Tuple[BERTopic, List[int], Optional[np.ndarray]]:
    """
    Build a BERTopic model with user-specified UMAP and clustering settings.
    
    Args:
        docs: List of documents to analyze
        embedding_model: Name of the sentence transformer model
        umap_params: Parameters for UMAP dimensionality reduction
        hdbscan_params: Parameters for HDBSCAN clustering
        calculate_probabilities: Whether to calculate topic probabilities
        clustering_model: Type of clustering ("hdbscan" or "kmeans")
        kmeans_params: Parameters for KMeans clustering (if used)
        
    Returns:
        tuple: (topic_model, topics, probabilities)
    """
    # Default UMAP parameters
    if umap_params is None:
        umap_params = {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42
        }
    umap_model = UMAP(**umap_params)

    # Choose clustering model
    if clustering_model.lower() == "kmeans":
        if kmeans_params is None:
            kmeans_params = {
                "n_clusters": 5,
                "random_state": 42,
                "n_init": 10
            }
        clustering_model_obj = KMeans(**kmeans_params)
    else:  # HDBSCAN
        if hdbscan_params is None:
            hdbscan_params = {
                "min_cluster_size": 15,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
                "prediction_data": True
            }
        clustering_model_obj = HDBSCAN(**hdbscan_params)

    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=clustering_model_obj if clustering_model.lower() == "hdbscan" else None,
        calculate_probabilities=calculate_probabilities
    )
    
    # Fit and transform
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs


def get_umap_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get preset UMAP parameter configurations for different use cases.
    
    Returns:
        dict: Dictionary of preset configurations
    """
    presets = {
        "conservative": {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42
        },
        "aggressive": {
            "n_neighbors": 5,
            "n_components": 10,
            "min_dist": 0.1,
            "metric": "euclidean",
            "random_state": 42
        },
        "balanced": {
            "n_neighbors": 10,
            "n_components": 7,
            "min_dist": 0.05,
            "metric": "cosine",
            "random_state": 42
        },
        "high_dimensional": {
            "n_neighbors": 20,
            "n_components": 15,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42
        }
    }
    return presets


def get_hdbscan_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get preset HDBSCAN parameter configurations for different use cases.
    
    Returns:
        dict: Dictionary of preset configurations
    """
    presets = {
        "conservative": {
            "min_cluster_size": 20,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True
        },
        "aggressive": {
            "min_cluster_size": 5,
            "metric": "euclidean",
            "cluster_selection_method": "leaf",
            "prediction_data": True
        },
        "balanced": {
            "min_cluster_size": 10,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True
        },
        "large_datasets": {
            "min_cluster_size": 50,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True
        }
    }
    return presets


def compare_model_configurations(
    docs: list[str],
    configs: List[Dict[str, Any]],
    embedding_model: str = "embeddinggemma"
) -> Dict[str, Any]:
    """
    Compare different model configurations and return results.
    
    Args:
        docs: List of documents
        configs: List of configuration dictionaries
        embedding_model: Embedding model to use
        
    Returns:
        dict: Comparison results
    """
    results = {}
    
    for i, config in enumerate(configs):
        try:
            model, topics, probs = build_topic_model_with_custom_components(
                docs, embedding_model=embedding_model, **config
            )
            
            results[f"config_{i}"] = {
                "model": model,
                "topics": topics,
                "probs": probs,
                "n_topics": len(model.get_topic_info()),
                "outlier_percentage": (topics.count(-1) / len(topics)) * 100,
                "config": config
            }
        except Exception as e:
            results[f"config_{i}"] = {
                "error": str(e),
                "config": config
            }
    
    return results


if __name__ == "__main__":
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
        "Economic policies influence inflation and employment rates.",
        "Deep learning neural networks require large datasets.",
        "Natural language processing is advancing rapidly.",
        "Computer vision applications are expanding in healthcare.",
        "Robotics and automation are transforming manufacturing.",
        "Internet of Things devices are becoming more prevalent.",
        "Machine learning models are being deployed in production systems.",
        "Data privacy regulations are becoming more stringent.",
        "Edge computing is bringing AI closer to devices.",
        "Explainable AI is gaining importance in critical applications.",
        "Federated learning allows training without sharing raw data."
    ]
    
    print("=== Default Configuration ===")
    model_default, topics_default, probs_default = build_topic_model_with_custom_components(
        docs, calculate_probabilities=True
    )
    print(f"Default model - Topics: {len(model_default.get_topic_info())}")
    print(f"Outlier percentage: {(topics_default.count(-1) / len(topics_default)) * 100:.1f}%")
    
    print("\n=== Custom UMAP and HDBSCAN ===")
    custom_umap = {"n_neighbors": 10, "n_components": 7, "min_dist": 0.1, "random_state": 0}
    custom_hdb = {"min_cluster_size": 8, "prediction_data": True}
    
    model_custom, topics_custom, probs_custom = build_topic_model_with_custom_components(
        docs,
        umap_params=custom_umap,
        hdbscan_params=custom_hdb,
        calculate_probabilities=True
    )
    print(f"Custom model - Topics: {len(model_custom.get_topic_info())}")
    print(f"Outlier percentage: {(topics_custom.count(-1) / len(topics_custom)) * 100:.1f}%")
    
    print("\n=== Using KMeans Clustering ===")
    model_kmeans, topics_kmeans, probs_kmeans = build_topic_model_with_custom_components(
        docs,
        clustering_model="kmeans",
        kmeans_params={"n_clusters": 6, "random_state": 42},
        calculate_probabilities=True
    )
    print(f"KMeans model - Topics: {len(model_kmeans.get_topic_info())}")
    print(f"Outlier percentage: {(topics_kmeans.count(-1) / len(topics_kmeans)) * 100:.1f}%")
    
    print("\n=== Comparing Different Presets ===")
    umap_presets = get_umap_presets()
    hdbscan_presets = get_hdbscan_presets()
    
    configs = [
        {"umap_params": umap_presets["conservative"], "hdbscan_params": hdbscan_presets["conservative"]},
        {"umap_params": umap_presets["aggressive"], "hdbscan_params": hdbscan_presets["aggressive"]},
        {"umap_params": umap_presets["balanced"], "hdbscan_params": hdbscan_presets["balanced"]}
    ]
    
    comparison = compare_model_configurations(docs, configs)
    
    print("Configuration comparison:")
    for config_name, result in comparison.items():
        if "error" in result:
            print(f"  {config_name}: Error - {result['error']}")
        else:
            print(f"  {config_name}: {result['n_topics']} topics, {result['outlier_percentage']:.1f}% outliers")
    
    print("\n=== Topic Details from Best Model ===")
    best_model = model_custom
    print("Topic information:")
    print(best_model.get_topic_info())
    
    print("\nTop topics:")
    for topic_id in range(min(3, len(best_model.get_topic_info()))):
        topic_words = best_model.get_topic(topic_id)
        print(f"Topic {topic_id}: {topic_words[:5]}")
