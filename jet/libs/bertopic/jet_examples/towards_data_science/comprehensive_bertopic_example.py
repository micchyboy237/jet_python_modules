"""
Comprehensive BERTopic Example - A Practical Guide Implementation

This script demonstrates all the major BERTopic workflows from "A Practical Guide to BERTopic"
and similar tutorials, wrapped into reusable functions with comprehensive examples.

Features demonstrated:
1. Basic topic modeling with fit/transform
2. Custom UMAP and HDBSCAN configurations
3. Topic representation updates
4. Topic reduction
5. Similar topic finding
6. Visualizations
7. Topics over time analysis
8. Model serialization
9. Advanced configurations and comparisons

Author: Generated for BERTopic examples
Date: 2024
"""

import os
import warnings
from typing import List, Tuple

from jet.logger import logger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import all our custom modules
from jet.libs.bertopic.jet_examples.mock import load_sample_data
from topic_model_fit_transform import (
    topic_model_fit_transform, 
    precompute_embeddings, 
    get_topic_statistics
)
from build_topic_model_with_custom_components import (
    build_topic_model_with_custom_components,
    get_umap_presets,
    get_hdbscan_presets
)
from update_topic_representation import (
    update_topic_representation,
    create_custom_vectorizer
)
from reduce_topic_count import reduce_topic_count
from find_similar_topics import find_similar_topics
from visualize_model import visualize_model, show_visualization
from topics_over_time_analysis import (
    topics_over_time_analysis,
    analyze_topic_trends,
    get_topic_evolution
)
from save_topic_model import (
    save_topic_model,
    load_topic_model,
    get_model_info,
    compare_models
)

from datetime import datetime, timedelta
import numpy as np

_sample_data_cache = None

def create_sample_dataset() -> Tuple[List[str], List[str]]:
    """
    Create a comprehensive sample dataset covering multiple topics and time periods.
    Uses cached data if available and generates reproducible timestamps based on document count.
    Returns:
        tuple: (documents, timestamps)
    """
    global _sample_data_cache
    if _sample_data_cache is not None:
        logger.info(f"Reusing sample data cache ({len(_sample_data_cache)} documents)")
        docs = _sample_data_cache["documents"]
        timestamps = _sample_data_cache["timestamps"]
    else:
        docs = load_sample_data()
        # Set random seed for reproducibility
        np.random.seed(42)
        # Generate timestamps spanning 5 years, proportional to document count
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2025, 12, 31)
        total_days = (end_date - start_date).days
        timestamps = []
        
        for _ in range(len(docs)):
            days_offset = np.random.randint(0, total_days)
            timestamp = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
            timestamps.append(timestamp)
        
        timestamps.sort()  # Sort timestamps chronologically
        _sample_data_cache = {"documents": docs, "timestamps": timestamps}

    return docs, timestamps


def demonstrate_basic_workflow():
    """Demonstrate the basic BERTopic workflow."""
    print("=" * 60)
    print("1. BASIC BERTOPIC WORKFLOW")
    print("=" * 60)
    
    docs, timestamps = create_sample_dataset()
    
    # Basic model training
    print("Training basic BERTopic model...")
    model, topics, probs = topic_model_fit_transform(
        docs, 
        calculate_probabilities=True,
    )
    
    # Get statistics
    stats = get_topic_statistics(model, topics)
    print(f"Number of topics: {stats['n_topics']}")
    print(f"Number of documents: {stats['n_documents']}")
    print(f"Outlier percentage: {stats['outlier_percentage']:.1f}%")
    print(f"Average documents per topic: {stats['avg_docs_per_topic']:.1f}")
    
    # Show topic information
    print("\nTopic Information:")
    print(model.get_topic_info())
    
    return model, topics, probs, docs, timestamps


def demonstrate_custom_components():
    """Demonstrate custom UMAP and HDBSCAN configurations."""
    print("\n" + "=" * 60)
    print("2. CUSTOM COMPONENTS CONFIGURATION")
    print("=" * 60)
    
    docs, _ = create_sample_dataset()
    
    # Get presets
    umap_presets = get_umap_presets()
    hdbscan_presets = get_hdbscan_presets()
    
    print("Available UMAP presets:", list(umap_presets.keys()))
    print("Available HDBSCAN presets:", list(hdbscan_presets.keys()))
    
    # Test different configurations
    configs = [
        {
            "name": "Conservative",
            "umap_params": umap_presets["conservative"],
            "hdbscan_params": hdbscan_presets["conservative"]
        },
        {
            "name": "Aggressive", 
            "umap_params": umap_presets["aggressive"],
            "hdbscan_params": hdbscan_presets["aggressive"]
        },
        {
            "name": "Balanced",
            "umap_params": umap_presets["balanced"],
            "hdbscan_params": hdbscan_presets["balanced"]
        }
    ]
    
    results = {}
    for config in configs:
        print(f"\nTesting {config['name']} configuration...")
        try:
            model, topics, probs = build_topic_model_with_custom_components(
                docs,
                umap_params=config["umap_params"],
                hdbscan_params=config["hdbscan_params"],
                calculate_probabilities=True
            )
            
            n_topics = len(model.get_topic_info())
            outlier_pct = (topics.count(-1) / len(topics)) * 100
            
            results[config['name']] = {
                "model": model,
                "topics": topics,
                "probs": probs,
                "n_topics": n_topics,
                "outlier_percentage": outlier_pct
            }
            
            print(f"  Topics: {n_topics}, Outliers: {outlier_pct:.1f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[config['name']] = {"error": str(e)}
    
    # Choose best configuration
    best_config = None
    best_score = float('inf')
    
    for name, result in results.items():
        if "error" not in result:
            # Simple scoring: prefer fewer outliers, reasonable number of topics
            score = result["outlier_percentage"] + abs(result["n_topics"] - 8) * 2
            if score < best_score:
                best_score = score
                best_config = name
    
    if best_config:
        print(f"\nBest configuration: {best_config}")
        return results[best_config]["model"], results[best_config]["topics"], results[best_config]["probs"]
    else:
        print("No successful configurations found")
        return None, None, None


def demonstrate_topic_representation():
    """Demonstrate topic representation updates."""
    print("\n" + "=" * 60)
    print("3. TOPIC REPRESENTATION UPDATES")
    print("=" * 60)
    
    docs, _ = create_sample_dataset()
    
    # Train initial model
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    # Store original topics
    original_topics = {}
    for topic_id in range(len(model.get_topic_info())):
        original_topics[topic_id] = model.get_topic(topic_id)
    
    print("Original topic representations (first 3 topics):")
    for topic_id in range(min(3, len(original_topics))):
        print(f"Topic {topic_id}: {original_topics[topic_id][:5]}")
    
    # Test different n-gram ranges
    n_gram_configs = [
        ((1, 1), "Unigrams only"),
        ((1, 2), "Unigrams and bigrams"), 
        ((1, 3), "Unigrams, bigrams, and trigrams"),
        ((2, 2), "Bigrams only")
    ]
    
    for n_gram_range, description in n_gram_configs:
        print(f"\n{description}:")
        model_updated = update_topic_representation(
            model, docs, topics,
            n_gram_range=n_gram_range,
            stop_words="english"
        )
        
        for topic_id in range(min(2, len(model_updated.get_topic_info()))):
            topic_words = model_updated.get_topic(topic_id)
            print(f"  Topic {topic_id}: {topic_words[:5]}")
    
    # Test custom vectorizer
    print("\nUsing TF-IDF vectorizer:")
    custom_vectorizer = create_custom_vectorizer(
        vectorizer_type="tfidf",
        n_gram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    
    model_tfidf = update_topic_representation(
        model, docs, topics,
        vectorizer=custom_vectorizer
    )
    
    for topic_id in range(min(3, len(model_tfidf.get_topic_info()))):
        topic_words = model_tfidf.get_topic(topic_id)
        print(f"Topic {topic_id}: {topic_words[:5]}")


def demonstrate_topic_reduction():
    """Demonstrate topic reduction techniques."""
    print("\n" + "=" * 60)
    print("4. TOPIC REDUCTION")
    print("=" * 60)
    
    docs, _ = create_sample_dataset()
    
    # Train model with many topics
    model, topics, probs = topic_model_fit_transform(
        docs, 
        calculate_probabilities=True,
        nr_topics="auto"  # Let it find many topics
    )
    
    print(f"Original number of topics: {len(model.get_topic_info())}")
    print("Original topic distribution:")
    topic_counts = {}
    for topic in topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    for topic_id, count in sorted(topic_counts.items()):
        print(f"  Topic {topic_id}: {count} documents")
    
    # Reduce topics
    target_topics = 5
    model_reduced, new_topics, new_probs = reduce_topic_count(
        model, docs, nr_topics=target_topics
    )
    
    print(f"\nAfter reduction to {target_topics} topics:")
    print(f"New number of topics: {len(model_reduced.get_topic_info())}")
    
    print("New topic distribution:")
    new_topic_counts = {}
    for topic in new_topics:
        new_topic_counts[topic] = new_topic_counts.get(topic, 0) + 1
    
    for topic_id, count in sorted(new_topic_counts.items()):
        print(f"  Topic {topic_id}: {count} documents")
    
    # Show topic details
    print("\nReduced topic details:")
    for topic_id in range(min(3, len(model_reduced.get_topic_info()))):
        topic_words = model_reduced.get_topic(topic_id)
        print(f"Topic {topic_id}: {topic_words[:5]}")


def demonstrate_similar_topics():
    """Demonstrate finding similar topics."""
    print("\n" + "=" * 60)
    print("5. SIMILAR TOPIC FINDING")
    print("=" * 60)
    
    docs, _ = create_sample_dataset()
    
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    # Test different search terms
    search_terms = ["data", "health", "technology", "climate", "economy", "AI"]
    
    for term in search_terms:
        print(f"\nSearching for topics similar to '{term}':")
        try:
            similar_topics, sim_scores = find_similar_topics(model, term, top_n=3)
            
            for topic_id, score in zip(similar_topics, sim_scores):
                topic_words = model.get_topic(topic_id)
                print(f"  Topic {topic_id} (similarity: {score:.3f}): {topic_words[:5]}")
        except Exception as e:
            print(f"  Error finding similar topics: {e}")


def demonstrate_visualizations():
    """Demonstrate BERTopic visualizations."""
    print("\n" + "=" * 60)
    print("6. VISUALIZATIONS")
    print("=" * 60)
    
    docs, timestamps = create_sample_dataset()
    
    # Train model
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    # Get topics over time
    topics_time, _ = topics_over_time_analysis(
        model, docs, topics, timestamps, 
        datetime_format="%Y-%m-%d"
    )
    
    print("Creating visualizations...")
    
    # Create all visualizations
    plots = visualize_model(
        model,
        topics_over_time=topics_time,
        probs=probs,
        doc_index=0,
        save_plots=True,
        plot_path="bertopic_plots"
    )
    
    print("Available visualizations:")
    for plot_name, plot in plots.items():
        if plot is not None:
            print(f"  ✓ {plot_name}")
        else:
            print(f"  ✗ {plot_name} (failed)")
    
    # Show specific visualizations
    print("\nShowing intertopic distance map...")
    show_visualization(plots, "intertopic")
    
    print("Showing topic bar chart...")
    show_visualization(plots, "barchart")


def demonstrate_topics_over_time():
    """Demonstrate topics over time analysis."""
    print("\n" + "=" * 60)
    print("7. TOPICS OVER TIME ANALYSIS")
    print("=" * 60)
    
    docs, timestamps = create_sample_dataset()
    
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    # Analyze topics over time
    topics_time, fig = topics_over_time_analysis(
        model, docs, topics, timestamps,
        datetime_format="%Y-%m-%d"
    )
    
    print("Topics over time data (first 10 rows):")
    print(topics_time.head(10))
    
    # Analyze trends
    print("\nAnalyzing topic trends...")
    trends = analyze_topic_trends(topics_time, top_n=5)
    print("Top 5 topic trends:")
    print(trends)
    
    # Analyze evolution of specific topics
    if len(topics_time) > 0:
        unique_topics = topics_time['Topic'].unique()
        for topic_id in unique_topics[:3]:  # Analyze first 3 topics
            evolution = get_topic_evolution(topics_time, topic_id)
            print(f"\nEvolution of topic {topic_id}:")
            for key, value in evolution.items():
                print(f"  {key}: {value}")
    
    # Show visualization
    print("\nShowing topics over time visualization...")
    fig.show()


def demonstrate_model_serialization():
    """Demonstrate model saving and loading."""
    print("\n" + "=" * 60)
    print("8. MODEL SERIALIZATION")
    print("=" * 60)
    
    docs, _ = create_sample_dataset()
    
    # Train model
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    # Get model information
    info = get_model_info(model)
    print("Model information:")
    for key, value in info.items():
        if key != "topic_info":
            print(f"  {key}: {value}")
    
    # Save model
    save_path = "saved_bertopic_model"
    print(f"\nSaving model to {save_path}...")
    
    saved_info = save_topic_model(
        model,
        save_path,
        serialization="safetensors",
        save_metadata=True
    )
    
    print("Saved model information:")
    for key, value in saved_info.items():
        print(f"  {key}: {value}")
    
    # Load model
    print(f"\nLoading model from {save_path}...")
    loaded_model, metadata = load_topic_model(save_path, load_metadata=True)
    
    print("Loaded model metadata:")
    if metadata:
        print(f"  Saved at: {metadata.get('saved_at', 'unknown')}")
        print(f"  Number of topics: {metadata['model_info']['n_topics']}")
        print(f"  Embedding model: {metadata['model_info']['embedding_model']}")
    
    # Compare models
    print("\nComparing original and loaded models...")
    comparison = compare_models(model, loaded_model)
    print(f"  Number of topics difference: {comparison['n_topics_diff']}")
    print(f"  Embedding models same: {comparison['embedding_models_same']}")
    print(f"  Languages same: {comparison['languages_same']}")
    
    # Test loaded model
    print("\nTesting loaded model...")
    print("Topic information from loaded model:")
    print(loaded_model.get_topic_info())
    
    # Clean up
    import shutil
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f"\nCleaned up saved model directory: {save_path}")


def demonstrate_advanced_workflows():
    """Demonstrate advanced BERTopic workflows."""
    print("\n" + "=" * 60)
    print("9. ADVANCED WORKFLOWS")
    print("=" * 60)
    
    docs, timestamps = create_sample_dataset()
    
    # Pre-compute embeddings for efficiency
    print("Pre-computing embeddings...")
    embeddings = precompute_embeddings(docs)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Use pre-computed embeddings
    model, topics, probs = topic_model_fit_transform(
        docs,
        precomputed_embeddings=embeddings,
        calculate_probabilities=True
    )
    
    print("Model trained with pre-computed embeddings")
    print(f"Number of topics: {len(model.get_topic_info())}")
    
    # Demonstrate hierarchical topic modeling
    print("\nDemonstrating hierarchical topic modeling...")
    
    # Create a hierarchical structure by reducing topics step by step
    original_topics = len(model.get_topic_info())
    print(f"Original topics: {original_topics}")
    
    # Reduce to fewer topics
    model_hierarchical, topics_hierarchical, probs_hierarchical = reduce_topic_count(
        model, docs, topics, probs, nr_topics=3
    )
    
    print(f"Hierarchical topics: {len(model_hierarchical.get_topic_info())}")
    
    # Show hierarchical topic structure
    print("Hierarchical topic structure:")
    for topic_id in range(len(model_hierarchical.get_topic_info())):
        topic_words = model_hierarchical.get_topic(topic_id)
        print(f"  Level 1 - Topic {topic_id}: {topic_words[:5]}")
    
    # Demonstrate topic evolution analysis
    print("\nAnalyzing topic evolution...")
    topics_time, _ = topics_over_time_analysis(
        model, docs, topics, timestamps,
        datetime_format="%Y-%m-%d"
    )
    
    # Find topics with significant changes
    topic_evolutions = {}
    for topic_id in topics_time['Topic'].unique():
        evolution = get_topic_evolution(topics_time, topic_id)
        if "error" not in evolution:
            topic_evolutions[topic_id] = evolution
    
    print(f"Analyzed evolution for {len(topic_evolutions)} topics")
    
    # Show most dynamic topics
    dynamic_topics = sorted(
        topic_evolutions.items(),
        key=lambda x: x[1].get('total_periods', 0),
        reverse=True
    )[:3]
    
    print("Most dynamic topics:")
    for topic_id, evolution in dynamic_topics:
        print(f"  Topic {topic_id}: {evolution['total_periods']} periods, trend: {evolution['trend']}")


def main():
    """Run the comprehensive BERTopic demonstration."""
    print("COMPREHENSIVE BERTOPIC EXAMPLE")
    print("A Practical Guide to BERTopic Implementation")
    print("=" * 60)
    
    try:
        # 1. Basic workflow
        model, topics, probs, docs, timestamps = demonstrate_basic_workflow()
        
        # 2. Custom components
        demonstrate_custom_components()
        
        # 3. Topic representation
        demonstrate_topic_representation()
        
        # 4. Topic reduction
        demonstrate_topic_reduction()
        
        # 5. Similar topics
        demonstrate_similar_topics()
        
        # 6. Visualizations
        demonstrate_visualizations()
        
        # 7. Topics over time
        demonstrate_topics_over_time()
        
        # 8. Model serialization
        demonstrate_model_serialization()
        
        # 9. Advanced workflows
        demonstrate_advanced_workflows()
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE DEMONSTRATION COMPLETED")
        print("=" * 60)
        print("All BERTopic workflows have been demonstrated successfully!")
        print("Check the 'bertopic_plots' directory for saved visualizations.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
