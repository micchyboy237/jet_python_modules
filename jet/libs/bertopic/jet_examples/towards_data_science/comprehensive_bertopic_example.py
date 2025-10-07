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
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from bertopic import BERTopic

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
import pandas as pd

from jet.file.utils import save_file
from jet.logger import logger
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

_sample_data_cache = None

class BasicWorkflowResult(TypedDict):
    model: BERTopic
    topics: List[int]
    probs: Optional[np.ndarray]
    docs: List[str]
    timestamps: List[str]
    stats: Dict[str, float]

class CustomComponentsResult(TypedDict):
    model: Optional[BERTopic]
    topics: Optional[List[int]]
    probs: Optional[np.ndarray]
    n_topics: int
    outlier_percentage: float

class TopicRepresentationResult(TypedDict):
    original: Dict[int, List[Tuple[str, float]]]
    unigram: List[Tuple[str, float]]
    unigram_bigram: List[Tuple[str, float]]
    unigram_bigram_trigram: List[Tuple[str, float]]
    bigram_only: List[Tuple[str, float]]
    tfidf: List[Tuple[str, float]]

class TopicReductionResult(TypedDict):
    original_count: int
    reduced_count: int
    original_distribution: Dict[int, int]
    reduced_distribution: Dict[int, int]
    reduced_topics: List[Tuple[int, List[Tuple[str, float]]]]

class SimilarTopicsResult(TypedDict):
    term_results: Dict[str, List[Tuple[int, float, List[Tuple[str, float]]]]]

class VisualizationsResult(TypedDict):
    plots: Dict[str, Any]
    status: Dict[str, bool]

class TopicsOverTimeResult(TypedDict):
    topics_time: pd.DataFrame
    trends: pd.DataFrame
    evolutions: Dict[int, Dict]

class SerializationResult(TypedDict):
    original_info: Dict
    saved_info: Dict
    loaded_metadata: Dict
    comparison: Dict

class AdvancedWorkflowResult(TypedDict):
    embeddings_shape: Tuple[int, int]
    original_topics: int
    hierarchical_topics: int
    dynamic_topics: List[Tuple[int, Dict]]

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

def demonstrate_basic_workflow() -> BasicWorkflowResult:
    """Demonstrate the basic BERTopic workflow."""
    sub_dir = os.path.join(OUTPUT_DIR, "basic_workflow")
    shutil.rmtree(sub_dir, ignore_errors=True)
    os.makedirs(sub_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("1. BASIC BERTOPIC WORKFLOW")
    logger.info("=" * 60)
    
    docs, timestamps = create_sample_dataset()
    logger.info("Training basic BERTopic model...")
    
    model, topics, probs = topic_model_fit_transform(
        docs,
        calculate_probabilities=True,
    )
    stats = get_topic_statistics(model, topics)
    
    logger.info(f"Number of topics: {stats['n_topics']}")
    logger.info(f"Number of documents: {stats['n_documents']}")
    logger.info(f"Outlier percentage: {stats['outlier_percentage']:.1f}%")
    logger.info(f"Average documents per topic: {stats['avg_docs_per_topic']:.1f}")
    logger.info("\nTopic Information:")
    logger.info(str(model.get_topic_info()))
    
    save_file(stats, os.path.join(sub_dir, "stats.json"))
    save_file(model.get_topic_info().to_dict(), os.path.join(sub_dir, "topic_info.json"))
    
    return {
        "model": model,
        "topics": topics,
        "probs": probs,
        "docs": docs,
        "timestamps": timestamps,
        "stats": stats
    }

def demonstrate_custom_components() -> CustomComponentsResult:
    """Demonstrate custom UMAP and HDBSCAN configurations."""
    sub_dir = os.path.join(OUTPUT_DIR, "custom_components")
    shutil.rmtree(sub_dir, ignore_errors=True)
    os.makedirs(sub_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("2. CUSTOM COMPONENTS CONFIGURATION")
    logger.info("=" * 60)
    
    docs, _ = create_sample_dataset()
    umap_presets = get_umap_presets()
    hdbscan_presets = get_hdbscan_presets()
    
    logger.info(f"Available UMAP presets: {list(umap_presets.keys())}")
    logger.info(f"Available HDBSCAN presets: {list(hdbscan_presets.keys())}")
    
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
        logger.info(f"\nTesting {config['name']} configuration...")
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
            logger.info(f"  Topics: {n_topics}, Outliers: {outlier_pct:.1f}%")
            save_file(
                model.get_topic_info().to_dict(),
                os.path.join(sub_dir, f"{config['name'].lower()}_topics.json")
            )
        except Exception as e:
            logger.error(f"  Error: {e}")
            results[config['name']] = {"error": str(e)}
    
    best_config = None
    best_score = float('inf')
    for name, result in results.items():
        if "error" not in result:
            score = result["outlier_percentage"] + abs(result["n_topics"] - 8) * 2
            if score < best_score:
                best_score = score
                best_config = name
    
    if best_config:
        logger.info(f"\nBest configuration: {best_config}")
        return results[best_config]
    else:
        logger.warning("No successful configurations found")
        return {
            "model": None,
            "topics": None,
            "probs": None,
            "n_topics": 0,
            "outlier_percentage": 0.0
        }

def demonstrate_topic_representation() -> TopicRepresentationResult:
    """Demonstrate topic representation updates."""
    sub_dir = os.path.join(OUTPUT_DIR, "topic_representation")
    shutil.rmtree(sub_dir, ignore_errors=True)
    os.makedirs(sub_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("3. TOPIC REPRESENTATION UPDATES")
    logger.info("=" * 60)
    
    docs, _ = create_sample_dataset()
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    original_topics = {}
    for topic_id in range(len(model.get_topic_info())):
        original_topics[topic_id] = model.get_topic(topic_id)
    
    logger.info("Original topic representations (first 3 topics):")
    for topic_id in range(min(3, len(original_topics))):
        logger.info(f"Topic {topic_id}: {original_topics[topic_id][:5]}")
    
    n_gram_configs = [
        ((1, 1), "Unigrams only"),
        ((1, 2), "Unigrams and bigrams"),
        ((1, 3), "Unigrams, bigrams, and trigrams"),
        ((2, 2), "Bigrams only")
    ]
    
    result = {
        "original": original_topics,
        "unigram": [],
        "unigram_bigram": [],
        "unigram_bigram_trigram": [],
        "bigram_only": [],
        "tfidf": []
    }
    
    for n_gram_range, description in n_gram_configs:
        logger.info(f"\n{description}:")
        model_updated = update_topic_representation(
            model, docs, topics,
            n_gram_range=n_gram_range,
            stop_words="english"
        )
        topic_words = []
        for topic_id in range(min(2, len(model_updated.get_topic_info()))):
            words = model_updated.get_topic(topic_id)
            topic_words.append(words[:5])
            logger.info(f"  Topic {topic_id}: {words[:5]}")
        result[description.lower().replace(" ", "_")] = topic_words
        save_file(
            topic_words,
            os.path.join(sub_dir, f"{description.lower().replace(' ', '_')}_topics.json")
        )
    
    logger.info("\nUsing TF-IDF vectorizer:")
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
    tfidf_words = []
    for topic_id in range(min(3, len(model_tfidf.get_topic_info()))):
        topic_words = model_tfidf.get_topic(topic_id)
        tfidf_words.append(topic_words[:5])
        logger.info(f"Topic {topic_id}: {topic_words[:5]}")
    
    result["tfidf"] = tfidf_words
    save_file(tfidf_words, os.path.join(sub_dir, "tfidf_topics.json"))
    
    return result

def demonstrate_topic_reduction() -> TopicReductionResult:
    """Demonstrate topic reduction techniques."""
    sub_dir = os.path.join(OUTPUT_DIR, "topic_reduction")
    shutil.rmtree(sub_dir, ignore_errors=True)
    os.makedirs(sub_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("4. TOPIC REDUCTION")
    logger.info("=" * 60)
    
    docs, _ = create_sample_dataset()
    model, topics, probs = topic_model_fit_transform(
        docs,
        calculate_probabilities=True,
        nr_topics="auto"
    )
    
    original_count = len(model.get_topic_info())
    logger.info(f"Original number of topics: {original_count}")
    
    topic_counts = {}
    for topic in topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    logger.info("Original topic distribution:")
    for topic_id, count in sorted(topic_counts.items()):
        logger.info(f"  Topic {topic_id}: {count} documents")
    
    target_topics = 5
    model_reduced, new_topics, new_probs = reduce_topic_count(
        model, docs, nr_topics=target_topics
    )
    
    reduced_count = len(model_reduced.get_topic_info())
    logger.info(f"\nAfter reduction to {target_topics} topics:")
    logger.info(f"New number of topics: {reduced_count}")
    
    new_topic_counts = {}
    for topic in new_topics:
        new_topic_counts[topic] = new_topic_counts.get(topic, 0) + 1
    
    logger.info("New topic distribution:")
    for topic_id, count in sorted(new_topic_counts.items()):
        logger.info(f"  Topic {topic_id}: {count} documents")
    
    logger.info("\nReduced topic details:")
    reduced_topics = []
    for topic_id in range(min(3, len(model_reduced.get_topic_info()))):
        topic_words = model_reduced.get_topic(topic_id)
        reduced_topics.append((topic_id, topic_words[:5]))
        logger.info(f"Topic {topic_id}: {topic_words[:5]}")
    
    save_file(topic_counts, os.path.join(sub_dir, "original_distribution.json"))
    save_file(new_topic_counts, os.path.join(sub_dir, "reduced_distribution.json"))
    save_file(reduced_topics, os.path.join(sub_dir, "reduced_topics.json"))
    
    return {
        "original_count": original_count,
        "reduced_count": reduced_count,
        "original_distribution": topic_counts,
        "reduced_distribution": new_topic_counts,
        "reduced_topics": reduced_topics
    }

def demonstrate_similar_topics() -> SimilarTopicsResult:
    """Demonstrate finding similar topics."""
    sub_dir = os.path.join(OUTPUT_DIR, "similar_topics")
    shutil.rmtree(sub_dir, ignore_errors=True)
    os.makedirs(sub_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("5. SIMILAR TOPIC FINDING")
    logger.info("=" * 60)
    
    docs, _ = create_sample_dataset()
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    search_terms = ["data", "health", "technology", "climate", "economy", "AI"]
    result = {"term_results": {}}
    
    for term in search_terms:
        logger.info(f"\nSearching for topics similar to '{term}':")
        try:
            similar_topics, sim_scores = find_similar_topics(model, term, top_n=3)
            term_result = []
            for topic_id, score in zip(similar_topics, sim_scores):
                topic_words = model.get_topic(topic_id)
                term_result.append((topic_id, score, topic_words[:5]))
                logger.info(f"  Topic {topic_id} (similarity: {score:.3f}): {topic_words[:5]}")
            result["term_results"][term] = term_result
            save_file(term_result, os.path.join(sub_dir, f"{term}_similar_topics.json"))
        except Exception as e:
            logger.error(f"  Error finding similar topics: {e}")
    
    return result

def demonstrate_visualizations() -> VisualizationsResult:
    """Demonstrate BERTopic visualizations."""
    sub_dir = os.path.join(OUTPUT_DIR, "visualizations")
    shutil.rmtree(sub_dir, ignore_errors=True)
    os.makedirs(sub_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("6. VISUALIZATIONS")
    logger.info("=" * 60)
    
    docs, timestamps = create_sample_dataset()
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    topics_time, _ = topics_over_time_analysis(
        model, docs, topics, timestamps,
        datetime_format="%Y-%m-%d"
    )
    
    logger.info("Creating visualizations...")
    plots = visualize_model(
        model,
        topics_over_time=topics_time,
        probs=probs,
        doc_index=0,
        save_plots=True,
        plot_path=sub_dir
    )
    
    status = {}
    logger.info("Available visualizations:")
    for plot_name, plot in plots.items():
        status[plot_name] = plot is not None
        logger.info(f"  {'✓' if plot is not None else '✗'} {plot_name}")
        if plot is not None:
            save_file(plot.to_image(), os.path.join(sub_dir, f"{plot_name}.png"))
    
    logger.info("\nShowing intertopic distance map...")
    show_visualization(plots, "intertopic")
    logger.info("Showing topic bar chart...")
    show_visualization(plots, "barchart")
    
    return {"plots": plots, "status": status}

def demonstrate_topics_over_time() -> TopicsOverTimeResult:
    """Demonstrate topics over time analysis."""
    sub_dir = os.path.join(OUTPUT_DIR, "topics_over_time")
    shutil.rmtree(sub_dir, ignore_errors=True)
    os.makedirs(sub_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("7. TOPICS OVER TIME ANALYSIS")
    logger.info("=" * 60)
    
    docs, timestamps = create_sample_dataset()
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    topics_time, fig = topics_over_time_analysis(
        model, docs, topics, timestamps,
        datetime_format="%Y-%m-%d"
    )
    
    logger.info("Topics over time data (first 10 rows):")
    logger.info(str(topics_time.head(10)))
    save_file(topics_time.to_dict(), os.path.join(sub_dir, "topics_time.json"))
    
    logger.info("\nAnalyzing topic trends...")
    trends = analyze_topic_trends(topics_time, top_n=5)
    logger.info("Top 5 topic trends:")
    logger.info(str(trends))
    save_file(trends.to_dict(), os.path.join(sub_dir, "trends.json"))
    
    evolutions = {}
    if len(topics_time) > 0:
        unique_topics = topics_time['Topic'].unique()
        for topic_id in unique_topics[:3]:
            evolution = get_topic_evolution(topics_time, topic_id)
            evolutions[topic_id] = evolution
            logger.info(f"\nEvolution of topic {topic_id}:")
            for key, value in evolution.items():
                logger.info(f"  {key}: {value}")
            save_file(evolution, os.path.join(sub_dir, f"topic_{topic_id}_evolution.json"))
    
    logger.info("\nSaving topics over time visualization...")
    save_file(fig.to_image(), os.path.join(sub_dir, "visualization.png"))
    
    return {
        "topics_time": topics_time,
        "trends": trends,
        "evolutions": evolutions
    }

def demonstrate_model_serialization() -> SerializationResult:
    """Demonstrate model saving and loading."""
    sub_dir = os.path.join(OUTPUT_DIR, "model_serialization")
    shutil.rmtree(sub_dir, ignore_errors=True)
    os.makedirs(sub_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("8. MODEL SERIALIZATION")
    logger.info("=" * 60)
    
    docs, _ = create_sample_dataset()
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    info = get_model_info(model)
    logger.info("Model information:")
    for key, value in info.items():
        if key != "topic_info":
            logger.info(f"  {key}: {value}")
    save_file(info, os.path.join(sub_dir, "model_info.json"))
    
    save_path = os.path.join(sub_dir, "saved_model")
    logger.info(f"\nSaving model to {save_path}...")
    saved_info = save_topic_model(
        model,
        save_path,
        serialization="safetensors",
        save_metadata=True
    )
    logger.info("Saved model information:")
    for key, value in saved_info.items():
        logger.info(f"  {key}: {value}")
    save_file(saved_info, os.path.join(sub_dir, "saved_info.json"))
    
    logger.info(f"\nLoading model from {save_path}...")
    loaded_model, metadata = load_topic_model(save_path, load_metadata=True)
    logger.info("Loaded model metadata:")
    if metadata:
        logger.info(f"  Saved at: {metadata.get('saved_at', 'unknown')}")
        logger.info(f"  Number of topics: {metadata['model_info']['n_topics']}")
        logger.info(f"  Embedding model: {metadata['model_info']['embedding_model']}")
    save_file(metadata, os.path.join(sub_dir, "loaded_metadata.json"))
    
    logger.info("\nComparing original and loaded models...")
    comparison = compare_models(model, loaded_model)
    logger.info(f"  Number of topics difference: {comparison['n_topics_diff']}")
    logger.info(f"  Embedding models same: {comparison['embedding_models_same']}")
    logger.info(f"  Languages same: {comparison['languages_same']}")
    save_file(comparison, os.path.join(sub_dir, "comparison.json"))
    
    logger.info("\nTesting loaded model...")
    logger.info("Topic information from loaded model:")
    logger.info(str(loaded_model.get_topic_info()))
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        logger.info(f"\nCleaned up saved model directory: {save_path}")
    
    return {
        "original_info": info,
        "saved_info": saved_info,
        "loaded_metadata": metadata,
        "comparison": comparison
    }

def demonstrate_advanced_workflows() -> AdvancedWorkflowResult:
    """Demonstrate advanced BERTopic workflows."""
    sub_dir = os.path.join(OUTPUT_DIR, "advanced_workflows")
    shutil.rmtree(sub_dir, ignore_errors=True)
    os.makedirs(sub_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("9. ADVANCED WORKFLOWS")
    logger.info("=" * 60)
    
    docs, timestamps = create_sample_dataset()
    logger.info("Pre-computing embeddings...")
    embeddings = precompute_embeddings(docs)
    logger.info(f"Embeddings type: {type(embeddings)}")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    model, topics, probs = topic_model_fit_transform(
        docs,
        precomputed_embeddings=embeddings,
        calculate_probabilities=True
    )
    original_topics = len(model.get_topic_info())
    logger.info("Model trained with pre-computed embeddings")
    logger.info(f"Number of topics: {original_topics}")
    
    logger.info("\nDemonstrating hierarchical topic modeling...")
    model_hierarchical, topics_hierarchical, probs_hierarchical = reduce_topic_count(
        model, docs, nr_topics=3
    )
    hierarchical_topics = len(model_hierarchical.get_topic_info())
    logger.info(f"Hierarchical topics: {hierarchical_topics}")
    
    logger.info("Hierarchical topic structure:")
    hierarchical_structure = []
    for topic_id in range(hierarchical_topics):
        topic_words = model_hierarchical.get_topic(topic_id)
        hierarchical_structure.append((topic_id, topic_words[:5]))
        logger.info(f"  Level 1 - Topic {topic_id}: {topic_words[:5]}")
    save_file(hierarchical_structure, os.path.join(sub_dir, "hierarchical_structure.json"))
    
    logger.info("\nAnalyzing topic evolution...")
    topics_time, _ = topics_over_time_analysis(
        model, docs, topics, timestamps,
        datetime_format="%Y-%m-%d"
    )
    topic_evolutions = {}
    for topic_id in topics_time['Topic'].unique():
        evolution = get_topic_evolution(topics_time, topic_id)
        if "error" not in evolution:
            topic_evolutions[topic_id] = evolution
            save_file(evolution, os.path.join(sub_dir, f"topic_{topic_id}_evolution.json"))
    
    logger.info(f"Analyzed evolution for {len(topic_evolutions)} topics")
    dynamic_topics = sorted(
        topic_evolutions.items(),
        key=lambda x: x[1].get('total_periods', 0),
        reverse=True
    )[:3]
    
    logger.info("Most dynamic topics:")
    for topic_id, evolution in dynamic_topics:
        logger.info(f"  Topic {topic_id}: {evolution['total_periods']} periods, trend: {evolution['trend']}")
    
    save_file(
        [(tid, ev) for tid, ev in dynamic_topics],
        os.path.join(sub_dir, "dynamic_topics.json")
    )
    
    return {
        "embeddings_shape": embeddings.shape,
        "original_topics": original_topics,
        "hierarchical_topics": hierarchical_topics,
        "dynamic_topics": dynamic_topics
    }