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
import shutil
import warnings
from typing import TypedDict

from jet.file.utils import save_file
from jet.logger import CustomLogger
from jet.utils.text import format_file_path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import all our custom modules
from build_topic_model_with_custom_components import (
    build_topic_model_with_custom_components,
    get_hdbscan_presets,
    get_umap_presets,
)
from find_similar_topics import find_similar_topics
from jet.libs.bertopic.examples.mock import load_sample_data
from reduce_topic_count import reduce_topic_count
from save_topic_model import (
    compare_models,
    get_model_info,
    load_topic_model,
    save_topic_model,
)
from topic_model_fit_transform import (
    get_topic_statistics,
    precompute_embeddings,
    topic_model_fit_transform,
)
from topics_over_time_analysis import (
    analyze_topic_trends,
    get_topic_evolution,
    topics_over_time_analysis,
)
from update_topic_representation import (
    create_custom_vectorizer,
    update_topic_representation,
)
from visualize_model import visualize_model


class TopicCategory(TypedDict):
    label: int
    category: str
    representation: list[str]
    count: int


class TopicEntry(TopicCategory):
    doc_index: int
    text: str  # Original chunk text


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)

CHUNK_SIZE = 128
CHUNK_OVERLAP = 32
EMBEDDING_MODEL = "nomic-embed-text"

OUTPUT_DIR = f"{OUTPUT_DIR}/chunked_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")

logger = CustomLogger(__name__)
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

_sample_data_cache = None


def create_sample_dataset() -> list[str]:
    """
    Create a comprehensive sample dataset covering multiple topics.
    Uses cached data if available.
    Returns:
        documents (list): The sample documents.
    """
    global _sample_data_cache
    if _sample_data_cache is not None:
        logger.info(f"Reusing sample data cache ({len(_sample_data_cache)} documents)")
        docs = _sample_data_cache["documents"]
    else:
        docs = load_sample_data(EMBEDDING_MODEL)
        _sample_data_cache = {"documents": docs}
    return docs


def _01_demonstrate_basic_workflow():
    """Demonstrate the basic BERTopic workflow."""
    logger = CustomLogger("01_basic_workflow")
    output_dir = f"{OUTPUT_DIR}/01_basic_workflow"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/main.log"
    logger.basicConfig(filename=log_file)
    logger.orange(f"Log: {log_file}")

    logger.gray("=" * 60)
    logger.orange("1. BASIC BERTOPIC WORKFLOW")
    logger.gray("=" * 60)

    docs = create_sample_dataset()

    # Basic model training
    logger.info("Training basic BERTopic model...")
    model, topics, probs = topic_model_fit_transform(
        docs,
        calculate_probabilities=True,
    )
    save_file(topics, f"{output_dir}/01_topics.json")
    save_file(probs, f"{output_dir}/01_probs.json")

    # Get statistics
    stats = get_topic_statistics(model, topics)
    topic_info = stats.pop("topic_info")
    logger.success(f"Number of topics: {stats['n_topics']}")
    logger.success(f"Number of documents: {stats['n_documents']}")
    logger.success(f"Outlier percentage: {stats['outlier_percentage']:.1f}%")
    logger.success(f"Average documents per topic: {stats['avg_docs_per_topic']:.1f}")
    save_file(stats, f"{output_dir}/01_stats.json")

    # Show topic information
    logger.debug("\nTopic Information:")
    logger.success(topic_info)
    save_file(topic_info.to_dict(orient="records"), f"{output_dir}/01_topic_info.json")


def _02_demonstrate_hierarchical_topics():
    """Demonstrate hierarchical topic modeling with BERTopic."""
    logger = CustomLogger("02_hierarchical_topics")
    output_dir = f"{OUTPUT_DIR}/02_hierarchical_topics"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/main.log"
    logger.basicConfig(filename=log_file)
    logger.orange(f"Log: {log_file}")

    logger.gray("\n" + "=" * 60)
    logger.orange("2. HIERARCHICAL TOPIC MODELING")
    logger.gray("=" * 60)

    docs = create_sample_dataset()

    # Train initial model
    logger.info("Training BERTopic model for hierarchical analysis...")
    model, topics, probs = topic_model_fit_transform(
        docs,
        calculate_probabilities=True,
    )
    save_file(topics, f"{output_dir}/02_topics.json")
    save_file(probs, f"{output_dir}/02_probs.json")
    save_file(
        model.get_topic_info().to_dict(orient="records"),
        f"{output_dir}/02_topic_info.json",
    )

    # Generate hierarchical topics
    logger.info("Generating hierarchical topic structure...")
    try:
        hierarchical_topics = model.hierarchical_topics(docs=docs)
        logger.success(
            f"Hierarchical topics generated with {len(hierarchical_topics)} parent topics"
        )

        # Log and save hierarchical structure
        logger.debug("Hierarchical topic structure (first 5 entries):")
        for _, row in hierarchical_topics.head(5).iterrows():
            logger.debug(
                f"Parent {row['Parent_ID']}: {row['Parent_Name']} "
                f"(Topics: {row['Topics']}, Distance: {row['Distance']:.3f})"
            )
        save_file(
            hierarchical_topics.to_dict(orient="records"),
            f"{output_dir}/02_hierarchical_topics.json",
        )

        # Summarize hierarchy
        hierarchy_summary = {
            "total_parent_topics": len(hierarchical_topics),
            "max_depth": hierarchical_topics["Distance"].max()
            if not hierarchical_topics.empty
            else 0,
            "avg_child_topics": hierarchical_topics["Topics"].apply(len).mean()
            if not hierarchical_topics.empty
            else 0,
        }
        logger.info("Hierarchy summary:")
        for key, value in hierarchy_summary.items():
            logger.debug(
                f" {key}: {value:.2f}"
                if isinstance(value, float)
                else f" {key}: {value}"
            )
        save_file(hierarchy_summary, f"{output_dir}/02_hierarchy_summary.json")

        tree = model.get_topic_tree(hierarchical_topics, tight_layout=False)
        save_file(tree, f"{output_dir}/02_hierarchy_tree.txt")

    except Exception as e:
        logger.error(f"Error generating hierarchical topics: {e}")
        return

    # Example of accessing a specific parent topic
    if not hierarchical_topics.empty:
        example_parent = hierarchical_topics.iloc[0]
        logger.debug(
            f"\nExample parent topic analysis (Parent {example_parent['Parent_ID']}):"
        )
        logger.debug(f" Name: {example_parent['Parent_Name']}")
        logger.debug(f" Child Topics: {example_parent['Topics']}")
        logger.debug(f" Distance: {example_parent['Distance']:.3f}")

        # Save example parent topic details
        save_file(
            example_parent.to_dict(), f"{output_dir}/02_example_parent_topic.json"
        )


def _03_demonstrate_custom_components():
    """Demonstrate custom UMAP and HDBSCAN configurations."""
    logger = CustomLogger("03_custom_components")
    output_dir = f"{OUTPUT_DIR}/03_custom_components"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/main.log"
    logger.basicConfig(filename=log_file)
    logger.orange(f"Log: {log_file}")

    logger.gray("\n" + "=" * 60)
    logger.orange("3. CUSTOM COMPONENTS CONFIGURATION")
    logger.gray("=" * 60)

    docs = create_sample_dataset()

    # Get presets
    umap_presets = get_umap_presets()
    hdbscan_presets = get_hdbscan_presets()

    logger.info("Available UMAP presets:", list(umap_presets.keys()))
    logger.info("Available HDBSCAN presets:", list(hdbscan_presets.keys()))

    # Test different configurations
    configs = [
        {
            "name": "Conservative",
            "umap_params": umap_presets["conservative"],
            "hdbscan_params": hdbscan_presets["conservative"],
        },
        {
            "name": "Aggressive",
            "umap_params": umap_presets["aggressive"],
            "hdbscan_params": hdbscan_presets["aggressive"],
        },
        {
            "name": "Balanced",
            "umap_params": umap_presets["balanced"],
            "hdbscan_params": hdbscan_presets["balanced"],
        },
    ]

    results = {}
    all_topic_entries = {}  # Store topic entries per config
    for config in configs:
        logger.debug(f"\nTesting {config['name']} configuration...")
        try:
            model, topics, probs = build_topic_model_with_custom_components(
                docs,
                embedding_model=EMBEDDING_MODEL,
                umap_params=config["umap_params"],
                hdbscan_params=config["hdbscan_params"],
                calculate_probabilities=True,
            )

            n_topics = len(model.get_topic_info())
            outlier_pct = (topics.count(-1) / len(topics)) * 100

            results[config["name"]] = {
                "model": model,
                "topics": topics,
                "probs": probs,
                "n_topics": n_topics,
                "outlier_percentage": outlier_pct,
            }

            logger.success(f"  Topics: {n_topics}, Outliers: {outlier_pct:.1f}%")

            topic_info = model.get_topic_info()
            save_file(
                topic_info.to_dict(orient="records"),
                f"{output_dir}/{format_file_path(config['name'])}/03_topic_info.json",
            )

            config_topic_entries = []
            for rank, topic_row in enumerate(topic_info.itertuples(), start=1):
                # Safely access columns by index for a DataFrame row tuple
                # topic_info columns are ['Topic', 'Name', 'Count', ...]
                topic_id = int(getattr(topic_row, "Topic", -1))
                category_name = str(getattr(topic_row, "Name", f"Topic {topic_id}"))
                representation = getattr(topic_row, "Representation", [])
                count = float(getattr(topic_row, "Count", 0))
                doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
                doc_index = doc_indices[0] if doc_indices else -1
                text = docs[doc_index] if doc_index >= 0 else ""
                topic_entry: TopicEntry = {
                    "doc_index": doc_index,
                    "label": topic_id,
                    "count": count,
                    "category": category_name,
                    "representation": representation,
                    "text": text,
                }
                config_topic_entries.append(topic_entry)
                topic_id_suffix = topic_id if topic_id != -1 else "outliers"
                save_file(
                    {
                        **config,
                        "topic": topic_entry,
                    },
                    f"{output_dir}/{format_file_path(config['name'])}/03_config_topics_{topic_id_suffix}.json",
                )
            all_topic_entries[config["name"]] = config_topic_entries

        except Exception as e:
            logger.error(f"  Error: {e}")
            results[config["name"]] = {"error": str(e)}

    # Choose best configuration
    best_config = None
    best_score = float("inf")

    for name, result in results.items():
        if "error" not in result:
            # Simple scoring: prefer fewer outliers, reasonable number of topics
            score = result["outlier_percentage"] + abs(result["n_topics"] - 8) * 2
            if score < best_score:
                best_score = score
                best_config = name

    if best_config:
        logger.success(f"\nBest configuration: {best_config}")
        # Save the best config and all its topics in a separate file
        best_config_topics = all_topic_entries.get(best_config, [])
        save_file(
            {
                "config": configs[[cfg["name"] for cfg in configs].index(best_config)],
                "topics": best_config_topics,
            },
            f"{output_dir}/03_best_config_and_topics.json",
        )
        return (
            results[best_config]["model"],
            results[best_config]["topics"],
            results[best_config]["probs"],
        )
    else:
        logger.warning("No successful configurations found")
        return None, None, None


def _04_demonstrate_topic_representation():
    """Demonstrate topic representation updates."""
    logger = CustomLogger("04_topic_representation")
    output_dir = f"{OUTPUT_DIR}/04_topic_representation"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/main.log"
    logger.basicConfig(filename=log_file)
    logger.orange(f"Log: {log_file}")

    logger.gray("\n" + "=" * 60)
    logger.orange("4. TOPIC REPRESENTATION UPDATES")
    logger.gray("=" * 60)

    docs = create_sample_dataset()

    # Train initial model
    model, topics, probs = topic_model_fit_transform(
        docs, embedding_model=EMBEDDING_MODEL, calculate_probabilities=True
    )

    # Store original topics
    original_topics = {}
    for topic_id in range(len(model.get_topic_info())):
        original_topics[topic_id] = model.get_topic(topic_id)
    save_file(original_topics, f"{output_dir}/04_original_topics.json")

    logger.info("Original topic representations (first 3 topics):")
    for topic_id in range(min(3, len(original_topics))):
        logger.success(
            f"Topic {topic_id}: {original_topics[topic_id][:5] if original_topics[topic_id] else 'None'}"
        )

    # Test different n-gram ranges
    n_gram_configs = [
        ((1, 1), "Unigrams only"),
        ((1, 2), "Unigrams and bigrams"),
        ((1, 3), "Unigrams, bigrams, and trigrams"),
        ((2, 2), "Bigrams only"),
    ]

    for n_gram_range, description in n_gram_configs:
        logger.debug(f"\n{description}:")
        model_updated = update_topic_representation(
            model, docs, topics, n_gram_range=n_gram_range, stop_words="english"
        )

        for topic_id in range(min(2, len(model_updated.get_topic_info()))):
            topic_words = model_updated.get_topic(topic_id)
            if not topic_words:
                topic_words = []
            logger.success(f"  Topic {topic_id}: {topic_words[:5]}")
            save_file(
                {
                    "n_gram_range": n_gram_range,
                    "description": description,
                    "count": len(topic_words),
                    "topics": topic_words,
                },
                f"{output_dir}/04_ngram/top_topics_{topic_id}_n_{n_gram_range[0]}_{n_gram_range[1]}.json",
            )

    # Test custom vectorizer
    logger.info("\nUsing TF-IDF vectorizer:")
    custom_vectorizer = create_custom_vectorizer(
        vectorizer_type="tfidf", n_gram_range=(1, 2), min_df=1, max_df=0.9
    )

    model_tfidf = update_topic_representation(
        model, docs, topics, vectorizer=custom_vectorizer
    )

    for topic_id in range(min(3, len(model_tfidf.get_topic_info()))):
        topic_words = model_tfidf.get_topic(topic_id)
        if not topic_words:
            topic_words = []
        logger.success(f"Topic {topic_id}: {topic_words[:5]}")
        save_file(topic_words, f"{output_dir}/04_tfidf/top_topics_{topic_id}.json")
        save_file(
            {
                "count": len(topic_words),
                "topics": topic_words,
            },
            f"{output_dir}/04_tfidf/top_topics_{topic_id}.json",
        )


def _05_demonstrate_topic_reduction():
    """Demonstrate topic reduction techniques."""
    logger = CustomLogger("05_topic_reduction")
    output_dir = f"{OUTPUT_DIR}/05_topic_reduction"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/main.log"
    logger.basicConfig(filename=log_file)
    logger.orange(f"Log: {log_file}")

    logger.gray("\n" + "=" * 60)
    logger.orange("5. TOPIC REDUCTION")
    logger.gray("=" * 60)

    docs = create_sample_dataset()

    # Train model with many topics
    model, topics, probs = topic_model_fit_transform(
        docs,
        embedding_model=EMBEDDING_MODEL,
        calculate_probabilities=True,
        nr_topics="auto",  # Let it find many topics
    )

    logger.info(f"Original number of topics: {len(model.get_topic_info())}")
    logger.info("Original topic distribution:")
    topic_counts = {}
    for topic in topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    for topic_id, count in sorted(topic_counts.items()):
        logger.debug(f"  Topic {topic_id}: {count} documents")

    save_file(topic_counts, f"{output_dir}/05_orig/topic_counts.json")
    save_file(topics, f"{output_dir}/05_orig/topics.json")
    save_file(probs, f"{output_dir}/05_orig/probs.json")
    save_file(
        model.get_topic_info().to_dict(orient="records"),
        f"{output_dir}/05_orig/topic_info.json",
    )

    # Reduce topics
    target_topics = 5
    model_reduced, new_topics, new_probs = reduce_topic_count(
        model, docs, nr_topics=target_topics
    )

    logger.info(f"\nAfter reduction to {target_topics} topics:")
    logger.info(f"New number of topics: {len(model_reduced.get_topic_info())}")

    logger.info("New topic distribution:")
    new_topic_counts = {}
    for topic in new_topics:
        new_topic_counts[topic] = new_topic_counts.get(topic, 0) + 1

    for topic_id, count in sorted(new_topic_counts.items()):
        logger.debug(f"  Topic {topic_id}: {count} documents")

    # Show topic details
    logger.info("\nReduced topic details:")
    for topic_id in range(min(3, len(model_reduced.get_topic_info()))):
        topic_words = model_reduced.get_topic(topic_id)
        if not topic_words:
            topic_words = []
        logger.debug(f"Topic {topic_id}: {topic_words[:5]}")

    save_file(topic_counts, f"{output_dir}/05_reduced/topic_counts.json")
    save_file(new_topics, f"{output_dir}/05_reduced/topics.json")
    save_file(new_probs, f"{output_dir}/05_reduced/probs.json")
    save_file(
        model_reduced.get_topic_info().to_dict(orient="records"),
        f"{output_dir}/05_reduced/topic_info.json",
    )


def _06_demonstrate_similar_topics():
    """Demonstrate finding similar topics."""
    logger = CustomLogger("06_similar_topics")
    output_dir = f"{OUTPUT_DIR}/06_similar_topics"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/main.log"
    logger.basicConfig(filename=log_file)
    logger.orange(f"Log: {log_file}")

    logger.gray("\n" + "=" * 60)
    logger.orange("6. SIMILAR TOPIC FINDING")
    logger.gray("=" * 60)

    docs = create_sample_dataset()

    model, topics, probs = topic_model_fit_transform(
        docs, embedding_model=EMBEDDING_MODEL, calculate_probabilities=True
    )

    # Test different search terms
    search_terms = ["isekai", "anime", "2025"]
    top_n = 10

    class SearchResult(TypedDict):
        rank: int
        score: float
        doc_index: int
        category: TopicCategory
        text: str

    all_search_results: dict[str, list[SearchResult]] = {}

    for term in search_terms:
        logger.debug(f"\nSearching for topics similar to '{term}':")
        search_results: list[SearchResult] = []
        try:
            similar_topics, sim_scores = find_similar_topics(model, term, top_n=top_n)
            for rank, (topic_id, score) in enumerate(zip(similar_topics, sim_scores)):
                topic_info = model.get_topic_info()
                # Get the first doc_index (if any) for this topic
                # Find a document index with this topic
                doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
                doc_index = doc_indices[0] if doc_indices else -1
                # Get topic label/category if available in topic_info
                topic_row = topic_info[topic_info["Topic"] == topic_id]
                category: TopicCategory = {
                    "label": int(topic_id),
                    "category": str(topic_row.iloc[0]["Name"])
                    if not topic_row.empty and "Name" in topic_row
                    else f"Topic {topic_id}",
                }
                # Get the original document text
                doc_text = docs[doc_index] if doc_index >= 0 else ""
                # Log success and also append the result
                topic_words = model.get_topic(topic_id)
                logger.success(
                    f"  Topic {topic_id} (similarity: {score:.3f}): {topic_words[:5]}"
                )
                search_results.append(
                    {
                        "rank": rank,
                        "score": float(score),
                        "doc_index": int(doc_index),
                        "category": category,
                        "text": doc_text,  # Added text field to SearchResult
                    }
                )
        except Exception as e:
            logger.error(f"  Error finding similar topics: {e}")
        all_search_results[term] = search_results

    save_file(search_terms, f"{output_dir}/06_terms.json")
    for term, results in all_search_results.items():
        term_num = f"{search_terms.index(term) + 1:02d}"
        save_file(
            {"term": term, "count": len(results), "results": results},
            f"{output_dir}/06_search_results/{term_num}_{format_file_path(term)}.json",
        )


def _07_demonstrate_visualizations():
    """Demonstrate BERTopic visualizations."""
    logger = CustomLogger("07_visualizations")
    output_dir = f"{OUTPUT_DIR}/07_visualizations"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/main.log"
    logger.basicConfig(filename=log_file)
    logger.orange(f"Log: {log_file}")

    logger.gray("\n" + "=" * 60)
    logger.orange("7. VISUALIZATIONS")
    logger.gray("=" * 60)

    docs = create_sample_dataset()

    # Train model
    model, topics, probs = topic_model_fit_transform(
        docs, embedding_model=EMBEDDING_MODEL, calculate_probabilities=True
    )

    save_file(topics, f"{output_dir}/07_topics.json")
    save_file(probs, f"{output_dir}/07_probs.json")
    save_file(
        model.get_topic_info().to_dict(orient="records"),
        f"{output_dir}/07_topic_info.json",
    )

    logger.debug("Creating visualizations...")

    # Create all visualizations
    plots = visualize_model(
        model,
        # topics_over_time=topics_time,
        probs=probs,
        doc_index=0,
        save_plots=True,
        plot_path=f"{output_dir}/07_plots",
    )

    logger.debug("Available visualizations:")
    for plot_name, plot in plots.items():
        if plot is not None:
            logger.debug(f"  ✓ {plot_name}")
        else:
            logger.debug(f"  ✗ {plot_name} (failed)")


def _08_demonstrate_topics_over_time():
    """Demonstrate topics over time analysis."""
    logger = CustomLogger("08_topics_over_time")
    output_dir = f"{OUTPUT_DIR}/08_topics_over_time"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/main.log"
    logger.basicConfig(filename=log_file)
    logger.orange(f"Log: {log_file}")

    logger.gray("\n" + "=" * 60)
    logger.orange("8. TOPICS OVER TIME ANALYSIS")
    logger.gray("=" * 60)

    docs, timestamps = create_sample_dataset()

    model, topics, probs = topic_model_fit_transform(
        docs, embedding_model=EMBEDDING_MODEL, calculate_probabilities=True
    )

    # Analyze topics over time
    topics_time, fig = topics_over_time_analysis(
        model, docs, topics, timestamps, datetime_format="%Y-%m-%d"
    )

    logger.debug("Topics over time data (first 10 rows):")
    logger.debug(topics_time.head(10))

    # Analyze trends
    logger.debug("\nAnalyzing topic trends...")
    trends = analyze_topic_trends(topics_time, top_n=5)
    logger.debug("Top 5 topic trends:")
    logger.debug(trends)

    # Analyze evolution of specific topics
    if len(topics_time) > 0:
        unique_topics = topics_time["Topic"].unique()
        for topic_id in unique_topics[:3]:  # Analyze first 3 topics
            evolution = get_topic_evolution(topics_time, topic_id)
            logger.debug(f"\nEvolution of topic {topic_id}:")
            for key, value in evolution.items():
                logger.debug(f"  {key}: {value}")

    # Show visualization
    logger.debug("\nShowing topics over time visualization...")
    fig.show()


def _09_demonstrate_model_serialization():
    """Demonstrate model saving and loading."""
    logger = CustomLogger("09_model_serialization")
    output_dir = f"{OUTPUT_DIR}/09_model_serialization"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/main.log"
    logger.basicConfig(filename=log_file)
    logger.orange(f"Log: {log_file}")

    logger.gray("\n" + "=" * 60)
    logger.orange("9. MODEL SERIALIZATION")
    logger.gray("=" * 60)

    docs = create_sample_dataset()

    # Train model
    model, topics, probs = topic_model_fit_transform(
        docs, embedding_model=EMBEDDING_MODEL, calculate_probabilities=True
    )

    # Get model information
    info = get_model_info(model)
    logger.debug("Model information:")
    for key, value in info.items():
        if key != "topic_info":
            logger.debug(f"  {key}: {value}")

    # Save model
    save_path = "saved_bertopic_model"
    logger.debug(f"\nSaving model to {save_path}...")

    saved_info = save_topic_model(
        model, save_path, serialization="safetensors", save_metadata=True
    )

    logger.debug("Saved model information:")
    for key, value in saved_info.items():
        logger.debug(f"  {key}: {value}")

    # Load model
    logger.debug(f"\nLoading model from {save_path}...")
    loaded_model, metadata = load_topic_model(save_path, load_metadata=True)

    logger.debug("Loaded model metadata:")
    if metadata:
        logger.debug(f"  Saved at: {metadata.get('saved_at', 'unknown')}")
        logger.debug(f"  Number of topics: {metadata['model_info']['n_topics']}")
        logger.debug(f"  Embedding model: {metadata['model_info']['embedding_model']}")

    # Compare models
    logger.debug("\nComparing original and loaded models...")
    comparison = compare_models(model, loaded_model)
    logger.debug(f"  Number of topics difference: {comparison['n_topics_diff']}")
    logger.debug(f"  Embedding models same: {comparison['embedding_models_same']}")
    logger.debug(f"  Languages same: {comparison['languages_same']}")

    # Test loaded model
    logger.debug("\nTesting loaded model...")
    logger.debug("Topic information from loaded model:")
    logger.debug(loaded_model.get_topic_info())

    # Clean up
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        logger.debug(f"\nCleaned up saved model directory: {save_path}")


def _10_demonstrate_advanced_workflows():
    """Demonstrate advanced BERTopic workflows."""
    logger = CustomLogger("10_advanced_workflows")
    output_dir = f"{OUTPUT_DIR}/10_advanced_workflows"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/main.log"
    logger.basicConfig(filename=log_file)
    logger.orange(f"Log: {log_file}")

    logger.gray("\n" + "=" * 60)
    logger.orange("10. ADVANCED WORKFLOWS")
    logger.gray("=" * 60)

    docs, timestamps = create_sample_dataset()

    # Pre-compute embeddings for efficiency
    logger.debug("Pre-computing embeddings...")
    embeddings = precompute_embeddings(docs)
    logger.debug(f"Embeddings shape: {embeddings.shape}")

    # Use pre-computed embeddings
    model, topics, probs = topic_model_fit_transform(
        docs, precomputed_embeddings=embeddings, calculate_probabilities=True
    )

    logger.debug("Model trained with pre-computed embeddings")
    logger.debug(f"Number of topics: {len(model.get_topic_info())}")

    # Demonstrate hierarchical topic modeling
    logger.debug("\nDemonstrating hierarchical topic modeling...")

    # Create a hierarchical structure by reducing topics step by step
    original_topics = len(model.get_topic_info())
    logger.debug(f"Original topics: {original_topics}")

    # Reduce to fewer topics
    model_hierarchical, topics_hierarchical, probs_hierarchical = reduce_topic_count(
        model, docs, topics, probs, nr_topics=3
    )

    logger.debug(f"Hierarchical topics: {len(model_hierarchical.get_topic_info())}")

    # Show hierarchical topic structure
    logger.debug("Hierarchical topic structure:")
    for topic_id in range(len(model_hierarchical.get_topic_info())):
        topic_words = model_hierarchical.get_topic(topic_id)
        logger.debug(f"  Level 1 - Topic {topic_id}: {topic_words[:5]}")

    # Demonstrate topic evolution analysis
    logger.debug("\nAnalyzing topic evolution...")
    topics_time, _ = topics_over_time_analysis(
        model, docs, topics, timestamps, datetime_format="%Y-%m-%d"
    )

    # Find topics with significant changes
    topic_evolutions = {}
    for topic_id in topics_time["Topic"].unique():
        evolution = get_topic_evolution(topics_time, topic_id)
        if "error" not in evolution:
            topic_evolutions[topic_id] = evolution

    logger.debug(f"Analyzed evolution for {len(topic_evolutions)} topics")

    # Show most dynamic topics
    dynamic_topics = sorted(
        topic_evolutions.items(),
        key=lambda x: x[1].get("total_periods", 0),
        reverse=True,
    )[:3]

    logger.debug("Most dynamic topics:")
    for topic_id, evolution in dynamic_topics:
        logger.debug(
            f"  Topic {topic_id}: {evolution['total_periods']} periods, trend: {evolution['trend']}"
        )


def main():
    """Run the comprehensive BERTopic demonstration."""
    logger.debug("COMPREHENSIVE BERTOPIC EXAMPLE")
    logger.debug("A Practical Guide to BERTopic Implementation")
    logger.gray("=" * 60)

    try:
        # 01. Basic workflow
        _01_demonstrate_basic_workflow()

        # 02. Hierarchical Topics
        _02_demonstrate_hierarchical_topics()

        # 03. Custom components
        _03_demonstrate_custom_components()

        # 04. Topic representation
        _04_demonstrate_topic_representation()

        # 05. Topic reduction
        _05_demonstrate_topic_reduction()

        # 06. Similar topics
        _06_demonstrate_similar_topics()

        # 07. Visualizations
        _07_demonstrate_visualizations()

        # # 08. Topics over time
        # _08_demonstrate_topics_over_time()

        # # 09. Model serialization
        # _09_demonstrate_model_serialization()

        # # 10. Advanced workflows
        # _10_demonstrate_advanced_workflows()

        logger.gray("\n" + "=" * 60)
        logger.teal("COMPREHENSIVE DEMONSTRATION COMPLETED")
        logger.gray("=" * 60)
        logger.info("All BERTopic workflows have been demonstrated successfully!")
        logger.info("Check the 'bertopic_plots' directory for saved visualizations.")

    except Exception as e:
        logger.error(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
