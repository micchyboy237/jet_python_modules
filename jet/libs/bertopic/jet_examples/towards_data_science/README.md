# BERTopic Examples - A Practical Guide Implementation

This directory contains comprehensive BERTopic examples based on "A Practical Guide to BERTopic" and similar tutorials. All examples are wrapped into reusable functions with sample inputs, comments, and usage patterns.

## Files Overview

### Core Functionality

1. **`topic_model_fit_transform.py`** - Basic BERTopic model training

   - `topic_model_fit_transform()` - Fit and transform documents
   - `precompute_embeddings()` - Pre-compute embeddings for efficiency
   - `get_topic_statistics()` - Get comprehensive topic statistics

2. **`build_topic_model_with_custom_components.py`** - Custom UMAP and HDBSCAN configurations

   - `build_topic_model_with_custom_components()` - Build model with custom parameters
   - `get_umap_presets()` - Get preset UMAP configurations
   - `get_hdbscan_presets()` - Get preset HDBSCAN configurations
   - `compare_model_configurations()` - Compare different model setups

3. **`update_topic_representation.py`** - Update topic representations

   - `update_topic_representation()` - Update topic keywords with custom vectorizers
   - `create_custom_vectorizer()` - Create custom CountVectorizer or TfidfVectorizer
   - `compare_topic_representations()` - Compare before/after representations
   - `get_topic_keywords()` - Get top keywords for specific topics

4. **`reduce_topic_count.py`** - Topic reduction techniques

   - `reduce_topic_count()` - Reduce number of topics in existing model

5. **`find_similar_topics.py`** - Find semantically similar topics
   - `find_similar_topics()` - Find topics similar to a query term

### Visualization and Analysis

6. **`visualize_model.py`** - Comprehensive visualizations

   - `visualize_model()` - Create all BERTopic visualizations
   - `show_visualization()` - Display specific visualizations
   - Supports: intertopic distance, bar charts, heatmaps, document distributions, topics over time

7. **`topics_over_time_analysis.py`** - Time series analysis
   - `topics_over_time_analysis()` - Analyze topics over time
   - `analyze_topic_trends()` - Analyze trends for specific topics
   - `get_topic_evolution()` - Get detailed evolution statistics

### Model Management

8. **`save_topic_model.py`** - Model serialization
   - `save_topic_model()` - Save model with metadata
   - `load_topic_model()` - Load saved model
   - `get_model_info()` - Get comprehensive model information
   - `compare_models()` - Compare two models

### Comprehensive Example

9. **`comprehensive_bertopic_example.py`** - Complete demonstration
   - Demonstrates all BERTopic workflows
   - Includes sample dataset with timestamps
   - Shows advanced techniques and best practices

## Quick Start

### Basic Usage

```python
from topic_model_fit_transform import topic_model_fit_transform

# Sample documents
docs = [
    "Machine learning and AI are revolutionizing technology.",
    "Climate change is affecting weather patterns worldwide.",
    "Economic policies influence inflation and employment rates."
]

# Train model
model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)

# Get topic information
print(model.get_topic_info())
```

### Custom Components

```python
from build_topic_model_with_custom_components import build_topic_model_with_custom_components

# Custom UMAP and HDBSCAN parameters
umap_params = {"n_neighbors": 10, "n_components": 7, "min_dist": 0.1}
hdbscan_params = {"min_cluster_size": 8, "prediction_data": True}

model, topics, probs = build_topic_model_with_custom_components(
    docs,
    umap_params=umap_params,
    hdbscan_params=hdbscan_params,
    calculate_probabilities=True
)
```

### Topic Representation Updates

```python
from update_topic_representation import update_topic_representation

# Update with different n-gram ranges
model_updated = update_topic_representation(
    model, docs, topics,
    n_gram_range=(1, 3),  # Include trigrams
    stop_words="english"
)
```

### Visualizations

```python
from visualize_model import visualize_model

# Create all visualizations
plots = visualize_model(
    model,
    topics_over_time=topics_time,
    probs=probs,
    doc_index=0,
    save_plots=True
)

# Show specific visualization
plots["intertopic"].show()
```

### Topics Over Time

```python
from topics_over_time_analysis import topics_over_time_analysis

# Analyze topics over time
timestamps = ["2020-01-15", "2020-06-10", "2021-02-14", "2021-08-12"]
topics_time, fig = topics_over_time_analysis(
    model, docs, topics, timestamps,
    datetime_format="%Y-%m-%d"
)
```

### Model Serialization

```python
from save_topic_model import save_topic_model, load_topic_model

# Save model
save_topic_model(model, "my_model", serialization="safetensors")

# Load model
loaded_model, metadata = load_topic_model("my_model")
```

## Running Examples

### Individual Examples

Each file can be run independently:

```bash
python topic_model_fit_transform.py
python build_topic_model_with_custom_components.py
python update_topic_representation.py
# ... etc
```

### Comprehensive Example

Run the complete demonstration:

```bash
python comprehensive_bertopic_example.py
```

This will demonstrate all BERTopic workflows with a comprehensive sample dataset.

## Features Demonstrated

### Core BERTopic Workflows

- ✅ Basic fit/transform with custom parameters
- ✅ Custom UMAP and HDBSCAN configurations
- ✅ Topic representation updates with different vectorizers
- ✅ Topic reduction techniques
- ✅ Similar topic finding
- ✅ Comprehensive visualizations
- ✅ Topics over time analysis
- ✅ Model serialization and loading

### Advanced Techniques

- ✅ Pre-computed embeddings for efficiency
- ✅ Hierarchical topic modeling
- ✅ Topic evolution analysis
- ✅ Model comparison and evaluation
- ✅ Custom vectorizer configurations
- ✅ Multiple clustering algorithms (HDBSCAN, KMeans)

### Best Practices

- ✅ Reproducible results with random seeds
- ✅ Comprehensive error handling
- ✅ Detailed documentation and examples
- ✅ Modular, reusable functions
- ✅ Performance optimization techniques

## Dependencies

```bash
pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn plotly pandas
```

## Sample Dataset

The examples use a comprehensive sample dataset covering:

- **Technology & AI** (2020-2024)
- **Data Science & Analytics** (2020-2024)
- **Health & Medicine** (2020-2024)
- **Climate & Environment** (2020-2024)
- **Economics & Finance** (2020-2024)
- **Recent Topics** (2023-2024)

With timestamps spanning 2020-2024 for time series analysis.

## Output Files

When running examples, the following files may be created:

- `bertopic_plots/` - Directory containing saved visualizations
- `saved_bertopic_model/` - Directory containing saved model files
- Various HTML and PNG visualization files

## Notes

- All examples include comprehensive error handling
- Functions are designed to be reusable and modular
- Examples demonstrate both basic and advanced BERTopic workflows
- The comprehensive example shows how to combine all techniques
- All code follows BERTopic best practices for reproducibility

## References

Based on:

- "A Practical Guide to BERTopic" by Maarten Grootendorst
- BERTopic official documentation
- Various BERTopic tutorials and examples
- Best practices for topic modeling workflows
