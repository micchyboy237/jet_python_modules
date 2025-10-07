# BERTopic Usage Examples

This directory contains comprehensive examples demonstrating various aspects of BERTopic topic modeling. Each example focuses on a specific component or workflow, making it easy to understand and implement BERTopic in your projects.

## Examples Overview

### 1. Embedding Model (`01_embedding_model.py`)

- **Purpose**: Create and configure SentenceTransformer embedding models
- **Key Features**: Model selection, embedding generation, similarity analysis
- **Use Case**: Foundation for all BERTopic workflows

### 2. Document Encoding (`02_encode_documents.py`)

- **Purpose**: Encode documents into dense vector embeddings
- **Key Features**: Batch processing, embedding analysis, similarity calculations
- **Use Case**: Pre-computing embeddings or inspecting embedding space

### 3. UMAP Dimensionality Reduction (`03_umap_model.py`)

- **Purpose**: Configure UMAP for reducing embedding dimensions
- **Key Features**: Parameter tuning, structure preservation, configuration comparison
- **Use Case**: Essential for clustering high-dimensional embeddings

### 4. HDBSCAN Clustering (`04_hdbscan_model.py`)

- **Purpose**: Configure HDBSCAN for density-based clustering
- **Key Features**: Outlier detection, cluster analysis, parameter effects
- **Use Case**: Default clustering algorithm in BERTopic

### 5. K-Means Clustering (`05_kmeans_model.py`)

- **Purpose**: Configure K-Means as alternative to HDBSCAN
- **Key Features**: Fixed number of clusters, spherical clusters, comparison with HDBSCAN
- **Use Case**: When you need a fixed number of topics

### 6. CountVectorizer (`06_vectorizer_model.py`)

- **Purpose**: Configure text vectorization for topic refinement
- **Key Features**: Stop word removal, n-gram support, document frequency filtering
- **Use Case**: Text preprocessing for topic modeling

### 7. c-TF-IDF Transformer (`07_ctfidf_model.py`)

- **Purpose**: Configure c-TF-IDF for topic-level word weighting
- **Key Features**: BM25 weighting, topic keywords, word importance scoring
- **Use Case**: Refining topic keywords and importance

### 8. KeyBERT Representation (`08_keybert_representation.py`)

- **Purpose**: Configure KeyBERT for semantic keyword extraction
- **Key Features**: Semantic similarity, keyword refinement, topic representation
- **Use Case**: Improving topic keyword quality

### 9. Basic Topic Model (`09_basic_topic_model.py`)

- **Purpose**: Create and fit basic BERTopic models
- **Key Features**: Default components, custom components, topic analysis
- **Use Case**: Foundation for topic modeling

### 10. Advanced Topic Model (`10_advanced_topic_model.py`)

- **Purpose**: Create advanced BERTopic models with all components
- **Key Features**: Full pipeline, all refinements, comprehensive modeling
- **Use Case**: Production-ready topic modeling

### 11. Topic Information Utility (`11_topic_info_utility.py`)

- **Purpose**: Extract and analyze topic information
- **Key Features**: Topic statistics, keyword analysis, document analysis
- **Use Case**: Understanding and interpreting topic modeling results

### 12. Comprehensive Example (`12_comprehensive_example.py`)

- **Purpose**: Complete BERTopic workflow demonstration
- **Key Features**: All components working together, best practices, export capabilities
- **Use Case**: End-to-end topic modeling pipeline

## Getting Started

### Prerequisites

```bash
pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn keybert
```

### Running Examples

```bash
# Run individual examples
python 01_embedding_model.py
python 02_encode_documents.py
# ... etc

# Run comprehensive example
python 12_comprehensive_example.py
```

## Key Components

### Embedding Models

- **all-MiniLM-L6-v2**: Lightweight, fast, good for most use cases
- **BAAI/bge-base-en-v1.5**: Better semantic understanding, larger model
- **Custom models**: Domain-specific or multilingual models

### Clustering Algorithms

- **HDBSCAN**: Default, density-based, can identify outliers
- **K-Means**: Fixed number of clusters, spherical clusters
- **Custom**: Any sklearn-compatible clustering algorithm

### Text Processing

- **CountVectorizer**: Basic text vectorization
- **c-TF-IDF**: Topic-level word weighting
- **KeyBERT**: Semantic keyword extraction

## Best Practices

### Model Selection

1. Start with basic pipeline for quick prototyping
2. Add custom components for better results
3. Use advanced pipeline for production systems
4. Always validate results with domain experts

### Parameter Tuning

1. **UMAP**: Lower n_neighbors for local structure, higher for global
2. **HDBSCAN**: Adjust min_cluster_size based on expected topic size
3. **Vectorizer**: Use stop_words and max_df to filter noise

### Performance Tips

1. Pre-compute embeddings for large datasets
2. Use batch processing for document encoding
3. Consider model size vs. accuracy trade-offs
4. Monitor memory usage with large document sets

## Example Workflows

### Quick Start

```python
from jet.adapters.bertopic import BERTopic

# Basic usage
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(documents)
```

### Custom Components

```python
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

# Custom components
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model = umap.UMAP(n_neighbors=5, min_dist=0.01)
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15)

# Custom BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model
)
```

### Advanced Pipeline

```python
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer

# Advanced components
vectorizer_model = CountVectorizer(max_df=0.8, stop_words="english")
ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
keybert_model = KeyBERTInspired()

# Advanced BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=keybert_model
)
```

## Output Files

The comprehensive example generates several output files:

- `comprehensive_topic_analysis.csv`: Document-topic assignments
- `topic_information.csv`: Topic statistics and information

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or use smaller models
2. **Slow processing**: Use faster embedding models or pre-compute embeddings
3. **Poor topic quality**: Adjust clustering parameters or use different models
4. **KeyBERT errors**: Ensure all dependencies are installed correctly

### Performance Optimization

1. Use GPU acceleration when available
2. Pre-compute embeddings for large datasets
3. Use appropriate model sizes for your use case
4. Monitor memory usage and adjust batch sizes

## Contributing

Feel free to contribute additional examples or improve existing ones. Each example should be:

- Self-contained and runnable
- Well-documented with clear explanations
- Focused on a specific aspect of BERTopic
- Include error handling and best practices

## License

These examples are provided as-is for educational and demonstration purposes. Please refer to the BERTopic documentation for the latest information and updates.
