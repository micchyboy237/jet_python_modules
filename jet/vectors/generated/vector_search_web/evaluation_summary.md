# Vector Search Evaluation Summary

## Overview
This summary presents the evaluation results of different embedding models for semantic search.
- **Chunk Sizes Used**: 150, 250, 350
- **Top-K Results Evaluated**: 3
- **Best Model**: sentence-transformers/all-MiniLM-L6-v2
  - Precision@3: 0.0000
  - Recall@3: 0.0000
  - MRR: 0.0000

## Model Performance
| Model | Precision | Recall | MRR |
|-------|-----------|--------|-----|
| sentence-transformers/all-MiniLM-L6-v2 | 0.0000 | 0.0000 | 0.0000 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | 0.0000 | 0.0000 | 0.0000 |
| Snowflake/snowflake-arctic-embed-s | 0.0000 | 0.0000 | 0.0000 |

## Top Results
The highest-scoring chunk for each query.
| Query | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|

## Example Query Results
### Query: What are the applications of AI?
| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|

### Query: What is deep learning?
| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|

