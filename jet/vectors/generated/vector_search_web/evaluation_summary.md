# Vector Search Evaluation Summary

## Overview
This summary presents the evaluation results of different embedding models for semantic search.
- **Chunk Sizes Used**: 150, 250, 350
- **Top-K Results Evaluated**: 3
- **Best Model**: sentence-transformers/multi-qa-MiniLM-L6-cos-v1
  - Precision@3: 0.5000
  - Recall@3: 0.7500
  - MRR: 0.7500

## Model Performance
| Model | Precision | Recall | MRR |
|-------|-----------|--------|-----|
| sentence-transformers/all-MiniLM-L6-v2 | 0.3333 | 0.5000 | 0.7500 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | 0.5000 | 0.7500 | 0.7500 |
| Snowflake/snowflake-arctic-embed-s | 0.1667 | 0.2500 | 0.5000 |

## Example Query Results
### Query: What is the main topic of AI?
| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc3 | 0 | Overview | 1.0000 | False | Overview
Overview of AI technologies. Overview of AI technologies. Overview of AI technologies. Over... |
| doc2 | 0 | Main Topic | 0.0000 | True | Main Topic
Content about machine learning. Content about machine learning. Content about machine lea... |

### Query: Explain AI technologies in detail.
| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc3 | 1 | Conclusion | 0.0000 | True | Conclusion
AI is transformative. AI is transformative. AI is transformative. AI is transformative. A... |
| doc3 | 0 | Overview | 1.0000 | True | Overview
Overview of AI technologies. Overview of AI technologies. Overview of AI technologies. Over... |

