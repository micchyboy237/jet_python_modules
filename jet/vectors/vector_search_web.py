import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk
import re
import gc
import torch
from huggingface_hub import snapshot_download
from pathlib import Path
from typing import List, Tuple, Dict
import json

from jet.file.utils import save_file
from jet.logger import logger

nltk.download('punkt', quiet=True)


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)


class VectorSearchWeb:
    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
                 max_context_size: int = 512):
        """Initialize with embedding model, cross-encoder, and context size."""
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.embed_model = SentenceTransformer(
            embed_model_name, device=self.device)
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self.max_context_size = max_context_size
        self.index = None
        self.chunk_metadata = []
        logger.info("Initialized with model %s, context size %d",
                    embed_model_name, max_context_size)

    def preprocess_web_document(self, text: str) -> List[Tuple[str, str]]:
        """Split web-scraped text into header-content pairs, with noise filtering."""
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid document text")
            return []
        text = re.sub(r'(<script.*?</script>|<style.*?</style>|<!--.*?-->)',
                      '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        header_pattern = r'(<h[1-6]>.*?</h[1-6]>)'
        sections = []
        current_header = "No Header"
        current_content = []
        parts = re.split(header_pattern, text, flags=re.IGNORECASE)
        for i in range(0, len(parts), 2):
            content = parts[i].strip()
            if content:
                current_content.append(content)
            if i + 1 < len(parts):
                next_header = parts[i + 1].strip()
                if current_content:
                    sections.append(
                        (current_header, '\n'.join(current_content)))
                    current_content = []
                current_header = re.sub(r'<[^>]+>', '', next_header)
        if current_content:
            sections.append((current_header, '\n'.join(current_content)))
        sections = [(h, c) for h, c in sections if len(c.strip()) > 50 and not re.match(
            r'^(Navigation|Footer|Copyright|Login|Advertisement|Sidebar)', h, re.IGNORECASE)]
        return sections

    def chunk_document(self, text: str, doc_id: str, chunk_size: int, overlap: int) -> List[Tuple[str, str, int, str]]:
        """Chunk document, respecting headers and context size."""
        if chunk_size > self.max_context_size:
            logger.warning("Chunk size %d exceeds context size %d; capping at %d",
                           chunk_size, self.max_context_size, self.max_context_size)
            chunk_size = self.max_context_size
        sections = self.preprocess_web_document(text)
        chunks = []
        chunk_idx = 0
        for header, content in sections:
            section_text = f"{header}\n{content}" if header != "No Header" else content
            section_tokens = len(self.tokenizer.encode(
                section_text, add_special_tokens=False, truncation=True, max_length=self.max_context_size))
            if section_tokens <= self.max_context_size:
                chunks.append((doc_id, section_text, chunk_idx, header))
                chunk_idx += 1
            else:
                logger.info("Section '%s' in doc %s has %d tokens, exceeds context size %d; splitting",
                            header, doc_id, section_tokens, self.max_context_size)
                token_chunks = self._chunk_by_tokens(
                    section_text, doc_id, chunk_size, overlap, chunk_idx, header)
                chunks.extend(token_chunks)
                chunk_idx += len(token_chunks)
        return chunks

    def _chunk_by_tokens(self, text: str, doc_id: str, chunk_size: int, overlap: int,
                         start_idx: int, header: str) -> List[Tuple[str, str, int, str]]:
        """Chunk text by tokens with overlap."""
        tokens = self.tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=self.max_context_size)
        chunks = []
        i = 0
        chunk_idx = start_idx
        while i < len(tokens):
            end_idx = min(i + chunk_size, len(tokens))
            chunk_tokens = tokens[i:end_idx]
            chunk_text = self.tokenizer.decode(
                chunk_tokens, skip_special_tokens=True)
            chunk_token_count = len(self.tokenizer.encode(
                chunk_text, add_special_tokens=False, truncation=True, max_length=self.max_context_size))
            if chunk_token_count > self.max_context_size:
                logger.warning("Chunk %d in doc %s has %d tokens, exceeds context size %d; skipping",
                               chunk_idx, doc_id, chunk_token_count, self.max_context_size)
                i += chunk_size - overlap
                chunk_idx += 1
                continue
            if chunk_text.strip():
                chunks.append((doc_id, chunk_text, chunk_idx, header))
            chunk_idx += 1
            i += chunk_size - overlap
        return chunks

    def index_documents(self, documents: List[Tuple[str, str]], chunk_sizes: List[int], overlap_ratio: float = 0.2):
        """Index documents with multiple chunk sizes."""
        all_chunks = []
        for doc_id, text in documents:
            for chunk_size in chunk_sizes:
                overlap = int(chunk_size * overlap_ratio)
                chunks = self.chunk_document(text, doc_id, chunk_size, overlap)
                all_chunks.extend(chunks)
        if not all_chunks:
            logger.error("No chunks generated for indexing")
            return
        chunk_texts = [chunk[1] for chunk in all_chunks]
        embeddings = self.embed_model.encode(
            chunk_texts, show_progress_bar=True, batch_size=16, device=self.device)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunk_metadata = all_chunks
        logger.info(f"Indexed {len(all_chunks)} chunks with dimension %d", dim)

    def search(self, query: str, k: int = 5, use_cross_encoder: bool = True, query_type: str = "short") -> List[Tuple[str, str, int, str, float]]:
        """Search with deduplication to reduce redundant neighbors."""
        chunk_size_preference = 150 if query_type == "short" else 250
        query_embedding = self.embed_model.encode(
            [query], show_progress_bar=False)[0]
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        if not self.index or not self.chunk_metadata:
            logger.error("Index or metadata not initialized")
            return []
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), k * 2)
        candidates = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.chunk_metadata):
                candidates.append(self.chunk_metadata[idx] + (score,))
        if use_cross_encoder:
            pairs = [[query, chunk[1]] for chunk in candidates]
            scores = self.cross_encoder.predict(pairs, convert_to_numpy=True)
            scores = (scores - scores.min()) / \
                (scores.max() - scores.min() + 1e-8)
            candidates = [(c[0], c[1], c[2], c[3], s)
                          for c, s in zip(candidates, scores)]
            candidates = sorted(candidates, key=lambda x: x[4], reverse=True)
        seen_headers = {}
        deduped = []
        for candidate in candidates:
            doc_id, _, _, header, score = candidate
            key = (doc_id, header)
            if key not in seen_headers or score > seen_headers[key][4]:
                seen_headers[key] = candidate
        candidates = list(seen_headers.values())
        candidates = sorted(candidates, key=lambda x: (
            abs(len(self.tokenizer.encode(
                x[1], add_special_tokens=False, truncation=True, max_length=self.max_context_size)) - chunk_size_preference),
            -x[4]
        ))[:k]
        return candidates

    def evaluate_models(self, documents: List[Tuple[str, str]],
                        validation_set: List[Tuple[str, List[Tuple[str, int]]]],
                        model_names: List[str], chunk_sizes: List[int],
                        overlap_ratio: float = 0.2, k: int = 5) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple models and return performance metrics."""
        results = {}
        original_model = self.embed_model.model_card_data.get(
            'model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        original_tokenizer = self.tokenizer
        original_context_size = self.max_context_size
        for model_name in model_names:
            logger.info("Evaluating model: %s", model_name)
            try:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
                logger.info("Using device: %s for model %s",
                            self.device, model_name)
                self.embed_model = SentenceTransformer(
                    model_name, device=self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.max_context_size = 512
                self.index_documents(
                    documents, chunk_sizes=chunk_sizes, overlap_ratio=overlap_ratio)
                if not self.index:
                    logger.error(
                        "Failed to index documents for model %s", model_name)
                    results[model_name] = {
                        'precision': 0.0, 'recall': 0.0, 'mrr': 0.0}
                    continue
                precision_sum = 0.0
                recall_sum = 0.0
                mrr_sum = 0.0
                total_queries = len(validation_set)
                for query, relevant_chunks in validation_set:
                    query_type = "short" if len(self.tokenizer.encode(
                        query, add_special_tokens=False, truncation=True, max_length=self.max_context_size)) < 50 else "long"
                    search_results = self.search(
                        query, k=k, query_type=query_type)
                    relevant_set = set((doc_id, chunk_idx)
                                       for doc_id, chunk_idx in relevant_chunks)
                    retrieved_set = set((doc_id, chunk_idx)
                                        for doc_id, _, chunk_idx, _, _ in search_results)
                    precision = len(
                        relevant_set & retrieved_set) / k if k > 0 else 0.0
                    precision_sum += precision
                    recall = len(relevant_set & retrieved_set) / \
                        len(relevant_set) if relevant_set else 0.0
                    recall_sum += recall
                    mrr = 0.0
                    for rank, (doc_id, _, chunk_idx, _, _) in enumerate(search_results, 1):
                        if (doc_id, chunk_idx) in relevant_set:
                            mrr = 1.0 / rank
                            break
                    mrr_sum += mrr
                results[model_name] = {
                    'precision': precision_sum / total_queries if total_queries else 0.0,
                    'recall': recall_sum / total_queries if total_queries else 0.0,
                    'mrr': mrr_sum / total_queries if total_queries else 0.0
                }
                logger.info("Model %s: Precision@%d=%.4f, Recall@%d=%.4f, MRR=%.4f",
                            model_name, k, results[model_name]['precision'],
                            k, results[model_name]['recall'], results[model_name]['mrr'])
            except Exception as e:
                logger.error("Error evaluating model %s: %s",
                             model_name, str(e))
                results[model_name] = {
                    'precision': 0.0, 'recall': 0.0, 'mrr': 0.0}
            finally:
                self.embed_model = None
                self.tokenizer = None
                self.index = None
                self.chunk_metadata = []
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
        self.embed_model = SentenceTransformer(
            original_model, device=self.device)
        self.tokenizer = original_tokenizer
        self.max_context_size = original_context_size
        self.index = None
        self.chunk_metadata = []
        return results

    def evaluate_retrieval_examples(self, documents: List[Tuple[str, str]],
                                    example_queries: List[Tuple[str, List[Tuple[str, int]]]],
                                    chunk_sizes: List[int], overlap_ratio: float = 0.2, k: int = 5) -> Dict[str, List[Dict]]:
        """Evaluate example queries and return detailed results."""
        self.index_documents(
            documents, chunk_sizes=chunk_sizes, overlap_ratio=overlap_ratio)
        results = {}
        for query, relevant_chunks in example_queries:
            query_type = "short" if len(self.tokenizer.encode(
                query, add_special_tokens=False)) < 50 else "long"
            search_results = self.search(query, k=k, query_type=query_type)
            relevant_set = set((doc_id, chunk_idx)
                               for doc_id, chunk_idx in relevant_chunks)
            result_list = []
            for doc_id, chunk_text, chunk_idx, header, score in search_results:
                is_relevant = (doc_id, chunk_idx) in relevant_set
                result_list.append({
                    'doc_id': doc_id,
                    'chunk_idx': chunk_idx,
                    'header': header,
                    'score': score,
                    'text': chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                    'is_relevant': is_relevant
                })
            results[query] = result_list
        return results

    def generate_summary(self, model_scores: Dict[str, Dict[str, float]],
                         example_results: Dict[str, List[Dict]],
                         chunk_sizes: List[int], k: int,
                         validation_set: List[Tuple[str, List[Tuple[str, int]]]] = None) -> None:
        """Generate a markdown summary and HTML chart of evaluation results, including top results."""
        # Find the best model based on precision
        best_model = max(
            model_scores, key=lambda x: model_scores[x]['precision'])
        best_metrics = model_scores[best_model]

        # Generate markdown summary
        markdown_content = "# Vector Search Evaluation Summary\n\n"
        markdown_content += "## Overview\n"
        markdown_content += "This summary presents the evaluation results of different embedding models for semantic search.\n"
        markdown_content += f"- **Chunk Sizes Used**: {', '.join(map(str, chunk_sizes))}\n"
        markdown_content += f"- **Top-K Results Evaluated**: {k}\n"
        markdown_content += f"- **Best Model**: {best_model}\n"
        markdown_content += f"  - Precision@{k}: {best_metrics['precision']:.4f}\n"
        markdown_content += f"  - Recall@{k}: {best_metrics['recall']:.4f}\n"
        markdown_content += f"  - MRR: {best_metrics['mrr']:.4f}\n\n"

        # Model Performance Table
        markdown_content += "## Model Performance\n"
        markdown_content += "| Model | Precision | Recall | MRR |\n"
        markdown_content += "|-------|-----------|--------|-----|\n"
        for model, metrics in model_scores.items():
            markdown_content += f"| {model} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['mrr']:.4f} |\n"
        markdown_content += "\n"

        # Top Results (highest-scoring chunk per query)
        markdown_content += "## Top Results\n"
        markdown_content += "The highest-scoring chunk for each query.\n"
        markdown_content += "| Query | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |\n"
        markdown_content += "|-------|--------|----------|--------|-------|----------|--------------|\n"

        # Process example queries
        for query, results in example_results.items():
            if results:  # Get the highest-scoring result
                top_result = max(results, key=lambda x: x['score'])
                markdown_content += f"| {query} | {top_result['doc_id']} | {top_result['chunk_idx']} | {top_result['header']} | {top_result['score']:.4f} | {top_result['is_relevant']} | {top_result['text']} |\n"

        # Process validation set queries (if provided and not in example_queries)
        if validation_set:
            for query, _ in validation_set:
                if query not in example_results:
                    query_type = "short" if len(self.tokenizer.encode(
                        query, add_special_tokens=False)) < 50 else "long"
                    search_results = self.search(
                        query, k=k, query_type=query_type)
                    if search_results:
                        top_result = max(search_results, key=lambda x: x[4])
                        doc_id, chunk_text, chunk_idx, header, score = top_result
                        markdown_content += f"| {query} | {doc_id} | {chunk_idx} | {header} | {score:.4f} | N/A | {chunk_text[:100] + '...' if len(chunk_text) > 100 else chunk_text} |\n"
        markdown_content += "\n"

        # Example Query Results (all top-k results)
        markdown_content += "## Example Query Results\n"
        for query, results in example_results.items():
            markdown_content += f"### Query: {query}\n"
            markdown_content += "| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |\n"
            markdown_content += "|--------|----------|--------|-------|----------|--------------|\n"
            for result in results:
                markdown_content += f"| {result['doc_id']} | {result['chunk_idx']} | {result['header']} | {result['score']:.4f} | {result['is_relevant']} | {result['text']} |\n"
            markdown_content += "\n"

        # Save markdown summary
        with open(f"{OUTPUT_DIR}/evaluation_summary.md", "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.success(
            f"Generated markdown summary at {OUTPUT_DIR}/evaluation_summary.md")

        # Generate HTML chart
        chart_data = {
            "labels": list(model_scores.keys()),
            "precision": [model_scores[model]["precision"] for model in model_scores],
            "recall": [model_scores[model]["recall"] for model in model_scores],
            "mrr": [model_scores[model]["mrr"] for model in model_scores]
        }

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Performance Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Model Performance Comparison</h1>
    <canvas id="performanceChart" width="800" height="400"></canvas>
    <script>
        const ctx = document.getElementById('performanceChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_data["labels"])},
                datasets: [
                    {{
                        label: 'Precision',
                        data: {json.dumps(chart_data["precision"])},
                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Recall',
                        data: {json.dumps(chart_data["recall"])},
                        backgroundColor: 'rgba(255, 99, 132, 0.5)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'MRR',
                        data: {json.dumps(chart_data["mrr"])},
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1,
                        title: {{ display: true, text: 'Metric Value' }}
                    }},
                    x: {{
                        title: {{ display: true, text: 'Model' }}
                    }}
                }},
                plugins: {{
                    legend: {{ display: true, position: 'top' }},
                    title: {{ display: true, text: 'Model Performance Metrics' }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        with open(f"{OUTPUT_DIR}/performance_chart.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.success(
            f"Generated HTML chart at {OUTPUT_DIR}/performance_chart.html")


if __name__ == "__main__":
    import json

    data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/mocks/vector_search"

    # Load data from JSON file (or define directly)
    with open(f"{data_dir}/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        documents = [(doc["doc_id"], doc["text"]) for doc in data["documents"]]
        validation_set = [(item["query"], [(chunk["doc_id"], chunk["chunk_idx"]) for chunk in item["relevant_chunks"]])
                          for item in data["validation_set"]]
        example_queries = [(item["query"], [(chunk["doc_id"], chunk["chunk_idx"]) for chunk in item["relevant_chunks"]])
                           for item in data["example_queries"]]

    searcher = VectorSearchWeb(max_context_size=512)
    chunk_sizes = [150, 250, 350]
    model_names = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "Snowflake/snowflake-arctic-embed-s",
    ]
    model_scores = searcher.evaluate_models(
        documents, validation_set, model_names, chunk_sizes, overlap_ratio=0.2, k=3)
    best_model = max(model_scores, key=lambda x: model_scores[x]['precision'])
    logger.info("Best model: %s with Precision@3=%.4f, Recall@3=%.4f, MRR=%.4f",
                best_model, model_scores[best_model]['precision'],
                model_scores[best_model]['recall'], model_scores[best_model]['mrr'])
    searcher = VectorSearchWeb(
        embed_model_name=best_model, max_context_size=512)
    example_results = searcher.evaluate_retrieval_examples(
        documents, example_queries, chunk_sizes, overlap_ratio=0.2, k=3)
    searcher.generate_summary(
        model_scores, example_results, chunk_sizes, k=3, validation_set=validation_set)
