"""
Advanced Demo: Batch Processing 1000+ Documents
Uses jet.adapters.keybert.KeyBERT for optimized batch embeddings
and jet_telemetry for end-to-end tracing.
"""

import time

from jet.adapters.keybert import KeyBERT
from jet_telemetry import chain, get_trace_url, initialize_telemetry

initialize_telemetry(service_name="keybert-batch-1k")

kw_model = KeyBERT()

# Simulate 1000+ documents
docs = [
    f"Document {i} discusses artificial intelligence and machine learning applications in sector {i % 50}."
    for i in range(1200)
]


@chain(name="batch-keyword-extraction-1k")
def process_large_batch(documents: list[str]):
    start = time.perf_counter()

    # KeyBERT adapter automatically batches embeddings via llama.cpp
    all_keywords = kw_model.extract_keywords(
        documents, keyphrase_ngram_range=(1, 2), top_n=3, stop_words="english"
    )

    elapsed = time.perf_counter() - start
    print(
        f"✅ Processed {len(documents)} docs in {elapsed:.2f}s ({len(documents) / elapsed:.1f} docs/s)"
    )
    return all_keywords


if __name__ == "__main__":
    results = process_large_batch(docs)
    print(f"Sample result (doc 0): {results[0]}")
    print(f"🔗 Trace: {get_trace_url()}")
