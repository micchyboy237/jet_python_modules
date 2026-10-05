"""
Advanced Demo: Long Document Chunking & Keyword Aggregation
Uses jet.adapters.llama_cpp.chunking_utils.chunk_texts for semantic chunking
and jet_telemetry for granular span visibility.
"""

from collections import Counter

from jet.adapters.keybert import KeyBERT
from jet.adapters.llama_cpp.chunking_utils import chunk_texts
from jet_telemetry import chain, get_trace_url, initialize_telemetry, tool

initialize_telemetry(service_name="keybert-long-doc-chunking")

kw_model = KeyBERT()

LONG_DOC = (
    """
Artificial intelligence (AI) is transforming healthcare through predictive analytics and personalized medicine. 
Machine learning algorithms analyze patient data to identify patterns that human doctors might miss. 
Natural language processing enables automated clinical note summarization and coding. 
Computer vision assists in radiology image interpretation and surgical robotics. 
Reinforcement learning optimizes treatment protocols and drug discovery pipelines. 
Ethical AI frameworks ensure fairness, transparency, and accountability in medical decisions. 
Federated learning allows model training across hospitals without sharing sensitive patient data. 
Explainable AI builds trust by making black-box models interpretable to clinicians. 
AI-powered chatbots provide 24/7 patient support and triage. 
Genomic AI accelerates precision medicine by linking genetic variants to disease outcomes.
"""
    * 50
)  # Simulate a very long document


@tool(name="chunk-long-document")
def chunk_document(text: str, chunk_size: int = 128) -> list[str]:
    return chunk_texts(
        text, chunk_size=chunk_size, chunk_overlap=16, strict_sentences=True
    )


@tool(name="aggregate-chunk-keywords")
def aggregate_keywords(chunk_keywords: list[list[tuple]]) -> list[tuple]:
    all_kws = [kw[0] for kws in chunk_keywords for kw in kws]
    return Counter(all_kws).most_common(10)


@chain(name="long-doc-keyword-pipeline")
def extract_from_long_doc(doc: str):
    chunks = chunk_document(doc, chunk_size=128)
    print(f"📄 Split into {len(chunks)} chunks")

    # Batch extract keywords from all chunks at once
    chunk_kws = kw_model.extract_keywords(
        chunks, keyphrase_ngram_range=(1, 2), top_n=3, use_mmr=True, diversity=0.5
    )

    top_keywords = aggregate_keywords(chunk_kws)
    return top_keywords


if __name__ == "__main__":
    keywords = extract_from_long_doc(LONG_DOC)
    print("🏆 Top Global Keywords:")
    for kw, count in keywords:
        print(f"   {kw}: {count}")
    print(f"🔗 Trace: {get_trace_url()}")
