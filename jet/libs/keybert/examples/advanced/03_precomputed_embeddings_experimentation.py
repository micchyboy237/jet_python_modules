"""
Advanced Demo: Pre-computed Embeddings for Rapid Experimentation
Compute embeddings once via jet's llama.cpp backend, then test
multiple KeyBERT parameter configurations without re-embedding.
"""

from jet.adapters.keybert import KeyBERT
from jet_telemetry import chain, embedding, get_trace_url, initialize_telemetry

initialize_telemetry(service_name="keybert-precomputed-embeddings")

kw_model = KeyBERT()

docs = [
    "Deep learning is a subset of machine learning using neural networks.",
    "Natural language processing enables computers to understand human text.",
    "Reinforcement learning trains agents through reward-based feedback.",
    "Computer vision allows machines to interpret visual information.",
    "Transfer learning reuses pretrained models for new tasks efficiently.",
]


@embedding(model_name="nomic-embed")
def compute_embeddings(documents: list[str]):
    """Wrap the embed call so it appears as an EMBEDDING span."""
    doc_emb, word_emb = kw_model.extract_embeddings(documents)
    return doc_emb, word_emb


@chain(name="precomputed-experimentation")
def run_experiments(documents: list[str]):
    # 1. Compute embeddings ONCE (traced as EMBEDDING span)
    doc_embeddings, word_embeddings = compute_embeddings(documents)
    print(f"✅ Embeddings computed for {len(documents)} docs")

    # 2. Experiment A: Unigrams only
    kws_a = kw_model.extract_keywords(
        documents,
        doc_embeddings=doc_embeddings,
        word_embeddings=word_embeddings,
        keyphrase_ngram_range=(1, 1),
        top_n=3,
    )
    print(f"\n🅰️  Unigrams: {[k[0] for k in kws_a[0]]}")

    # 3. Experiment B: Bigrams with MMR
    kws_b = kw_model.extract_keywords(
        documents,
        doc_embeddings=doc_embeddings,
        word_embeddings=word_embeddings,
        keyphrase_ngram_range=(1, 2),
        top_n=3,
        use_mmr=True,
        diversity=0.7,
    )
    print(f"🅱️  Bigrams+MMR: {[k[0] for k in kws_b[0]]}")

    # 4. Experiment C: Guided extraction
    kws_c = kw_model.extract_keywords(
        documents,
        doc_embeddings=doc_embeddings,
        word_embeddings=word_embeddings,
        seed_keywords=["neural", "learning"],
        top_n=3,
    )
    print(f"🅲️  Guided: {[k[0] for k in kws_c[0]]}")

    return {"unigrams": kws_a, "bigrams_mmr": kws_b, "guided": kws_c}


if __name__ == "__main__":
    results = run_experiments(docs)
    print(f"\n🔗 Trace: {get_trace_url()}")
