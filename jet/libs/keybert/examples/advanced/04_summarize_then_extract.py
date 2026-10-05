"""
Advanced Demo: LLM Summarization → KeyBERT Extraction
For extremely long documents where chunking alone loses global context.
Uses jet's observed chat for summarization and KeyBERT for extraction.
"""

from jet.adapters.keybert import KeyBERT
from jet.adapters.llama_cpp.llm_utils_observed import chat
from jet_telemetry import chain, get_trace_url, initialize_telemetry

initialize_telemetry(service_name="keybert-summarize-then-extract")

kw_model = KeyBERT()

EXTREMELY_LONG_DOC = (
    """
The history of computing spans centuries, from the abacus to quantum processors. 
Early mechanical calculators like Pascal's calculator and Babbage's Analytical Engine 
laid the groundwork for programmable machines. The invention of the transistor in 1947 
revolutionized electronics, enabling smaller and faster computers. ENIAC, completed in 
1945, was among the first general-purpose electronic digital computers. The development 
of integrated circuits in the 1960s led to minicomputers and eventually personal computers. 
The ARPANET, created in 1969, evolved into the modern internet. Moore's Law predicted 
the doubling of transistors on chips every two years, driving exponential growth in 
computing power. The rise of graphical user interfaces in the 1980s made computers 
accessible to non-experts. Cloud computing emerged in the 2000s, democratizing access 
to scalable infrastructure. Machine learning and deep learning have transformed industries 
from healthcare to finance. Quantum computing promises to solve problems intractable for 
classical computers. Edge computing brings processing closer to data sources. The future 
of computing includes neuromorphic chips, DNA storage, and photonic processors. Ethical 
considerations around AI bias, privacy, and environmental impact are increasingly important. 
The convergence of computing with biology, physics, and cognitive science continues to 
push boundaries of what machines can achieve.
"""
    * 30
)  # ~15K tokens


@chain(name="summarize-and-extract-pipeline")
def summarize_then_extract(long_doc: str, max_summary_tokens: int = 300):
    # Step 1: Summarize with LLM (auto-traced as LLM span)
    print("📝 Summarizing long document...")
    summary_result = chat(
        f"Summarize the following text in {max_summary_tokens} words or fewer, "
        f"preserving all key topics and technical terms:\n\n{long_doc[:8000]}",
        project_name="keybert-summarize-then-extract",
    )
    summary = summary_result.content
    print(f"✅ Summary ({len(summary)} chars): {summary[:100]}...")

    # Step 2: Extract keywords from summary (auto-traced via KeyBERT adapter)
    print("\n🔑 Extracting keywords from summary...")
    keywords = kw_model.extract_keywords(
        summary,
        keyphrase_ngram_range=(1, 3),
        top_n=10,
        use_mmr=True,
        diversity=0.6,
    )
    return {"summary": summary, "keywords": keywords}


if __name__ == "__main__":
    result = summarize_then_extract(EXTREMELY_LONG_DOC)
    print("\n🏆 Keywords from Summary:")
    for kw, score in result["keywords"]:
        print(f"   {kw}: {score:.4f}")
    print(f"\n🔗 Trace: {get_trace_url()}")
