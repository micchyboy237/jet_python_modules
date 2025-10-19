# examples/bertopic_rag_usage_example.py
"""
Example: Using TopicRAG for topic-guided document retrieval in RAG pipelines.
"""

from typing import List
import random

from jet.libs.bertopic.rag_bertopic import TopicRAG


def build_large_corpus() -> List[str]:
    """Builds a realistic large corpus with varied topics."""
    tech_docs = [
        "AI models like GPT and BERT are revolutionizing natural language processing.",
        "Quantum computing offers exponential speed-ups for certain computations.",
        "New GPUs from NVIDIA and AMD improve deep learning performance.",
        "Edge AI is enabling real-time decision making on IoT devices.",
        "MLOps best practices include continuous integration for machine learning.",
        "Data engineers optimize ETL pipelines for scalable cloud analytics."
    ]
    sports_docs = [
        "The Lakers won the championship after an intense seven-game series.",
        "Cristiano Ronaldo scored twice leading Portugal to victory.",
        "Tennis legend Serena Williams retires after her final match.",
        "The national basketball team begins training for the upcoming Olympics.",
        "Rain delays the cricket world cup semi-final.",
        "Local marathon sees record participation this year."
    ]
    finance_docs = [
        "Federal Reserve raises interest rates to combat inflation.",
        "Stock markets rally as tech shares recover.",
        "Bitcoin surges above 60,000 USD amid investor optimism.",
        "Banks report strong quarterly profits driven by higher loan demand.",
        "Analysts expect slower GDP growth next quarter.",
        "Venture capital funding slows down in startup ecosystem."
    ]
    health_docs = [
        "New vaccine trials show promising immunity response against variant strains.",
        "Doctors recommend regular exercise and balanced diet for heart health.",
        "Mental health awareness campaigns grow across social media.",
        "Hospitals report fewer flu cases this season.",
        "Breakthrough in cancer immunotherapy offers hope to patients.",
        "Health insurance premiums expected to rise next year."
    ]
    travel_docs = [
        "Tourism rebounds as restrictions ease across Europe.",
        "Top 10 destinations for budget travelers in Southeast Asia.",
        "Airlines offer flexible booking policies post-pandemic.",
        "Luxury resorts attract digital nomads with long-term stays.",
        "Backpacking through South America remains a popular adventure.",
        "Cruise lines introduce AI-powered customer experiences."
    ]

    # Combine all categories and randomly expand
    all_docs = tech_docs + sports_docs + finance_docs + health_docs + travel_docs
    big_corpus = [random.choice(all_docs) for _ in range(200)]

    # Deduplicate
    seen = set()
    unique_docs = [d for d in big_corpus if not (d in seen or seen.add(d))]
    return unique_docs


def main() -> None:
    docs = build_large_corpus()
    print(f"Corpus size after deduplication: {len(docs)}")

    # Initialize TopicRAG with verbose output
    rag = TopicRAG(embedding_model="all-MiniLM-L6-v2", verbose=True)

    print("\n[Training topic model...]")
    rag.fit_topics(docs, nr_topics="auto", min_topic_size=5)

    print("\n[Topic Summary]")
    print(rag.model.get_topic_info().head())

    # Sample queries
    queries = [
        "Latest AI models and GPU advancements",
        "Who won the basketball finals?",
        "How to stay healthy and improve heart condition?",
        "Best travel destinations after pandemic",
        "AI in healthcare technology"
    ]

    for query in queries:
        print("\n" + "=" * 90)
        print(f"Query: {query}")
        results = rag.retrieve_for_query(query, top_topics=2, top_k=5, unique_by="text")
        print(f"Retrieved {len(results)} docs across top 2 topics:\n")
        for i, r in enumerate(results, start=1):
            preview = r["text"][:90] + ("..." if len(r["text"]) > 90 else "")
            print(f"{i:>2}. [Topic {r['topic']:>2}] (score={r['score']:.3f}) -> {preview}")


if __name__ == "__main__":
    main()
