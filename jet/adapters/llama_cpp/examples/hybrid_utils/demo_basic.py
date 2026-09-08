from jet.adapters.llama_cpp.hybrid_utils import hybrid_search, hybrid_search_pdr


def demo_hybrid_search_model():
    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
    ]

    print("\n" + "=" * 60)
    print("HYBRID SEARCH — RERANK METHOD: MODEL")
    print("=" * 60)

    results = hybrid_search(query, docs, normalize_scores=True, rerank_method="model")

    print(f"\nQuery: {query}\n")
    print("Final ranked results (after reranking):")
    print("Format: #rank  idx  score(0-1)  vector  raw  text")
    print("-" * 60)
    for r in results:
        print(
            f"  #{r['rank']}  idx={r['index']}  "
            f"score={r['score']:.4f}  vector={r['vector_score']:.4f}  "
            f"raw={r['rerank_score_raw']:.4f}  {r['text']}"
        )


def demo_hybrid_search_bm25():
    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
    ]

    print("\n" + "=" * 60)
    print("HYBRID SEARCH — RERANK METHOD: BM25")
    print("=" * 60)

    results = hybrid_search(query, docs, normalize_scores=True, rerank_method="bm25")

    print(f"\nQuery: {query}\n")
    print("Final ranked results (after reranking):")
    print("Format: #rank  idx  score(0-1)  vector  raw  text")
    print("-" * 60)
    for r in results:
        print(
            f"  #{r['rank']}  idx={r['index']}  "
            f"score={r['score']:.4f}  vector={r['vector_score']:.4f}  "
            f"raw={r['rerank_score_raw']:.4f}  {r['text']}"
        )


def demo_hybrid_search_auto():
    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
    ]

    print("\n" + "=" * 60)
    print("HYBRID SEARCH — RERANK METHOD: AUTO")
    print("=" * 60)

    results = hybrid_search(query, docs, normalize_scores=True, rerank_method="auto")

    print(f"\nQuery: {query}\n")
    print("Final ranked results (after reranking):")
    print("Format: #rank  idx  score(0-1)  vector  raw  text")
    print("-" * 60)
    for r in results:
        print(
            f"  #{r['rank']}  idx={r['index']}  "
            f"score={r['score']:.4f}  vector={r['vector_score']:.4f}  "
            f"raw={r['rerank_score_raw']:.4f}  {r['text']}"
        )


def demo_hybrid_search_pdr():
    print("\n" + "=" * 60)
    print("HYBRID SEARCH PDR RESULTS (default method)")
    print("=" * 60)

    pdr_result = {
        "parents": [
            {
                "id": "p1",
                "content": (
                    "The giant panda (Ailuropoda melanoleuca) is a bear species "
                    "endemic to China. It is characterised by its bold black-and-white "
                    "coat and rotund body. Pandas primarily eat bamboo and can consume "
                    "up to 38 kg of it per day. They live in mountainous regions of "
                    "central China."
                ),
                "num_tokens": 64,
            },
            {
                "id": "p2",
                "content": (
                    "Python is a high-level, general-purpose programming language. "
                    "Its design philosophy emphasises code readability. Python is "
                    "dynamically typed and garbage-collected. It supports multiple "
                    "programming paradigms including structured, object-oriented, "
                    "and functional programming."
                ),
                "num_tokens": 52,
            },
            {
                "id": "p3",
                "content": (
                    "Machine learning is a subset of artificial intelligence. "
                    "It gives systems the ability to learn from data without being "
                    "explicitly programmed. Common techniques include supervised "
                    "learning, unsupervised learning, and reinforcement learning."
                ),
                "num_tokens": 45,
            },
        ],
        "children": [
            {
                "id": "c1",
                "parent_id": "p1",
                "content": "The giant panda is a bear species endemic to China.",
            },
            {
                "id": "c2",
                "parent_id": "p1",
                "content": "Pandas eat bamboo and can consume up to 38 kg per day.",
            },
            {
                "id": "c3",
                "parent_id": "p1",
                "content": "Giant pandas live in mountainous regions of central China.",
            },
            {
                "id": "c4",
                "parent_id": "p2",
                "content": "Python is a high-level programming language.",
            },
            {
                "id": "c5",
                "parent_id": "p2",
                "content": "Python supports object-oriented and functional programming.",
            },
            {
                "id": "c6",
                "parent_id": "p3",
                "content": "Machine learning is a subset of artificial intelligence.",
            },
            {
                "id": "c7",
                "parent_id": "p3",
                "content": "ML systems learn from data without explicit programming.",
            },
        ],
    }

    pdr_query = "What do giant pandas eat and where do they live?"
    pdr_results = hybrid_search_pdr(pdr_query, pdr_result, top_n=2)

    print(f"\nQuery: {pdr_query}\n")
    print(
        "Format: #rank  score  parent_id  tokens  child_text → parent_text (truncated)"
    )
    print("-" * 60)
    for r in pdr_results:
        child_preview = r["child_text"][:60].rstrip()
        parent_preview = r["text"][:80].rstrip()
        print(
            f"  #{r['rank']}  score={r['score']:.4f}  "
            f"parent={r['parent_id']}  tokens={r['num_tokens']}\n"
            f"       child  : {child_preview!r}\n"
            f"       parent : {parent_preview!r}...\n"
        )


if __name__ == "__main__":
    demo_hybrid_search_model()
    demo_hybrid_search_bm25()
    demo_hybrid_search_auto()
    demo_hybrid_search_pdr()
