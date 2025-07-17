from typing import List
from jet.wordnet.keywords.keyword_extraction import rerank_by_keywords, SimilarityResult


def example_with_list_of_strings() -> List[SimilarityResult]:
    """
    Demonstrate rerank_by_keywords with a flat list of strings.
    """
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast fox leaps over obstacles in the forest.",
        "The dog sleeps peacefully by the fire."
    ]
    seed_keywords = ["fox", "dog"]

    results = rerank_by_keywords(
        texts=texts,
        seed_keywords=seed_keywords,
        top_n=3,
        show_progress=True
    )

    print("Example with List of Strings:")
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")
        print(f"Keywords: {[kw['text'] for kw in result['keywords']]}")
        print("-" * 50)

    return results


def example_with_matrix() -> List[SimilarityResult]:
    """
    Demonstrate rerank_by_keywords with a matrix of texts.
    """
    texts = [
        [
            "The quick brown fox jumps over the lazy dog.",
            "Foxes are clever animals that adapt well."
        ],
        [
            "A fast fox leaps over obstacles in the forest.",
            "The forest is home to many swift creatures."
        ]
    ]
    seed_keywords = ["fox", "dog"]

    results = rerank_by_keywords(
        texts=texts,
        seed_keywords=seed_keywords,
        top_n=3,
        show_progress=True
    )

    print("Example with Matrix of Texts:")
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")
        print(f"Keywords: {[kw['text'] for kw in result['keywords']]}")
        print("-" * 50)

    return results


if __name__ == "__main__":
    print("Running List of Strings Example:")
    example_with_list_of_strings()
    print("\nRunning Matrix Example:")
    example_with_matrix()
