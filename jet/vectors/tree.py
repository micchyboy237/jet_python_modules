from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def tokenize_document(document):
    """Splits the document into sentences and removes extra whitespace."""
    return [sentence.strip() for sentence in sent_tokenize(document)]


def vectorize_sentences(sentences):
    """Converts sentences to TF-IDF vectors."""
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(sentences), vectorizer


def cluster_sentences(X, num_clusters=5):
    """Clusters sentences using KMeans."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)
    return kmeans


def evaluate_thoughts(sentences, labels):
    """Evaluates sentences within clusters using Tree of Thoughts BFS."""
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(sentences[i])

    thought_tree = []
    for label, cluster_sentences in clusters.items():
        thought_tree.append({
            "cluster": label,
            "sentences": cluster_sentences,
            # Summarize first 2 sentences
            "summary": " ".join(cluster_sentences[:2])
        })

    return thought_tree


def generate_hierarchical_summary(thought_tree):
    """Generates a hierarchical structure from the thought tree."""
    top_summary = " ".join([node["summary"] for node in thought_tree])
    return {"summary": top_summary, "thoughts": thought_tree}


def main():
    document = """
    Natural language processing is an exciting field. It enables computers to understand human language.
    Applications of NLP include sentiment analysis, machine translation, and more.
    Clustering is a useful technique for grouping similar sentences.
    """
    sentences = tokenize_document(document)
    X, vectorizer = vectorize_sentences(sentences)
    kmeans = cluster_sentences(X, num_clusters=2)
    thought_tree = evaluate_thoughts(sentences, kmeans.labels_)
    hierarchical_summary = generate_hierarchical_summary(thought_tree)

    import json
    from jet.transformers.object import make_serializable
    from jet.logger import logger
    hierarchical_summary_str = json.dumps(
        make_serializable(hierarchical_summary), indent=2)
    print("Hierarchical Summary:")
    logger.success(hierarchical_summary_str)


if __name__ == "__main__":
    main()
