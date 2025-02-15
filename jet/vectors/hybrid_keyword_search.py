import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm1 * norm2) if norm1 and norm2 else 0.0


def search_tfidf(texts: list[str], query: str, top_n: int = 5) -> list[tuple[str, float]]:
    """TF-IDF search using cosine similarity."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts + [query])
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(
        query_vector, tfidf_matrix[:-1]).flatten()
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    return [(texts[i], similarity_scores[i]) for i in top_indices]


def search_word2vec(texts: list[str], query: str, model: Word2Vec, top_n: int = 5) -> list[tuple[str, float]]:
    """Word2Vec search using cosine similarity."""
    def get_embedding(text):
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
    text_vectors = np.array([get_embedding(text) for text in texts])
    query_vector = get_embedding(query)
    similarity_scores = np.array(
        [compute_cosine_similarity(query_vector, vec) for vec in text_vectors])
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    return [(texts[i], similarity_scores[i]) for i in top_indices]


def search_sbert(texts: list[str], query: str, model_name: str = 'all-MiniLM-L6-v2', top_n: int = 5) -> list[tuple[str, float]]:
    """SBERT search using cosine similarity."""
    model = SentenceTransformer(model_name)
    text_vectors = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    query_vector = model.encode(
        [query], convert_to_tensor=True).cpu().numpy()[0]
    similarity_scores = np.array(
        [compute_cosine_similarity(query_vector, vec) for vec in text_vectors])
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    return [(texts[i], similarity_scores[i]) for i in top_indices]


def main():
    texts = ["Machine learning is great", "Deep learning advances AI",
             "Natural language processing is cool"]
    query = "AI and machine learning"

    print("TF-IDF Search:")
    print(search_tfidf(texts, query))

    # Train a simple Word2Vec model
    sentences = [text.split() for text in texts]
    w2v_model = Word2Vec(sentences, vector_size=100, min_count=1, workers=4)
    print("\nWord2Vec Search:")
    print(search_word2vec(texts, query, w2v_model))

    print("\nSBERT Search:")
    print(search_sbert(texts, query))


if __name__ == "__main__":
    main()
