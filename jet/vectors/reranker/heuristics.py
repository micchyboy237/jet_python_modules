import re
from typing import List, TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Plus
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class SearchResult(TypedDict):
    doc_index: int
    score: float
    text: str


def preprocess_text(text: str) -> str:
    """
    Preprocess text by cleaning, lowercasing, removing stopwords, and lemmatizing.
    Returns preprocessed text as a string.
    """
    # Basic cleaning: remove special characters, extra spaces, and optionally numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a string
    return ' '.join(tokens)


def preprocess_corpus(corpus: List[str]) -> List[str]:
    """Preprocess a list of documents."""
    return [preprocess_text(doc) for doc in corpus]


def tfidf_search(corpus: List[str], query: str) -> List[SearchResult]:
    # Preprocess corpus and query
    preprocessed_corpus = preprocess_corpus(corpus)
    preprocessed_query = preprocess_text(query)

    # Initialize and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_corpus)
    query_vector = vectorizer.transform([preprocessed_query])

    # Compute similarities
    similarities = (tfidf_matrix * query_vector.T).toarray().flatten()
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_scores = similarities[ranked_indices]

    # Prepare results using original corpus for text field
    results: List[SearchResult] = [
        {"doc_index": int(idx), "score": float(score), "text": corpus[idx]}
        for idx, score in zip(ranked_indices, ranked_scores)
    ]
    return results


def bm25_plus_search(corpus: List[str], query: str) -> List[SearchResult]:
    # Preprocess corpus and query
    preprocessed_corpus = preprocess_corpus(corpus)
    preprocessed_query = preprocess_text(query)

    # Tokenize for BM25Plus
    tokenized_corpus = [doc.split() for doc in preprocessed_corpus]
    tokenized_query = preprocessed_query.split()

    # Initialize BM25Plus with no length normalization (b=0.0)
    bm25 = BM25Plus(tokenized_corpus, b=0.0)
    scores = bm25.get_scores(tokenized_query)

    # Rank results
    ranked_indices = np.argsort(scores)[::-1]
    ranked_scores = scores[ranked_indices]

    # Prepare results using original corpus for text field
    results: List[SearchResult] = [
        {"doc_index": int(idx), "score": float(score), "text": corpus[idx]}
        for idx, score in zip(ranked_indices, ranked_scores)
    ]
    return results
