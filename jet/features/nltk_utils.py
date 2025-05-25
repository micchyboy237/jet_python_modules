import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter

# Download required NLTK data (only needs to run once)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


def get_word_counts_lemmatized(text: str) -> dict[str, int]:
    """
    Get word count mappings from a text string with lemmatization, excluding stop words,
    sorted by count in descending order.

    Args:
        text (str): Input text string to analyze.

    Returns:
        dict[str, int]: Dictionary with lemmatized words as keys and their counts as values, sorted by count descending.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text.lower())

    words = [
        token for token in tokens
        if token.isalpha() and token not in stop_words
    ]

    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    counts = Counter(lemmatized_words)

    # Sort counts by descending frequency
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
