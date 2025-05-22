import unittest
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import string

# Download required NLTK data (only needs to run once)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


def get_word_counts_lemmatized(text: str) -> dict[str, int]:
    """
    Get word count mappings from a text string with lemmatization, excluding stop words.

    Args:
        text (str): Input text string to analyze.

    Returns:
        dict[str, int]: Dictionary with lemmatized words as keys and their counts as values.
    """
    # Initialize lemmatizer and stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Convert to lowercase and tokenize
    tokens = word_tokenize(text.lower())

    # Filter out punctuation, non-alphabetic tokens, and stop words
    words = [
        token for token in tokens if token not in string.punctuation and token not in stop_words]

    # Lemmatize words (default to noun lemmatization)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Use Counter for efficient counting
    return dict(Counter(lemmatized_words))
