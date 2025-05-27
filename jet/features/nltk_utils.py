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


def get_word_sentence_combination_counts(text: str, n: int = 2, min_count: int = 1, in_sequence: bool = False) -> dict[tuple[str, ...], int]:
    """
    Get counts of word combinations (n-grams) within sentences from a text string with lemmatization,
    excluding stop words, sorted by count in descending order, including only those with counts >= min_count.
    Combinations can be sequential or order-independent within each sentence.

    Args:
        text (str): Input text string to analyze.
        n (int): Size of word combinations (default is 2 for bigrams).
        min_count (int): Minimum count threshold for combinations to be included (default is 1).
        in_sequence (bool): If True, combinations are sequential n-grams (words must appear in order).
                            If False, combinations are order-independent (default is False).

    Returns:
        dict[tuple[str, ...], int]: Dictionary with tuples of lemmatized word combinations as keys
                                    and their counts as values, sorted by count descending.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Split text into sentences
    sentences = nltk.sent_tokenize(text.lower())

    # Initialize counter for word combinations
    counts = Counter()

    for sentence in sentences:
        # Tokenize sentence
        tokens = word_tokenize(sentence)

        # Filter alphabetic tokens and remove stop words
        words = [
            token for token in tokens
            if token.isalpha() and token not in stop_words
        ]

        # Lemmatize words
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

        # Generate combinations based on in_sequence
        if in_sequence:
            # Generate sequential n-grams
            combinations = [
                tuple(lemmatized_words[i:i + n])
                for i in range(len(lemmatized_words) - n + 1)
            ]
        else:
            # Generate all possible n-word combinations within the sentence
            combinations = list(itertools.combinations(lemmatized_words, n))

        # Update counter with combinations
        counts.update(combinations)

    # Filter by min_count and sort by descending frequency
    return dict(sorted(
        {ngram: count for ngram, count in counts.items() if count >=
         min_count}.items(),
        key=lambda x: x[1],
        reverse=True
    ))
