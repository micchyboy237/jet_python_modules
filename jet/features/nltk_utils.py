import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import string

# Download required NLTK data (only needs to run once)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


def get_pos_tag(text: str) -> list[tuple[str, str]]:
    """
    Map NLTK POS tags to WordNet POS tags for lemmatization for all words in a sentence.

    Args:
        text (str): Input text (word or sentence).

    Returns:
        list[tuple[str, str]]: List of (word, WordNet POS tag) tuples.
    """
    # Tokenize the input text
    tokens = word_tokenize(text)

    # Get NLTK POS tags for all tokens
    pos_tags = nltk.pos_tag(tokens)

    # Map NLTK POS tags to WordNet POS tags
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }

    # Convert NLTK tags to WordNet tags, default to NOUN for unknown tags
    result = [(word, tag_dict.get(tag[0].upper(), wordnet.NOUN))
              for word, tag in pos_tags]

    return result


def get_word_counts_lemmatized(text: str) -> dict[str, int]:
    """
    Get word count mappings from a text string with lemmatization.

    Args:
        text (str): Input text string to analyze.

    Returns:
        dict[str, int]: Dictionary with lemmatized words as keys and their counts as values.
    """
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase and get POS tags
    pos_tags = get_pos_tag(text.lower())

    # Filter out punctuation and non-alphabetic tokens, then lemmatize
    lemmatized_words = [
        lemmatizer.lemmatize(word, pos)
        for word, pos in pos_tags
        if word not in string.punctuation
    ]

    # Use Counter for efficient counting
    return dict(Counter(lemmatized_words))
