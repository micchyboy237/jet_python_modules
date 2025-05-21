import nltk
import unittest
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Define allowed POS tags for filtering
ALLOWED_POS = {
    'NN', 'NNS', 'NNP', 'NNPS',  # Nouns and proper nouns
    'PRP', 'PRP$',               # Pronouns
    'JJ', 'JJR', 'JJS',          # Adjectives
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
    'RB', 'RBR', 'RBS'          # Adverbs
}

# Map NLTK POS tags to WordNet POS tags for lemmatization


def get_wordnet_pos(nltk_pos):
    """
    Map NLTK POS tags to WordNet POS tags.

    Args:
        nltk_pos (str): NLTK POS tag

    Returns:
        str: Corresponding WordNet POS tag
    """
    if nltk_pos.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos.startswith('V'):
        return wordnet.VERB
    elif nltk_pos.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if unknown


def get_pos_tag(sentence):
    """
    Perform POS tagging and lemmatization on a sentence, filtering by allowed POS tags.

    Args:
        sentence (str): Input sentence to tag and lemmatize

    Returns:
        list: List of tuples (word, pos_tag, lemma) for allowed POS tags
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence.lower())
    pos_tags = nltk.pos_tag(tokens)
    # Filter by allowed POS tags and lemmatize
    return [(word, pos, lemmatizer.lemmatize(word, get_wordnet_pos(pos)))
            for word, pos in pos_tags if pos in ALLOWED_POS]


def search_by_pos(query, documents):
    """
    Search documents for query words using lemmatization, filter by POS, include POS tags,
    and sort by number of matches (with tie-breaker on document index).

    Args:
        query (str): The search query string
        documents (list): List of document strings to search through

    Returns:
        list: Sorted list of tuples (document_index, matching_words_count, matching_words_with_pos_and_lemma)
              sorted by matching_words_count (descending) and document_index (descending for ties)
    """
    # Get POS tags and lemmas for query, filtered by ALLOWED_POS
    query_pos = get_pos_tag(query)
    query_lemmas = set(lemma for _, _, lemma in query_pos)

    # Store results: (doc_index, match_count, matching_words_with_pos_and_lemma)
    results = []

    # Process each document
    for idx, doc in enumerate(documents):
        # Get POS tags and lemmas for document, filtered by ALLOWED_POS
        doc_pos = get_pos_tag(doc)
        doc_lemmas = set(lemma for _, _, lemma in doc_pos)

        # Find matching lemmas
        matches = query_lemmas.intersection(doc_lemmas)
        match_count = len(matches)

        # Get POS tags and original words for matching lemmas
        matching_words_with_pos = [(word, pos, lemma)
                                   for word, pos, lemma in doc_pos if lemma in matches]

        # Store result
        results.append((idx, match_count, matching_words_with_pos))

    # Sort results by match count (descending) and document index (descending for ties)
    results.sort(key=lambda x: (-x[1], -x[0]))

    return results


# Example usage
if __name__ == "__main__":
    # Sample documents
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "A fox fled from danger",
        "The dog sleeps peacefully",
        "Quick foxes climb steep hills"
    ]

    # Sample query (valid sentence)
    query = "The quick foxes run dangerously"

    # Get results
    results = search_by_pos(query, docs)

    # Print results
    for doc_idx, match_count, matches_with_pos in results:
        print(f"Document {doc_idx}:")
        print(f"  Text: {docs[doc_idx]}")
        print(f"  Matching words (word, POS, lemma): {matches_with_pos}")
        print(f"  Match count: {match_count}\n")
