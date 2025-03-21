import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from jet.utils.text import fix_and_unidecode


def lemmatize_text(text: str) -> str:
    """Lemmatizes text while preserving contractions, punctuation, spaces, and newlines."""

    text = fix_and_unidecode(text)
    lemmatizer = WordNetLemmatizer()

    # Match sentence splits while keeping leading/trailing spaces & newlines
    # Split but keep whitespace as separate tokens
    sentences = re.split(r'(\s+)', text)

    processed_sentences = []

    for segment in sentences:
        if segment.strip():  # If it's a sentence (not just whitespace)
            tokens = word_tokenize(segment)
            lemmatized_tokens = [
                lemmatizer.lemmatize(token) for token in tokens]
            detokenized = TreebankWordDetokenizer().detokenize(lemmatized_tokens)
            processed_sentences.append(detokenized)
        else:
            processed_sentences.append(segment)  # Preserve whitespace exactly

    # Reassemble text with spaces & newlines intact
    return "".join(processed_sentences)
