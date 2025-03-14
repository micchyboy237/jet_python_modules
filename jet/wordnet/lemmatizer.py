
import unidecode

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from jet.utils.text import fix_and_unidecode


def lemmatize_text(text: str) -> list[str]:
    """Lemmatizes tokens in a given text, converting special characters to ASCII equivalents."""
    # Convert Unicode characters to closest ASCII equivalent
    text = fix_and_unidecode(text)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(token) for token in tokens]


# import spacy

# nlp = spacy.load("en_core_web_sm")


# def lemmatize_text(text: str) -> list[str]:
#     text = fix_and_unidecode(text)
#     doc = nlp(text)
#     return [token.text for token in doc]
