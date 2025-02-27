import spacy
import unittest
import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def lemmatize_text(text: str) -> list[str]:
    """Lemmatizes tokens in a given text, converting special characters to ASCII equivalents."""
    # Convert Unicode characters to closest ASCII equivalent
    text = unidecode.unidecode(text)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(token) for token in tokens]


nlp = spacy.load("en_core_web_sm")

text = "React.js and JavaScript are used in web development."
doc = nlp(text)

for token in doc:
    print(token.text, "->", token.lemma_)
