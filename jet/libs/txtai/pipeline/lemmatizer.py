
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


# import string
# import spacy

# nlp = spacy.load("en_core_web_sm")


# def lemmatize_text(text: str, include_puncts=[':', '.', '?', '!', '-', '/']) -> list[str]:
#     text = unidecode.unidecode(text)
#     doc = nlp(text)

#     lemmas = []
#     for i, token in enumerate(doc):
#         token_text = token.text
#         chars = list(token_text)
#         unique_chars = list(set(token_text))

#         if chars[0] in string.punctuation:
#             token = chars[0]
#             if token in include_puncts:
#                 if len(chars) > len(unique_chars):
#                     token_text = ''
#                 else:
#                     token_text = token
#             elif token not in include_puncts:
#                 token_text = ''

#             if token == ':':
#                 token_text = ': '
#             elif token == '/':
#                 token_text = ', '

#         lemmas.append(token_text)

#     return lemmas
