from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def lemmatize_text(text: str) -> list[str]:
    """Lemmatizes tokens in a given text."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(token) for token in tokens]


# Example usage
if __name__ == "__main__":
    text = "React.js and JavaScript are used in web development."
    lemmatized_tokens = lemmatize_text(text)
    print(lemmatized_tokens)
