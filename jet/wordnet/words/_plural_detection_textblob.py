from textblob import Word
from typing import Union


def is_plural_textblob(word: str) -> bool:
    """
    Detects if a word is plural using TextBlob by comparing it to its singularized form.

    Args:
        word (str): The word to check.

    Returns:
        bool: True if the word is plural, False otherwise.
    """
    blob = Word(word)
    singular_form = blob.singularize()
    return word != singular_form and singular_form != word


# Example usage
if __name__ == "__main__":
    words = ["boxes", "box", "teeth", "water"]
    for word in words:
        print(f"{word}: {'Plural' if is_plural_textblob(word) else 'Not plural'}")
