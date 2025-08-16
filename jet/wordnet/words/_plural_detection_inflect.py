import inflect
from typing import Union
from jet.logger import logger


def is_plural_inflect(word: str) -> bool:
    """
    Detects if a word is plural using the inflect library.

    Args:
        word (str): The word to check.

    Returns:
        bool: True if the word is plural, False otherwise.
    """
    p = inflect.engine()
    singular_form = p.singular_noun(word)
    if singular_form is False:
        return False
    return word != singular_form


# Example usage
if __name__ == "__main__":
    words = ["cats", "dog", "children", "furniture"]
    for word in words:
        print(f"{word}: {'Plural' if is_plural_inflect(word) else 'Not plural'}")
