import re
import typing


def capitalize_first_letter(text: str) -> str:
    # Find first letter index, keep in mind text can have space at the beginning
    first_letter_index = len(text) - len(text.lstrip())
    # Capitalize first letter
    return text[:first_letter_index] + text[first_letter_index].upper() + text[first_letter_index + 1:]


def lower_first_letter(text: str) -> str:
    first_letter_index = len(text) - len(text.lstrip())
    return text[:first_letter_index] + text[first_letter_index].lower() + text[first_letter_index + 1:]


def has_non_ascii(text: str) -> bool:
    return any(ord(char) >= 128 for char in text)


def split_words(text: str) -> list[str]:
    # Updated regex to handle cases like "A.F.&A.M."
    # This pattern matches words that start and end with an alphanumeric character,
    # including words with hyphens, apostrophes, periods, and ampersands in the middle
    return re.findall(r"(\b[\w'.&-]+\b)", text)


def split_tokenize_text(text: str) -> list[str]:
    # Split on whitespace and punctuation, except for apostrophes within words
    tokens = re.findall(r"\b\w+(?:'\w+)?\b|\b\d+\b|[^\w\s]", text)
    # Find contraction tokens and split them into separate tokens
    contractions = re.compile(r"(?<=\w)'(?=\w)")
    tokens = [token for token in tokens if token]
    for token_idx, token in enumerate(tokens):
        if contractions.search(token):
            # Split the token into separate tokens but the right token has the apostrophe prefix
            splitted_tokens = contractions.split(token)
            for i, token in enumerate(splitted_tokens):
                if i > 0:
                    splitted_tokens[i] = "'" + token

            # Remove the original token and insert the splitted tokens
            tokens.pop(token_idx)
            tokens[token_idx:token_idx] = splitted_tokens

    return tokens


def remove_non_alpha_numeric(text: str) -> str:
    splitted_words = split_words(text)
    return " ".join(splitted_words)


def has_special_characters(text: str) -> bool:
    # Regular expression to match any character that is not alphanumeric,
    # not a space, and not one of the specified allowed punctuation marks.
    pattern = r"[^a-zA-Z0-9 (/!?'\")-]"
    return bool(re.search(pattern, text))


def is_numeric(s):
    """Check if the string s represents a number, including scientific notation."""
    try:
        float(s)  # for int, float, and scientific notation
        return True
    except ValueError:
        return False


def snake_case(input_string: str) -> str:
    # First, insert underscores between words and uppercase letters
    intermediate = re.sub('(?<=[a-z0-9])([A-Z])', r'_\1', input_string)
    # Replace any spaces with underscores and convert to lowercase
    return re.sub(' ', '_', intermediate).lower()


def starts_with_whitespace(text: str) -> bool:
    return re.match(r"^\s", text) is not None
