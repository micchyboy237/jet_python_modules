from pydantic import BaseModel
import nltk
import re
import string
from nltk.tokenize import sent_tokenize
import json
from transformers import AutoTokenizer
import spacy
from nltk import word_tokenize, pos_tag
from typing import Callable, List, Optional, TypedDict

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')

# Load the English NLP model
nlp = None


def setup_nlp():
    global nlp

    if not nlp:
        nlp = spacy.load("en_core_web_sm")

    return nlp


def pos_tag_nltk(text):
    """Determine if the text contains non-proper English nouns."""
    tagged_words = pos_tag(word_tokenize(text))
    for word, tag in tagged_words:
        print(word, "-", tag)


def pos_tag_spacy(sentence):
    nlp = setup_nlp()

    # Process English sentence
    doc = nlp(sentence)

    # Display the tokens and their POS tags
    for token in doc:
        print(token.text, "-", token.pos_)


def split_words(text: str) -> list[str]:
    # Updated regex to handle cases like "A.F.&A.M."
    # This pattern matches words that start and end with an alphanumeric character,
    # including words with hyphens, apostrophes, periods, and ampersands in the middle
    return re.findall(r"(\b[\w'.&-]+\b)", text)


def get_words(
        text: str,
        n: int = 1,
        filter_word: Optional[Callable[[str], bool]] = None,
        ignore_punctuation: bool = False
) -> List[str]:
    # Check if the input is a string
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    if ignore_punctuation:
        # ignore all punctuatios except apostrophes
        punctuations = string.punctuation.replace("'", "")
        text = re.sub(rf"[{punctuations}]", "", text)

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Initialize an empty list to store results
    grouped_words = []

    # Process each sentence separately
    for sentence in sentences:
        # Find all words in the sentence, optionally ignoring punctuation
        words = split_words(sentence)

        # Call filter_word function if provided
        if filter_word:
            words = [word for word in words if filter_word(word)]

        # Group words in n-word sequences
        grouped_words.extend([" ".join(words[i:i+n])
                             for i in range(len(words) - n + 1)])

    return grouped_words


def get_non_words(text):
    # Check if the input is a string
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Define non-word characters
    whitespace = '\t\n\r\v\f'
    non_word_chars = whitespace + string.digits + string.punctuation

    # Find all non-word characters in the text
    non_words = [char for char in text if char in non_word_chars]

    return non_words


def count_words(text):
    return len(get_words(text))


def count_non_words(text):
    return len(get_non_words(text))


def process_dataset(task_name, max_length, tokenizer):
    with open(f"{task_name}.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    processed_data = []

    for item in data:
        prompt_tokens = tokenizer.tokenize(item["prompt"])
        response_tokens = tokenizer.tokenize(item["response"])

        # +2 for potential separator tokens
        if len(prompt_tokens) + len(response_tokens) + 2 <= max_length:
            concatenated_item = {
                "prompt": item["prompt"],
                "response": item["response"],
                "concatenated": item["prompt"] + " " + item["response"]
            }
            processed_data.append(concatenated_item)
        else:
            processed_data.append(item)

    with open(f"{task_name}.processed.json", "w", encoding="utf-8") as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=4)


def process_all_datasets(tasks, max_length, model):
    tokenizer = AutoTokenizer.from_pretrained(model)

    for task in tasks:
        process_dataset(task, max_length, tokenizer)


def list_all_spacy_pos_tags():
    nlp = setup_nlp()

    for tag in nlp.get_pipe("tagger").labels:
        print(tag)


def get_unique_sentences(data: List[str]):
    unique_words = list(set(data)) if all(isinstance(d, str)
                                          for d in data) else data

    return unique_words


def compare_words(word1: str, word2: str, case_sensitive: bool = False) -> bool:
    word1 = word1.strip().translate(str.maketrans('', '', string.punctuation))
    word2 = word2.strip().translate(str.maketrans('', '', string.punctuation))

    if case_sensitive:
        return word1 == word2
    else:
        return word1.lower() == word2.lower()


def count_syllables(word: str) -> int:
    """Counts the number of syllables in a word based on vowel groupings, including handling for consecutive vowels."""
    pattern = r'(?<!a)[aeiou]|a(?=[aeiou])|[aeiouAEIOU]+'

    # Find all non-overlapping matches of the pattern in the word.
    matches = re.findall(pattern, word)

    # Count matches as an initial approximation of syllable count.
    syllable_count = len(matches)

    # Further adjustments could be made here based on additional rules.

    return syllable_count


def split_by_syllables(word: str) -> List[str]:
    """Splits a word into syllables based on simple heuristics for English and Tagalog."""
    pass


def get_named_words(text):
    from nltk import ne_chunk, pos_tag, word_tokenize
    from nltk.tree import Tree

    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    current_chunk = []
    contiguous_chunk = []
    contiguous_chunks = []

    for i in chunked:
        print(f"{type(i)}: {i}")
        if type(i) == Tree:
            current_chunk = ' '.join([token for token, pos in i.leaves()])
            # Apparently, Tony and Morrison are two separate items,
            # but "Random House" and "New York City" are single items.
            contiguous_chunk.append(current_chunk)
        else:
            # discontiguous, append to known contiguous chunks.
            if len(contiguous_chunk) > 0:
                contiguous_chunks.append(' '.join(contiguous_chunk))
                contiguous_chunk = []
                current_chunk = []

    return contiguous_chunks


class SpacyWord(BaseModel):
    text: str
    lemma: str
    start_idx: int
    end_idx: int
    score: float  # Normalized score

    def __str__(self) -> str:
        """Return a readable string representation of the word."""
        return self.text


def get_spacy_words(text: str) -> List[SpacyWord]:
    nlp = setup_nlp()

    # Process the input text
    doc = nlp(text)

    # Extract vector norms
    vector_norms = [token.vector_norm for token in doc]

    if not vector_norms:
        return []

    max_norm = max(vector_norms)  # Find the max vector norm

    # Extract words with their normalized scores
    words = [
        SpacyWord(
            text=token.text,
            lemma=token.lemma_,
            start_idx=token.idx,
            end_idx=token.idx + len(token.text),
            score=(token.vector_norm / max_norm if max_norm > 0 else 0),
        )
        for token in doc
    ]

    return words


if __name__ == "__main__":
    sentence = 'Ang mga pang-uri o adjectives sa Ingles ay salitang nagbibigay turing o naglalarawan sa isang pangngalan o panghalip. Ito ay nagsasaad ng uri o katangian ng tao, bagay, hayop, pook, o pangyayari.'

    word_length = count_words(sentence)
    print(f"Word length: {word_length}")

    words = [*get_words(sentence)]
    print(f"Words: {words}")

    pos_tag_nltk(sentence)
    pos_tag_spacy(sentence)

    list_all_spacy_pos_tags()
