import nltk
from typing import List, Tuple
from nltk import pos_tag, word_tokenize
from jet.logger import logger


class SentenceValidator:
    """A class to validate sentences using POS tagging."""

    def __init__(self):
        """Initialize the validator with NLTK resources."""
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            nltk.download('punkt')

    def is_valid_sentence(self, sentence: str) -> bool:
        """
        Validate if the input string is a grammatically valid sentence.

        Args:
            sentence: The input sentence to validate.

        Returns:
            bool: True if the sentence is valid, False otherwise.
        """
        if not sentence or not isinstance(sentence, str):
            return False

        # Check if sentence ends with valid punctuation
        if not sentence.strip()[-1] in '.!?':
            return False

        # Tokenize and get POS tags
        tokens: List[str] = word_tokenize(sentence)
        # Count non-punctuation tokens for length check
        non_punct_tokens = [token for token in tokens if token not in '.!?,;:']
        if len(non_punct_tokens) < 3:  # Minimum length check for non-punctuation tokens
            return False

        pos_tags: List[Tuple[str, str]] = pos_tag(tokens)

        # Check for at least one noun and one verb
        has_noun = False
        has_verb = False

        noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
        verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

        for _, tag in pos_tags:
            if tag in noun_tags:
                has_noun = True
            if tag in verb_tags:
                has_verb = True
            if has_noun and has_verb:
                return True

        return False
