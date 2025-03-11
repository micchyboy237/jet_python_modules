from instruction_generator.wordnet.SpellingCorrectorNorvig import SpellingCorrectorNorvig
import re


class TextComparator:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.spell_corrector = SpellingCorrectorNorvig()

    @staticmethod
    def normalize(text):
        """Normalize texts by removing non-alphanumeric characters and converting to lower case."""
        result = re.sub(r'\W+', '', text).lower()
        return result

    def contains_segments(self, text1, text2):
        long_text = text1 if len(text1) > len(text2) else text2
        short_text = text2 if len(text1) > len(text2) else text1

        # Check if any of the texts are empty
        if not long_text or not short_text:
            return False

        # Split texts into lines and normalize each line
        normalized_long_lines = [self.normalize(
            line) for line in long_text.split('\n') if line.strip()]
        normalized_short_lines = [self.normalize(
            line) for line in short_text.split('\n') if line.strip()]

        # Ensure the list with fewer lines is considered the "shorter" one for comparison
        if len(normalized_long_lines) < len(normalized_short_lines):
            normalized_long_lines, normalized_short_lines = normalized_short_lines, normalized_long_lines

        # Check each segment from the shorter text against all segments in the longer text
        for short_line in normalized_short_lines:
            if not any(self.calculate_similarity_ratio(short_line, long_line) >= self.threshold for long_line in normalized_long_lines):
                return False
        return True

    def has_improved_spelling(self, updated_text, base_text):
        base_text_misspelled_words = self.spell_corrector.unknown_words(
            base_text)
        updated_text_misspelled_words = self.spell_corrector.unknown_words(
            updated_text)

        has_improved_spelling = len(updated_text_misspelled_words) < len(
            base_text_misspelled_words)
        return has_improved_spelling

    @staticmethod
    def calculate_similarity_ratio(text1, text2):
        """Calculate the similarity ratio based on the length of the longest common substring."""
        m = [[0] * (1 + len(text2)) for i in range(1 + len(text1))]
        longest = 0
        for x in range(1, 1 + len(text1)):
            for y in range(1, 1 + len(text2)):
                if text1[x - 1] == text2[y - 1]:
                    m[x][y] = m[x - 1][y - 1] + 1
                    longest = max(longest, m[x][y])
                else:
                    m[x][y] = 0
        denominator = min(len(text1), len(text2))
        return (longest / denominator) if denominator > 0 else 0

    def __getattr__(self, name):
        """
        Delegate attribute lookup to self.model.

        This method is called if the attribute `name` isn't found in the
        SpellingCorrectorNorvig instance. If `name` is a method or attribute of self.model,
        it returns that method/attribute. Otherwise, it raises an AttributeError.
        """
        # Check if the attribute exists in self.model and return it.
        # This allows direct access to methods and properties of self.model.
        try:
            return getattr(self.spell_corrector, name)
        except AttributeError:
            # If the attribute is not found in self.model, raise an AttributeError
            # to signal that this object doesn't have the requested attribute.
            raise AttributeError(
                f"'SpellingCorrectorNorvig' object has no attribute '{name}'")
