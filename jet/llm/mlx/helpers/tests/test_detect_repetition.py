from typing import Dict, List
from jet.models.tokenizer.helpers.char_tokenizer import CharTokenizer
import pytest
from jet.llm.mlx.helpers.detect_repetition import clean_repeated_ngrams, find_repeated_consecutive_ngrams, NgramRepeat
from transformers import AutoTokenizer, PreTrainedTokenizer


class TestFindRepeatedConsecutiveNgrams:
    @pytest.mark.parametrize(
        "sample,min_words,max_words,min_repeat,expected",
        [
            (
                "this is this is a repeated phrase.",
                1,
                None,
                2,
                [
                    NgramRepeat(
                        ngram="this is",
                        start_index=0,
                        end_index=7,
                        full_end_index=15,
                        num_of_repeats=2,
                    )
                ],
            ),
            (
                "The cat the cat sat on the mat.",
                1,
                None,
                2,
                [
                    NgramRepeat(
                        ngram="The cat",
                        start_index=0,
                        end_index=7,
                        full_end_index=15,
                        num_of_repeats=2,
                    )
                ],
            ),
            (
                "We need to to go now.",
                1,
                None,
                2,
                [
                    NgramRepeat(
                        ngram="to",
                        start_index=8,
                        end_index=10,
                        full_end_index=13,
                        num_of_repeats=2,
                    )
                ],
            ),
            (
                "Nothing here is repeated.",
                1,
                None,
                2,
                [],
            ),
            (
                "again again again again again.",
                1,
                None,
                2,
                [
                    NgramRepeat(
                        ngram="again",
                        start_index=0,
                        end_index=5,
                        full_end_index=29,
                        num_of_repeats=5,
                    ),
                ],
            ),
            (
                "it's it's a test.",
                1,
                None,
                2,
                [
                    NgramRepeat(
                        ngram="it's",
                        start_index=0,
                        end_index=4,
                        full_end_index=9,
                        num_of_repeats=2,
                    )
                ],
            ),
            (
                "word, word, another test.",
                1,
                None,
                2,
                [
                    NgramRepeat(
                        ngram="word",
                        start_index=0,
                        end_index=4,
                        full_end_index=10,
                        num_of_repeats=2,
                    )
                ],
            ),
        ],
    )
    def test_find_repeated_consecutive_ngrams(self, sample, min_words, max_words, min_repeat, expected):
        result = find_repeated_consecutive_ngrams(
            sample, min_words=min_words, max_words=max_words, min_repeat=min_repeat
        )
        assert result == expected


class TestFindRepeatedCaseSensitive:
    @pytest.mark.parametrize(
        "sample,case_sensitive,expected",
        [
            (
                "This is THIS is a repeated phrase.",
                False,
                [
                    NgramRepeat(
                        ngram="This is",
                        start_index=0,
                        end_index=7,
                        full_end_index=15,
                        num_of_repeats=2,
                    )
                ],
            ),
            (
                "This is THIS is a repeated phrase.",
                True,
                [],
            ),
            (
                "It's IT'S a test.",
                False,
                [
                    NgramRepeat(
                        ngram="It's",
                        start_index=0,
                        end_index=4,
                        full_end_index=9,
                        num_of_repeats=2,
                    )
                ],
            ),
            (
                "It's IT'S a test.",
                True,
                [],
            ),
        ],
    )
    def test_find_repeated_case_sensitive(self, sample, case_sensitive, expected):
        result = find_repeated_consecutive_ngrams(
            sample, case_sensitive=case_sensitive)
        assert result == expected


class TestFindRepeatedWithTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    @pytest.mark.parametrize(
        "sample,min_words,min_repeat,expected",
        [
            (
                "this is this is a repeated phrase.",
                1,
                2,
                [
                    NgramRepeat(
                        ngram="this is",
                        start_index=0,
                        end_index=7,
                        full_end_index=15,
                        num_of_repeats=2,
                    )
                ],
            ),
            (
                "word, word, another test.",
                1,
                2,
                [
                    NgramRepeat(
                        ngram="word,",
                        start_index=0,
                        end_index=5,
                        full_end_index=11,
                        num_of_repeats=2,
                    )
                ],
            ),
        ],
    )
    def test_find_repeated_with_tokenizer(self, sample, min_words, min_repeat, expected, tokenizer):
        result = find_repeated_consecutive_ngrams(
            sample, min_words=min_words, min_repeat=min_repeat, tokenizer=tokenizer
        )
        assert result == expected


class TestFindRepeatedConcatenatedTokens:
    @pytest.fixture
    def tokenizer(self):
        return CharTokenizer()

    @pytest.mark.parametrize(
        "sample,min_words,min_repeat,expected",
        [
            (
                "1000000",
                1,
                2,
                [
                    NgramRepeat(
                        ngram="0",
                        start_index=1,
                        end_index=2,
                        full_end_index=7,
                        num_of_repeats=6
                    )
                ],
            ),
        ],
    )
    def test_find_repeated_concatenated_tokens(self, sample, min_words, min_repeat, expected, tokenizer):
        result = find_repeated_consecutive_ngrams(
            sample, min_words=min_words, min_repeat=min_repeat, tokenizer=tokenizer
        )
        assert result == expected


class TestCleanRepeatedNgrams:
    @pytest.mark.parametrize(
        "sample,min_words,min_repeat,expected",
        [
            (
                "this is this is a repeated phrase.",
                1,
                2,
                "this is a repeated phrase.",
            ),
            (
                "The cat the cat sat on the mat.",
                1,
                2,
                "The cat sat on the mat.",
            ),
            (
                "We need to to go now.",
                1,
                2,
                "We need to go now.",
            ),
            (
                "Nothing here is repeated.",
                1,
                2,
                "Nothing here is repeated.",
            ),
            (
                "again again again again again.",
                1,
                2,
                "again.",
            ),
            (
                "it's it's a test.",
                1,
                2,
                "it's a test.",
            ),
            (
                "word, word, another test.",
                1,
                2,
                "word, another test.",
            ),
        ],
    )
    def test_clean_repeated_ngrams(self, sample: str, min_words: int, min_repeat: int, expected: str) -> None:
        # Given: A text sample with potential repeated n-grams
        # When: Cleaning repeated n-grams using the function
        result: str = clean_repeated_ngrams(
            sample, min_words=min_words, min_repeat=min_repeat
        )
        # Then: The result should match the expected cleaned text
        assert result == expected

    @pytest.mark.parametrize(
        "sample,case_sensitive,expected",
        [
            (
                "This is THIS is a repeated phrase.",
                False,
                "This is a repeated phrase.",
            ),
            (
                "This is THIS is a repeated phrase.",
                True,
                "This is THIS is a repeated phrase.",
            ),
            (
                "It's IT'S a test.",
                False,
                "It's a test.",
            ),
            (
                "It's IT'S a test.",
                True,
                "It's IT'S a test.",
            ),
        ],
    )
    def test_clean_repeated_case_sensitive(self, sample: str, case_sensitive: bool, expected: str) -> None:
        # Given: A text sample with case-sensitive considerations
        # When: Cleaning repeated n-grams with case sensitivity
        result: str = clean_repeated_ngrams(
            sample, case_sensitive=case_sensitive
        )
        # Then: The result should match the expected cleaned text
        assert result == expected

    @pytest.fixture
    def tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    @pytest.mark.parametrize(
        "sample,min_words,min_repeat,expected",
        [
            (
                "this is this is a repeated phrase.",
                1,
                2,
                "this is a repeated phrase.",
            ),
            (
                "word, word, another test.",
                1,
                2,
                "word, another test.",
            ),
        ],
    )
    def test_clean_repeated_with_tokenizer(
        self, sample: str, min_words: int, min_repeat: int, expected: str, tokenizer: PreTrainedTokenizer
    ) -> None:
        # Given: A text sample and a tokenizer
        # When: Cleaning repeated n-grams using the tokenizer
        result: str = clean_repeated_ngrams(
            sample, min_words=min_words, min_repeat=min_repeat, tokenizer=tokenizer
        )
        # Then: The result should match the expected cleaned text
        assert result == expected

    @pytest.fixture
    def char_tokenizer(self) -> PreTrainedTokenizer:
        return CharTokenizer()

    @pytest.mark.parametrize(
        "sample,min_words,min_repeat,expected",
        [
            (
                "1000000",
                1,
                2,
                "10",
            ),
        ],
    )
    def test_clean_repeated_concatenated_tokens(
        self, sample: str, min_words: int, min_repeat: int, expected: str, char_tokenizer: PreTrainedTokenizer
    ) -> None:
        # Given: A text sample with concatenated tokens and a char tokenizer
        # When: Cleaning repeated n-grams using the char tokenizer
        result: str = clean_repeated_ngrams(
            sample, min_words=min_words, min_repeat=min_repeat, tokenizer=char_tokenizer
        )
        # Then: The result should match the expected cleaned text
        assert result == expected
