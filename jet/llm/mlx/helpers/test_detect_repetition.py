import pytest
from .detect_repetition import find_repeated_consecutive_ngrams, NgramRepeat


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
                    end_index=6,
                    full_end_index=14,
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
                    end_index=6,
                    full_end_index=14,
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
                    end_index=9,
                    full_end_index=12,
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
                    end_index=4,
                    full_end_index=29,
                    num_of_repeats=5,
                ),
                NgramRepeat(
                    ngram="again again",
                    start_index=0,
                    end_index=10,
                    full_end_index=22,
                    num_of_repeats=2,
                ),
            ],
        ),
    ],
)
def test_find_repeated_consecutive_ngrams(sample, min_words, max_words, min_repeat, expected):
    result = find_repeated_consecutive_ngrams(
        sample, min_words=min_words, max_words=max_words, min_repeat=min_repeat
    )
    assert result == expected


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
                    end_index=6,
                    full_end_index=14,
                    num_of_repeats=2,
                )
            ],
        ),
        (
            "This is THIS is a repeated phrase.",
            True,
            [],  # Case-sensitive: "This" != "THIS"
        ),
    ],
)
def test_find_repeated_case_sensitive(sample, case_sensitive, expected):
    result = find_repeated_consecutive_ngrams(
        sample, case_sensitive=case_sensitive)
    assert result == expected
