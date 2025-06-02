import pytest
from .detect_repetition import find_repeated_consecutive_ngrams, NgramRepeat


@pytest.mark.parametrize("sample,expected", [
    (
        "this is this is a repeated phrase.",
        [
            NgramRepeat(
                ngram="this is",
                start_index=0,
                end_index=6,
                full_end_index=14,  # covers "this is this is"
                num_of_repeats=2,
            )
        ]
    ),
    (
        "the cat the cat sat on the mat.",
        [
            NgramRepeat(
                ngram="the cat",
                start_index=0,
                end_index=6,
                full_end_index=14,  # covers "the cat the cat"
                num_of_repeats=2,
            )
        ]
    ),
    (
        "we need to to go now.",
        [
            NgramRepeat(
                ngram="to",
                start_index=8,
                end_index=9,
                full_end_index=12,  # covers "to to"
                num_of_repeats=2,
            )
        ]
    ),
    (
        "nothing here is repeated.",
        []
    ),
    (
        "again again again again again.",
        [
            NgramRepeat(
                ngram="again",
                start_index=0,
                end_index=4,
                full_end_index=22,
                num_of_repeats=4,
            ),
            NgramRepeat(
                ngram="again again",
                start_index=0,
                end_index=10,
                full_end_index=22,
                num_of_repeats=2,
            ),
            NgramRepeat(
                ngram="again",
                start_index=6,
                end_index=10,
                full_end_index=22,
                num_of_repeats=3,
            ),
            NgramRepeat(
                ngram="again",
                start_index=12,
                end_index=16,
                full_end_index=22,
                num_of_repeats=2,
            ),
        ],
    ),

])
def test_find_repeated_consecutive_ngrams(sample, expected):
    result = find_repeated_consecutive_ngrams(sample.lower())
    assert result == expected
