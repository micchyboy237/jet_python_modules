import pytest
from .detect_repetition import find_repeated_consecutive_ngrams, NgramRepeat


@pytest.mark.parametrize("sample,expected", [
    (
        "this is this is a repeated phrase.",
        [NgramRepeat(ngram="this is", start_index=0,
                     end_index=3, num_of_repeats=2)]
    ),
    (
        "the cat the cat sat on the mat.",
        [NgramRepeat(ngram="the cat", start_index=0,
                     end_index=3, num_of_repeats=2)]
    ),
    (
        "we need to to go now.",
        [NgramRepeat(ngram="to", start_index=2, end_index=3, num_of_repeats=2)]
    ),
    (
        "nothing here is repeated.",
        []
    ),
    (
        "again again again again again.",
        [NgramRepeat(ngram="again again", start_index=0,
                     end_index=3, num_of_repeats=2)]
    ),
])
def test_find_repeated_consecutive_ngrams(sample, expected):
    result = find_repeated_consecutive_ngrams(sample.lower())
    assert result == expected
