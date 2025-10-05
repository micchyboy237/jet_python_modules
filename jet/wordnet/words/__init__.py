from ._words import (
    setup_nlp,
    pos_tag_nltk,
    pos_tag_spacy,
    split_words,
    get_words,
    get_unique_words,
    get_non_words,
    count_words,
    count_non_words,
    process_dataset,
    process_all_datasets,
    compare_words,
    count_syllables,
    split_by_syllables,
    get_named_words,
    get_spacy_words,
    list_all_spacy_pos_tags,
)
# from ._plural_detection_inflect import is_plural_inflect
# from ._plural_detection_textblob import is_plural_textblob

__all__ = [
    "setup_nlp",
    "pos_tag_nltk",
    "pos_tag_spacy",
    "split_words",
    "get_words",
    "get_unique_words",
    "get_non_words",
    "count_words",
    "count_non_words",
    "process_dataset",
    "process_all_datasets",
    "compare_words",
    "count_syllables",
    "split_by_syllables",
    "get_named_words",
    "get_spacy_words",
    "list_all_spacy_pos_tags",

    # "is_plural_inflect",
    # "is_plural_textblob",
]
