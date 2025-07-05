import nltk
from jet.models.embeddings.utils import (
    chunk_headers_by_hierarchy,
)


class TestMergeTextsByHierarchy:
    def test_chunk_headers_by_hierarchy_with_root(self):
        # Given
        markdown_text = "# Root1\nThis is a sentence in root.\n## Head 1\nThis is a very long sentence that fits chunksize. Short sentence.\nJoined short sentence for merging.\n### Head 2\nThis is another long sentence.\n### Head3\nThis is a long sibling sentence.\n# Root 2\n### Head3\nThis is the 5th long sentence."

        def tokenizer(x): return nltk.word_tokenize(x) if isinstance(
            x, str) else [nltk.word_tokenize(t) for t in x]
        split_fn = nltk.sent_tokenize
        chunk_size = 16

        # token_count = length of tokenized header + length of tokenized content
        expected = [
            {
                "content": "This is a sentence in root.",
                "token_count": 9,
                "header": "# Root1",
                "parent_header": None,
                "level": 1
            },
            {
                "content": "This is a very long sentence that fits chunksize.",
                "token_count": 16,
                "header": "## Head 1 - 1",
                "parent_header": "# Root1",
                "level": 2
            },
            {
                "content": "Short sentence.\nJoined short sentence for merging.",
                "token_count": 15,
                "header": "## Head 1 - 2",
                "parent_header": "# Root1",
                "level": 2
            },
            {
                "content": "This is another long sentence.",
                "token_count": 11,
                "header": "### Head 2",
                "parent_header": "## Head 1",
                "level": 3
            },
            {
                "content": "This is a long sibling sentence.",
                "token_count": 11,
                "header": "### Head3",
                "parent_header": "## Head 1",
                "level": 3
            },
            {
                "content": "This is the 5th long sentence.",
                "token_count": 11,
                "header": "### Head3",
                "parent_header": "# Root 2",
                "level": 3
            },
        ]

        # When
        results = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)

        # Then
        assert results == expected

    def test_chunk_headers_by_hierarchy_no_root(self):
        # Given
        markdown_text = "## Head 1\nThis is a very long sentence that fits chunksize. Short sentence.\nJoined short sentence for merging.\n### Head 2\nThis is another long sentence.\n### Head3\nThis is a long sibling sentence.\n# Root 2\n### Head3\nThis is the 5th long sentence."

        def tokenizer(x): return nltk.word_tokenize(x) if isinstance(
            x, str) else [nltk.word_tokenize(t) for t in x]
        split_fn = nltk.sent_tokenize
        chunk_size = 16

        # token_count = length of tokenized header + length of tokenized content
        expected = [
            {
                "content": "This is a very long sentence that fits chunksize.",
                "token_count": 16,
                "header": "## Head 1 - 1",
                "parent_header": None,
                "level": 2
            },
            {
                "content": "Short sentence.\nJoined short sentence for merging.",
                "token_count": 15,
                "header": "## Head 1 - 2",
                "parent_header": None,
                "level": 2
            },
            {
                "content": "This is another long sentence.",
                "token_count": 11,
                "header": "### Head 2",
                "parent_header": "## Head 1",
                "level": 3
            },
            {
                "content": "This is a long sibling sentence.",
                "token_count": 11,
                "header": "### Head3",
                "parent_header": "## Head 1",
                "level": 3
            },
            {
                "content": "This is the 5th long sentence.",
                "token_count": 11,
                "header": "### Head3",
                "parent_header": "# Root 2",
                "level": 3
            },
        ]

        # When
        results = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)

        # Then
        assert results == expected
