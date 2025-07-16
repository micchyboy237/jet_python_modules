import pytest
from unittest.mock import Mock, patch
from typing import List, Optional, Tuple, TypedDict
import torch
from jet.features.semantic_search import TextProcessor, SimilaritySearch, search_content, SearchResult, SearchOutput, TokenBreakdown

# Mock NLTK and spaCy dependencies


@pytest.fixture
def mock_nltk_spacy():
    with patch('jet.features.semantic_search.nltk') as mock_nltk, \
            patch('jet.features.semantic_search.spacy') as mock_spacy:
        mock_nltk.word_tokenize.return_value = ["test", "content"]
        mock_nltk.pos_tag.return_value = [("test", "NN"), ("content", "NN")]
        mock_nltk.WordNetLemmatizer().lemmatize.side_effect = lambda word, pos: word
        mock_nltk.sent_tokenize.return_value = [
            "This is a test sentence.", "Another sentence."]
        mock_nltk.stopwords.words.return_value = ["is", "a"]
        mock_spacy.load.return_value.__call__.return_value.ents = []
        mock_spacy.load.return_value.__call__.return_value.noun_chunks = [
            Mock(text="test content")
        ]
        yield


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.encode.return_value = [101, 102]  # Mock token IDs
    tokenizer.return_value = {'input_ids': torch.tensor([[101, 102]])}
    tokenizer.decode.return_value = "decoded text"
    return tokenizer


@pytest.fixture
def mock_model():
    model = Mock()
    model.return_value = [torch.tensor(
        [[[0.1, 0.2]]]), None, None]  # Mock embeddings
    return model


@pytest.fixture
def text_processor(mock_tokenizer, mock_model):
    return TextProcessor(tokenizer=mock_tokenizer, model=mock_model, min_length=10, max_length=50, debug=False)


@pytest.fixture
def similarity_search(mock_tokenizer, mock_model):
    return SimilaritySearch(model=mock_model, tokenizer=mock_tokenizer, max_length=50)


class TestTextProcessor:
    def test_clean_text_removes_markdown_and_extra_spaces(self, text_processor):
        # Given: A text with markdown, extra spaces, and newlines
        input_text = "# Header\n[Link]\n  Extra   spaces\n\nMultiple\nnewlines"
        expected = "Header Extra spaces Multiple newlines"

        # When: Cleaning the text
        result = text_processor.clean_text(input_text)

        # Then: Markdown and extra spaces/newlines are removed
        assert result == expected

    def test_truncate_header_within_limit(self, text_processor, mock_tokenizer):
        # Given: A short header within token limit
        header = "Short header"
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 102, 103]])}
        expected = "Short header"

        # When: Truncating the header
        result = text_processor.truncate_header(header)

        # Then: Header is unchanged
        assert result == expected

    def test_truncate_header_exceeds_limit(self, text_processor, mock_tokenizer):
        # Given: A long header exceeding token limit
        header = "This is a very long header that needs truncation"
        mock_tokenizer.return_value = {'input_ids': torch.tensor([[101] * 50])}
        mock_tokenizer.decode.return_value = "Truncated header"
        expected = "Truncated header"

        # When: Truncating the header
        result = text_processor.truncate_header(header)

        # Then: Header is truncated
        assert result == expected

    def test_get_tokens_success(self, text_processor, mock_tokenizer):
        # Given: A valid text input
        text = "Sample text"
        expected = [101, 102]

        # When: Getting tokens
        mock_tokenizer.return_value = {'input_ids': torch.tensor([[101, 102]])}
        result = text_processor.get_tokens(text)

        # Then: Correct tokens are returned
        assert result == expected

    def test_generate_tags_extracts_keywords(self, text_processor, mock_nltk_spacy):
        # Given: A list of texts with meaningful words
        texts = ["Test content about AI"]
        expected = [["test", "content"]]

        # When: Generating tags
        result = text_processor.generate_tags(texts)

        # Then: Tags are extracted correctly
        assert result == expected

    def test_preprocess_text_splits_into_segments(self, text_processor, mock_nltk_spacy, mock_tokenizer):
        # Given: A content string with header and multiple sentences
        content = "This is a test sentence. Another sentence."
        header = "Test Header"
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 102, 103]])}
        expected = [
            ("Test Header\nThis is a test sentence.", ["test", "content"]),
            ("Test Header\nAnother sentence.", ["test", "content"])
        ]

        # When: Preprocessing the text
        result = text_processor.preprocess_text(content, header)

        # Then: Text is split into segments with tags
        assert len(result) == 2
        assert result == expected

    def test_preprocess_query_cleans_and_truncates(self, text_processor, mock_tokenizer):
        # Given: A query with special characters and extra spaces
        query = "  Test [link] query!!  "
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 102, 103]])}
        mock_tokenizer.decode.return_value = "Test query"
        expected = "Test query"

        # When: Preprocessing the query
        result = text_processor.preprocess_query(query)

        # Then: Query is cleaned and within limit
        assert result == molded


class TestSimilaritySearch:
    def test_search_returns_top_k_results(self, similarity_search, mock_tokenizer, mock_model):
        # Given: A query and text tuples
        query = "Test query"
        text_tuples = [("Test content", ["test"]),
                       ("Other content", ["other"])]
        mock_tokenizer.return_value = {'input_ids': torch.tensor(
            [[101, 102]]), 'attention_mask': torch.tensor([[1, 1]])}
        mock_model.return_value = [torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4]]]), None, None]
        expected = [
            {
                'rank': 1,
                'score': pytest.approx(0.998, 0.01),
                'doc_index': 0,
                'text': "Test content",
                'header': None,
                'tags': ["test"],
                'tokens': {'header_tokens': 0, 'text_tokens': 2, 'tags_tokens': 2}
            }
        ]

        # When: Performing similarity search
        result = similarity_search.search(
            query, text_tuples, top_k=1, threshold=0.5)

        # Then: Top-k results are returned with correct structure
        assert len(result) == 1
        assert result[0]['rank'] == expected[0]['rank']
        assert result[0]['doc_index'] == expected[0]['doc_index']
        assert result[0]['text'] == expected[0]['text']
        assert result[0]['tags'] == expected[0]['tags']
        assert result[0]['tokens'] == expected[0]['tokens']


class TestSearchContent:
    def test_search_content_returns_expected_output(self, mock_tokenizer, mock_model, mock_nltk_spacy):
        # Given: A query and content
        query = "AI advancements"
        content = "Header\nAI is transforming industries. New algorithms improve efficiency."
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_model.from_pretrained.return_value = mock_model
        mock_tokenizer.return_value = {'input_ids': torch.tensor(
            [[101, 102]]), 'attention_mask': torch.tensor([[1, 1]])}
        mock_model.return_value = [torch.tensor([[[0.1, 0.2]]]), None, None]
        expected: SearchOutput = {
            'mean_pooling_results': [
                {
                    'rank': 1,
                    'score': pytest.approx(0.998, 0.01),
                    'doc_index': 0,
                    'text': "AI is transforming industries.",
                    'header': "Header",
                    'tags': ["ai", "industry"],
                    'tokens': {'header_tokens': 2, 'text_tokens': 2, 'tags_tokens': 2}
                }
            ],
            'cls_token_results': [
                {
                    'rank': 1,
                    'score': pytest.approx(0.998, 0.01),
                    'doc_index': 0,
                    'text': "AI is transforming industries.",
                    'header': "Header",
                    'tags': ["ai", "industry"],
                    'tokens': {'header_tokens': 2, 'text_tokens': 2, 'tags_tokens': 2}
                }
            ],
            'mean_pooling_text': "AI is transforming industries.",
            'cls_token_text': "AI is transforming industries.",
            'mean_pooling_tokens': 2,
            'cls_token_tokens': 2
        }

        # When: Running search_content
        result = search_content(query, content, top_k=1, threshold=0.5,
                                min_length=10, max_length=50, max_result_tokens=100)

        # Then: Expected SearchOutput is returned
        assert result['mean_pooling_results'][0]['text'] == expected['mean_pooling_results'][0]['text']
        assert result['cls_token_results'][0]['text'] == expected['cls_token_results'][0]['text']
        assert result['mean_pooling_text'] == expected['mean_pooling_text']
        assert result['cls_token_text'] == expected['cls_token_text']
        assert result['mean_pooling_tokens'] == expected['mean_pooling_tokens']
        assert result['cls_token_tokens'] == expected['cls_token_tokens']
