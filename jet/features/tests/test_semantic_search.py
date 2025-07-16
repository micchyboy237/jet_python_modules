import pytest
from unittest.mock import Mock, patch
from typing import List, Optional, Tuple, TypedDict
import torch
from jet.features.semantic_search import TextProcessor, SimilaritySearch, search_content, SearchResult, SearchOutput, TokenBreakdown
from jet.logger import logger

# Mock NLTK and spaCy dependencies


@pytest.fixture
def mock_nltk_spacy():
    with patch('jet.features.semantic_search.nltk') as mock_nltk, \
            patch('jet.features.semantic_search.spacy') as mock_spacy:
        mock_nltk.word_tokenize.return_value = ["test", "content", "ai"]
        mock_nltk.pos_tag.return_value = [
            ("test", "NN"), ("content", "NN"), ("ai", "NN")]
        mock_nltk.WordNetLemmatizer().lemmatize.side_effect = lambda word, pos: word
        mock_nltk.sent_tokenize.return_value = [
            "This is a test sentence.", "Another sentence."]
        mock_nltk.stopwords.words.return_value = ["is", "a"]

        # Mock spaCy document
        mock_doc = Mock()
        mock_doc.ents = []
        mock_doc.noun_chunks = [Mock(text="test content")]
        mock_spacy.load.return_value.return_value = mock_doc

        logger.debug("Mocked NLTK and spaCy dependencies")
        yield


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    # Define token mappings for specific texts
    token_map = {
        "Test Header": [101, 1000, 1001, 102],  # 4 tokens
        # 8 tokens
        "Test Header\nThis is a test sentence.": [101, 1000, 1001, 1002, 1003, 1004, 1005, 102],
        # 6 tokens
        "Test Header\nAnother sentence.": [101, 1000, 1001, 1006, 1007, 102],
        # 10 tokens
        "Test Header\nThis is a test sentence. Another sentence.": [101, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 102],
        # 50 tokens
        "This is a very long header that needs truncation": [101] + [1000] * 48 + [102],
        # 5 tokens
        "AI is transforming industries.": [101, 1008, 1009, 1010, 102],
        "AI advancements": [101, 1008, 1011, 102],  # 4 tokens
        "ai industry": [101, 1008, 1012, 102],  # 4 tokens
        "Test query": [101, 1013, 1014, 102],  # 4 tokens
        "Test content": [101, 1000, 1002, 102],  # 4 tokens
        "Other content": [101, 1015, 1002, 102],  # 4 tokens
        "test": [101, 1000, 102],  # 3 tokens
        "Test content test": [101, 1000, 1002, 1000, 102],  # 5 tokens
        "Other content other": [101, 1015, 1002, 1016, 102]  # 5 tokens
    }

    def encode_side_effect(text, add_special_tokens=True):
        logger.debug(f"Encoding text: {text}")
        return token_map.get(text, [101, 102])

    def tokenizer_side_effect(*args, add_special_tokens=True, return_tensors='pt', **kwargs):
        text = args[0] if args else ""
        if isinstance(text, list):
            # Handle list of texts for get_embeddings
            result = {
                'input_ids': torch.tensor([token_map.get(t, [101, 102]) for t in text]),
                'attention_mask': torch.tensor([[1] * len(token_map.get(t, [101, 102])) for t in text])
            }
        else:
            result = {
                'input_ids': torch.tensor([token_map.get(text, [101, 102])]),
                'attention_mask': torch.tensor([[1] * len(token_map.get(text, [101, 102]))])
            }
        logger.debug(f"Tokenizer output for {text}: {result}")
        return result

    def decode_side_effect(ids, skip_special_tokens=True):
        ids_tuple = tuple(ids.tolist() if isinstance(
            ids, torch.Tensor) else ids)
        decode_map = {
            # For max_length // 2 = 25
            tuple([101] + [1000] * 23 + [102]): "Truncated header",
            tuple([101, 1000, 1001, 102]): "Test Header",
            tuple([101, 1000, 1001, 1002, 1003, 1004, 1005, 102]): "Test Header This is a test sentence",
            tuple([101, 1000, 1001, 1006, 1007, 102]): "Test Header Another sentence",
            tuple([101, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 102]): "Test Header This is a test sentence Another sentence",
            tuple([101, 1008, 1009, 1010, 102]): "AI is transforming industries",
            tuple([101, 1008, 1011, 102]): "AI advancements",
            tuple([101, 1008, 1012, 102]): "ai industry",
            tuple([101, 1013, 1014, 102]): "Test query",
            tuple([101, 1000, 1002, 102]): "Test content",
            tuple([101, 1015, 1002, 102]): "Other content",
            tuple([101, 1000, 102]): "test",
            tuple([101, 1000, 1002, 1000, 102]): "Test content test",
            tuple([101, 1015, 1002, 1016, 102]): "Other content other"
        }
        decoded = decode_map.get(ids_tuple, "decoded text")
        logger.debug(f"Decoding {ids_tuple} to: {decoded}")
        return decoded

    tokenizer.encode.side_effect = encode_side_effect
    tokenizer.side_effect = tokenizer_side_effect
    tokenizer.decode.side_effect = decode_side_effect
    return tokenizer


@pytest.fixture
def mock_model():
    model = Mock()
    # Return embeddings with shape [batch_size, seq_length, hidden_size]

    def model_side_effect(**kwargs):
        input_ids = kwargs['input_ids']
        batch_size, seq_length = input_ids.shape
        # Return tensor of shape [batch_size, seq_length, 2] to match attention_mask
        return [torch.ones(batch_size, seq_length, 2), None, None]

    model.side_effect = model_side_effect
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
        expected = "Truncated header"

        # When: Truncating the header
        result = text_processor.truncate_header(header)
        logger.debug(f"Truncated header: {result}")

        # Then: Header is truncated
        assert result == expected, f"Expected '{expected}', but got '{result}'"

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
        expected = [["ai", "content", "test", "test content"]]

        # When: Generating tags
        result = text_processor.generate_tags(texts)
        logger.debug(f"Generated tags: {result}")

        # Then: Tags are extracted correctly
        assert result == expected, f"Expected {expected}, but got {result}"

    @pytest.mark.filterwarnings("ignore:.*split_arg_string.*:DeprecationWarning")
    def test_preprocess_text_splits_into_segments(self, text_processor, mock_nltk_spacy, mock_tokenizer):
        # Given: A content string with header and multiple sentences
        content = "This is a test sentence. Another sentence."
        header = "Test Header"
        expected = [
            ("Test Header\nThis is a test sentence.", [
             "ai", "content", "test", "test content"]),
            ("Test Header\nAnother sentence.", [
             "ai", "content", "test", "test content"])
        ]

        # When: Preprocessing the text
        result = text_processor.preprocess_text(content, header)
        logger.debug(f"Preprocessed segments: {result}")

        # Then: Text is split into segments with tags
        assert len(result) == 2, f"Expected 2 segments, got {len(result)}"
        assert result == expected, f"Expected {expected}, but got {result}"

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
        assert result == expected


class TestSimilaritySearch:
    def test_search_returns_top_k_results(self, similarity_search, mock_tokenizer, mock_model):
        # Given: A query and text tuples
        query = "Test query"
        text_tuples = [("Test content", ["test"]),
                       ("Other content", ["other"])]
        expected = [
            {
                'rank': 1,
                'score': pytest.approx(0.998, 0.01),
                'doc_index': 0,
                'text': "Test content",
                'header': None,
                'tags': ["test"],
                'tokens': {'header_tokens': 0, 'text_tokens': 4, 'tags_tokens': 3}
            }
        ]

        # When: Performing similarity search
        result = similarity_search.search(
            query, text_tuples, top_k=1, threshold=0.5)
        logger.debug(f"Search results: {result}")

        # Then: Top-k results are returned with correct structure
        assert len(result) == 1, f"Expected 1 result, got {len(result)}"
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
                    'tags': ["ai", "content", "test", "test content"],
                    'tokens': {'header_tokens': 4, 'text_tokens': 5, 'tags_tokens': 5}
                }
            ],
            'cls_token_results': [
                {
                    'rank': 1,
                    'score': pytest.approx(0.998, 0.01),
                    'doc_index': 0,
                    'text': "AI is transforming industries.",
                    'header': "Header",
                    'tags': ["ai", "content", "test", "test content"],
                    'tokens': {'header_tokens': 4, 'text_tokens': 5, 'tags_tokens': 5}
                }
            ],
            'mean_pooling_text': "AI is transforming industries.",
            'cls_token_text': "AI is transforming industries.",
            'mean_pooling_tokens': 5,
            'cls_token_tokens': 5
        }

        # When: Running search_content
        result = search_content(query, content, top_k=1, threshold=0.5,
                                min_length=10, max_length=50, max_result_tokens=100)
        logger.debug(f"Search content result: {result}")

        # Then: Expected SearchOutput is returned
        assert result['mean_pooling_results'][0]['text'] == expected['mean_pooling_results'][0]['text']
        assert result['cls_token_results'][0]['text'] == expected['cls_token_results'][0]['text']
        assert result['mean_pooling_text'] == expected['mean_pooling_text']
        assert result['cls_token_text'] == expected['cls_token_text']
        assert result['mean_pooling_tokens'] == expected['mean_pooling_tokens']
        assert result['cls_token_tokens'] == expected['cls_token_tokens']
