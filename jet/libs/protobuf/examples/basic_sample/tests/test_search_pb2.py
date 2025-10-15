"""Tests for search protobuf serialization."""
import pytest
from pathlib import Path

from src.search_example import create_search_request, create_search_response
from src.serialization_utils import save_proto_to_file, load_proto_from_file
from src.generated.proto.search_pb2 import SearchRequest, Result, Corpus


class TestSearchSerialization:
    """Test search request/response serialization."""
    
    @pytest.fixture
    def sample_request(self):
        """Sample search request fixture."""
        return create_search_request("test query", page=2, results_per_page=20)
    
    @pytest.fixture
    def sample_result(self):
        """Sample result fixture."""
        result = Result()
        result.url = "https://example.com"
        result.title = "Test Result"
        result.snippets.append("snippet 1")
        result.snippets.append("snippet 2")
        return result
    
    def test_basic_request_serialization(self, sample_request):
        # Given
        expected_query = "test query"
        expected_page = 2
        expected_results_per_page = 20
        
        # When
        temp_file = Path("test_request.bin")
        save_proto_to_file(sample_request, temp_file)
        
        # Then
        loaded = load_proto_from_file(SearchRequest, temp_file)
        assert loaded.query == expected_query
        assert loaded.page_number == expected_page
        assert loaded.results_per_page == expected_results_per_page
        assert loaded.HasField("geo")
        assert len(loaded.geo.coordinates) == 2
        
        temp_file.unlink()  # Cleanup
    
    def test_optional_fields(self, sample_request):
        # Given
        request_without_corpus = SearchRequest()
        request_without_corpus.query = "no corpus"
        
        # When & Then
        assert not request_without_corpus.HasField("corpus")
        assert request_without_corpus.corpus == Corpus.CORPUS_UNSPECIFIED
        
        # Verify sample has corpus
        assert sample_request.HasField("corpus")
        assert sample_request.corpus == Corpus.CORPUS_WEB
    
    def test_response_oneof(self, sample_result):
        # Given
        response_with_token = create_search_response([sample_result], use_token=True)
        response_with_number = create_search_response([sample_result], use_token=False)
        
        # When & Then
        assert response_with_token.WhichOneof("next_page") == "next_page_token"
        assert response_with_number.WhichOneof("next_page") == "next_page_number"
        assert len(response_with_token.results) == 1
        assert response_with_token.results[0].url == sample_result.url