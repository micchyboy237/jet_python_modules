"""Search protobuf example demonstrating nested messages and oneof."""
from typing import List

from src.generated.proto.search_pb2 import (
    SearchRequest, SearchResponse, Result, Corpus
)

def create_search_request(query: str, page: int = 1, results_per_page: int = 10) -> SearchRequest:
    """Create a sample search request."""
    request = SearchRequest()
    request.query = query
    request.page_number = page
    request.results_per_page = results_per_page
    request.corpus = Corpus.CORPUS_WEB
    
    # Add geo information
    geo = request.geo
    geo.country = "US"
    geo.coordinates.extend([40.7128, -74.0060])  # NYC coordinates
    
    return request

def create_search_response(results: List[Result], use_token: bool = True) -> SearchResponse:
    """Create a sample search response with oneof pagination."""
    response = SearchResponse()
    
    for result in results:
        response.results.append(result)
    
    if use_token:
        response.next_page_token = "page_token_abc123"
    else:
        response.next_page_number = 2
    
    return response

def demo_search_workflow():
    """Complete search workflow example."""
    # Create request
    request = create_search_request("python protobuf tutorial")
    print(f"Query: {request.query}")
    print(f"Geo set: {request.HasField('geo')}")
    print(f"Coordinates: {list(request.geo.coordinates)}")
    
    # Serialize
    from .serialization_utils import save_proto_to_file
    save_proto_to_file(request, "search_request.bin")
    
    # Create mock response
    result1 = Result(
        url="example.com",
        title="Protobuf Guide",
        snippets=["Learn protobuf...", "Efficient serialization..."]
    )
    response = create_search_response([result1])
    
    # Check oneof
    oneof_field = response.WhichOneof("next_page")
    print(f"Oneof field: {oneof_field}")
    if oneof_field == "next_page_token":
        print(f"Next page token: {response.next_page_token}")
    
    print(f"Results count: {len(response.results)}")

if __name__ == "__main__":
    demo_search_workflow()