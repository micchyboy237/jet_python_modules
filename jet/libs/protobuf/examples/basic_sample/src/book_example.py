"""Book protobuf example with edition compatibility."""
from typing import Dict, List

# Try different import names
try:
    from src.generated.proto.book_pb2 import Book
    PROTO_SOURCE = "edition_2024"
except ImportError:
    try:
        from src.generated.proto.book_fallback_pb2 import Book
        PROTO_SOURCE = "proto3_fallback"
    except ImportError:
        raise ImportError("Could not import Book message. Run generate_protos.sh first")

print(f"Using Book from: {PROTO_SOURCE}")

def create_sample_book(title: str, metadata: Dict[str, str], tags: List[str]) -> Book:
    """Create a sample book with metadata map and tags."""
    book = Book()
    book.title = title
    
    for key, value in metadata.items():
        book.metadata[key] = value
    
    book.tags.extend(tags)
    
    return book

def demo_book_operations():
    """Demonstrate book protobuf operations."""
    metadata = {
        "author": "Google Engineering Team",
        "year": "2025",
        "isbn": "978-1-234567-89-0",
        "edition": "3rd"
    }
    
    book = create_sample_book(
        title="Modern Protobuf Guide",
        metadata=metadata,
        tags=["tech", "serialization", "gRPC"]
    )
    
    print(f"Title: {book.title}")
    print(f"Metadata keys: {list(book.metadata.keys())}")
    print(f"Tags: {list(book.tags)}")
    
    # Serialize with deterministic ordering
    from .serialization_utils import save_proto_to_file
    save_proto_to_file(book, "book_data.bin", deterministic=True)
    
    # Demonstrate map iteration
    print("\nMetadata:")
    for key, value in book.metadata.items():
        print(f"  {key}: {value}")

def load_and_validate_book(filepath: str) -> Book:
    """Load and validate book from file."""
    from .serialization_utils import load_proto_from_file
    book = load_proto_from_file(Book, filepath)
    
    if not book.title:
        raise ValueError("Book must have a title")
    
    print(f"Loaded book: {book.title}")
    print(f"Tag count: {len(book.tags)}")
    return book

if __name__ == "__main__":
    demo_book_operations()
