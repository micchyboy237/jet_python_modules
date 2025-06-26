from pathlib import Path
from typing import List, Tuple, Union
from jet.data.header_docs import HeaderDocs, HeaderNode, TextNode
import numpy as np


def example_1_technical_documentation():
    """Example: Searching a technical Markdown document."""
    markdown_content = """
# API Documentation
## Authentication
To authenticate, use a Bearer token in the Authorization header.

### Token Generation
```bash
curl -X POST https://api.example.com/token -d "username=user&password=pass"
```

## Endpoints
### GET /users
Returns a list of users.

#### Parameters
- limit: Number of users to return (default: 10)
- offset: Starting point (default: 0)

### POST /users
Creates a new user.

#### Request Body
```json
{
  "name": "string",
  "email": "string"
}
```
"""
    # Parse Markdown and build HeaderDocs
    docs = HeaderDocs.from_string(markdown_content)

    # Search for content related to "authentication"
    results: List[Tuple[Union[HeaderNode, TextNode], float]
                  ] = docs.search("authentication", top_k=3)

    print("Example 1: Searching for 'authentication'")
    for node, score in results:
        content = node.title if isinstance(node, HeaderNode) else node.content
        node_type = "Header" if isinstance(node, HeaderNode) else "Text"
        print(f"[{node_type}] Score: {score:.3f}, Content: {content[:50]}...")


def example_2_project_notes():
    """Example: Searching project notes with mixed content."""
    markdown_content = """
# Project Alpha
## Sprint 1
- [x] Set up CI/CD pipeline
- [ ] Implement user authentication
- [ ] Design database schema

## Sprint 2
### Database Optimization
Consider indexing for faster queries.

#### Index Recommendations
| Field | Type | Index Type |
|-------|------|------------|
| user_id | UUID | BTREE |
| email | TEXT | GIN |

### API Development
Focus on RESTful endpoints for user management.
"""
    # Parse Markdown and build HeaderDocs
    docs = HeaderDocs.from_string(markdown_content)

    # Search for content related to "database"
    results: List[Tuple[Union[HeaderNode, TextNode], float]
                  ] = docs.search("database", top_k=2)

    print("\nExample 2: Searching for 'database'")
    for node, score in results:
        content = node.title if isinstance(node, HeaderNode) else node.content
        node_type = "Header" if isinstance(node, HeaderNode) else "Text"
        print(f"[{node_type}] Score: {score:.3f}, Content: {content[:50]}...")


def example_3_file_input():
    """Example: Searching a Markdown file from disk."""
    # Assume a file named 'readme.md' exists
    markdown_file = Path("readme.md")
    markdown_content = """
# Getting Started
Install dependencies using pip.

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
Set environment variables:
- DATABASE_URL
- API_KEY
"""
    # Write sample content to a file (for demonstration)
    markdown_file.write_text(markdown_content)

    # Parse Markdown from file
    docs = HeaderDocs.from_string(markdown_file)

    # Search for content related to "installation"
    results: List[Tuple[Union[HeaderNode, TextNode], float]
                  ] = docs.search("installation", top_k=2)

    print("\nExample 3: Searching for 'installation' in file")
    for node, score in results:
        content = node.title if isinstance(node, HeaderNode) else node.content
        node_type = "Header" if isinstance(node, HeaderNode) else "Text"
        print(f"[{node_type}] Score: {score:.3f}, Content: {content[:50]}...")


if __name__ == "__main__":
    print("Running vector search examples...")
    example_1_technical_documentation()
    example_2_project_notes()
    example_3_file_input()
