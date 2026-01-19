import unittest
from jet.utils.markdown import extract_block_content


class TestExtractBlockContent(unittest.TestCase):

    def test_extract_code_block_after_text(self):
        text = "Here is a code block:\n```python\nprint('Hello World!')\n```"
        expected = "print('Hello World!')"
        result = extract_block_content(text)
        self.assertEqual(result, expected)

    def test_extract_long_code_block(self):
        text = """```python
from pydantic import BaseModel, Field
from typing import Optional, List

class Answer(BaseModel):
    title: str = Field(
        ...,
        description="The exact title of the anime, as it appears in the document."
    )
    document_number: int = Field(
        ...,
        description="The number of the document that includes this anime (e.g., 'Document number: 3')."
    )
    release_year: Optional[int] = Field(
        None,
        description="The most recent known release year of the anime, if specified in the document."
    )

class QueryResponse(BaseModel):
    results: List[Answer] = Field(
        default_factory=list,
        description="List of relevant anime titles extracted from the documents, matching the user's query.\nEach entry includes the title, source document number, and release year (if known)."
    )
```"""
        expected = """from pydantic import BaseModel, Field
from typing import Optional, List

class Answer(BaseModel):
    title: str = Field(
        ...,
        description="The exact title of the anime, as it appears in the document."
    )
    document_number: int = Field(
        ...,
        description="The number of the document that includes this anime (e.g., 'Document number: 3')."
    )
    release_year: Optional[int] = Field(
        None,
        description="The most recent known release year of the anime, if specified in the document."
    )

class QueryResponse(BaseModel):
    results: List[Answer] = Field(
        default_factory=list,
        description="List of relevant anime titles extracted from the documents, matching the user's query.\nEach entry includes the title, source document number, and release year (if known)."
    )"""
        result = extract_block_content(text)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
