import textwrap
import unittest
import os
from jet.validation.python_validation import validate_python_syntax


class TestPythonSyntaxValidator(unittest.TestCase):

    def test_valid_code_string(self):
        sample = """def greet():\n    print('Hello')"""
        expected = None
        result = validate_python_syntax(sample)
        self.assertEqual(result, expected)

    def test_invalid_code_string(self):
        sample = """def greet()\n    print('Hello')"""
        result = validate_python_syntax(sample)
        self.assertEqual(len(result), 1)
        self.assertIn("SyntaxError", result[0]["message"])

    def test_multiple_sources(self):
        valid = """def a():\n    return 1"""
        invalid = """def oops(:\n    pass"""
        result = validate_python_syntax([valid, invalid])
        self.assertEqual(len(result), 1)
        self.assertTrue("SyntaxError" in result[0]["message"])

    def test_valid_pydantic_model(self):
        sample = """from pydantic import BaseModel\n\nclass User(BaseModel):\n    id: int\n    name: str"""
        expected = None
        result = validate_python_syntax(sample)
        self.assertEqual(result, expected)

    def test_invalid_pydantic_model(self):
        sample = """from pydantic import BaseModel\n\nclass User(BaseModel):\n    id int\n    name: str"""
        result = validate_python_syntax(sample)
        self.assertEqual(len(result), 1)
        self.assertIn("SyntaxError", result[0]["message"])

    def test_empty_string(self):
        sample = ""
        expected = None
        result = validate_python_syntax(sample)
        self.assertEqual(result, expected)

    def test_bad_indentation(self):
        sample = """def f():\nprint("No indent")"""
        result = validate_python_syntax(sample)
        self.assertEqual(len(result), 1)
        self.assertIn("SyntaxError", result[0]["message"])


def test_long_code(self):
    sample = """
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
    """
    # Use textwrap.dedent to remove any unintended leading indentation
    sample = textwrap.dedent(sample).strip()

    expected = None
    result = validate_python_syntax(sample)
    self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
