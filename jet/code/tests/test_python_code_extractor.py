import unittest
from pathlib import Path
import tempfile
import os
from jet.code.python_code_extractor import strip_comments, remove_comments


class TestCommentRemoval(unittest.TestCase):
    def test_strip_comments_single_line(self):
        """Test handling of single-line comments starting with '#'"""
        code = """
# This is a single-line comment
x = 1  # Inline comment
y = 2
"""
        expected = """
x = 1
y = 2
"""
        result = strip_comments(code)
        self.assertEqual(result, expected.strip())

    def test_strip_comments_docstring(self):
        """Test keeping only # comments in docstrings"""
        code = '''
def func():
    """
    This is a docstring
    # Keep this comment
    Another line
    """
    # Regular comment
    return True
'''
        expected = '''
def func():
    # Keep this comment
    return True
'''
        result = strip_comments(code)
        self.assertEqual(result, expected.strip())

    def test_strip_comments_multiline_docstring(self):
        """Test keeping # comments in multiline docstrings"""
        code = '''
class MyClass:
    """
    This is a multiline
    # Keep this
    docstring
    # And this
    """
    # Class comment
    def method(self):
        return None
'''
        expected = '''
class MyClass:
    # Keep this
    # And this
    def method(self):
        return None
'''
        result = strip_comments(code)
        self.assertEqual(result, expected.strip())

    def test_multiline_double_quoted_docstring(self):
        """Ensure multiple double quoted lines are correctly processed"""
        code = '''
    def example():
        """
        This function does something.

        # This should be kept
        More explanation here.
        # This too
        """
        x = 5  # Inline comment

        """
        # 2nd triple double quotes

        More explanation here.
        """
        return x
    '''
        expected = '''
    def example():
        # This should be kept
        # This too
        x = 5
        # 2nd triple double quotes
        return x
    '''
        result = strip_comments(code)
        self.assertEqual(result, expected.strip())

    def test_strip_comments_inline_comments(self):
        """Test handling of inline comments"""
        code = """
x = 1  # Inline comment
y = 2  # Another inline comment
# Keep this
"""
        expected = """
x = 1
y = 2
"""
        result = strip_comments(code)
        self.assertEqual(result, expected.strip())

    def test_strip_comments_empty_lines(self):
        """Test handling of extra empty lines"""
        code = '''
"""
x = 1


# Comment

y = 2
"""
'''
        expected = """
# Comment
"""
        result = strip_comments(code)
        self.assertEqual(result, expected.strip())

    def test_strip_comments_empty_input(self):
        """Test handling of empty input string"""
        result = strip_comments("")
        self.assertEqual(result, "")

    def test_remove_comments_string_input(self):
        """Test remove_comments with string input"""
        code = '''
def func():
    """Docstring"""
    # Keep this
    x = 1  # Inline
    return x
'''
        expected = '''
def func():
    """Docstring"""
    x = 1
    return x
'''
        result = remove_comments(code)
        self.assertEqual(result, expected.strip())

    def test_remove_comments_invalid_syntax(self):
        """Test handling of invalid Python syntax"""
        code = "def func():  # Missing colon"
        result = remove_comments(code)
        self.assertEqual(result, "def func():")

    def test_remove_comments_file(self):
        """Test remove_comments with file input"""
        code = '''
def func():
    """Docstring"""
    # Keep this
    return True
'''
        expected = '''
def func():
    """Docstring"""
    return True
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = remove_comments(temp_file)
            # Successful file modification returns None
            self.assertIsNone(result)
            with open(temp_file, 'r') as f:
                modified_content = f.read()
            self.assertEqual(modified_content, expected.strip())
        finally:
            os.unlink(temp_file)

    def test_remove_comments_file_not_found(self):
        """Test handling of non-existent file"""
        result = remove_comments("non_existent.py")
        self.assertEqual(result, "non_existent.py")

    def test_remove_comments_empty_input(self):
        """Test handling of empty input string"""
        result = strip_comments("")
        self.assertEqual(result, "")


if __name__ == '__main__':
    unittest.main()
