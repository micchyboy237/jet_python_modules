import unittest
from jet.code.extraction.extract_comments import extract_comments, remove_comments


class TestExtractComments(unittest.TestCase):
    def test_extract_single_line_comment(self):
        code = """
x = 1  # This is a comment
y = 2
"""
        expected = [("This is a comment", 2)]
        result = extract_comments(code)
        self.assertEqual(result, expected)

    def test_extract_multiple_comments(self):
        code = """
# First comment
x = 1
# Second comment
y = 2  # Inline comment
"""
        expected = [
            ("First comment", 2),
            ("Second comment", 4),
            ("Inline comment", 5)
        ]
        result = extract_comments(code)
        self.assertEqual(result, expected)

    def test_extract_comments_in_triple_quoted_string(self):
        code = '''
def func():
    """This is a docstring with # hashtag"""
    # Real comment
    return True
'''
        expected = [("Real comment", 4)]
        result = extract_comments(code)
        self.assertEqual(result, expected)

    def test_extract_no_comments(self):
        code = """
x = 1
y = 2
"""
        expected = []
        result = extract_comments(code)
        self.assertEqual(result, expected)

    def test_extract_malformed_code(self):
        code = """
x = 1
"unclosed string
# Comment
"""
        expected = []
        result = extract_comments(code)
        self.assertEqual(result, expected)

    def test_remove_single_line_comment(self):
        code = """
x = 1  # This is a comment
y = 2
"""
        expected = """
x = 1
y = 2"""
        result = remove_comments(code)
        self.assertEqual(result.strip(), expected.strip())

    def test_remove_comments_preserve_triple_quoted(self):
        code = '''
def func():
    """This is a docstring with # hashtag"""
    # Real comment
    return True
'''
        expected = '''
def func():
    """This is a docstring with # hashtag"""
    return True'''
        result = remove_comments(code)
        self.assertEqual(result.strip(), expected.strip())

    def test_remove_comments_empty_code(self):
        code = ""
        expected = ""
        result = remove_comments(code)
        self.assertEqual(result, expected)

    def test_remove_comments_malformed_code(self):
        code = """
x = 1
"unclosed string
# Comment
"""
        result = remove_comments(code)
        self.assertEqual(result, code)  # Should return original code

    def test_remove_comments_preserve_newlines(self):
        code = """
x = 1
# Comment
y = 2

z = 3
"""
        expected = """x = 1
y = 2
z = 3"""
        result = remove_comments(code)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
