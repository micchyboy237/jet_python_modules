import unittest

from jet.code.utils import remove_single_line_comments_preserving_triple_quotes


class TestRemoveSingleLineComments(unittest.TestCase):

    def test_remove_basic_comment(self):
        code = 'x = 1  # this is a comment\n'
        expected = 'x = 1\n'
        result = remove_single_line_comments_preserving_triple_quotes(code)
        self.assertEqual(result, expected)

    def test_preserve_triple_quoted_string(self):
        code = '''
"""
# Comment inside triple block
This is a triple-quoted string with a # hash that should stay
"""
x = 42  # this is a comment
print(x)  # print the value
'''
        expected = '''
"""
# Comment inside triple block
This is a triple-quoted string with a # hash that should stay
"""
x = 42
print(x)
'''
        result = remove_single_line_comments_preserving_triple_quotes(code)
        self.assertEqual(result, expected)

    def test_inline_string_with_hash(self):
        code = 'x = "# not a comment"\ny = 2  # real comment\n'
        expected = 'x = "# not a comment"\ny = 2\n'
        result = remove_single_line_comments_preserving_triple_quotes(code)
        self.assertEqual(result, expected)

    def test_no_comments(self):
        code = 'x = 1\ny = 2\nprint(x + y)\n'
        expected = code
        result = remove_single_line_comments_preserving_triple_quotes(code)
        self.assertEqual(result, expected)

    def test_multiple_comments(self):
        code = '''x = 1  # first comment
# full line comment
y = 2  # second comment
'''
        expected = '''x = 1

y = 2
'''
        result = remove_single_line_comments_preserving_triple_quotes(code)
        self.assertEqual(result, expected)

    def test_triple_quotes_with_comment_inside(self):
        code = '''"""
# Full-line comment that is actually inside a string
x = 1  # should not remove this
"""  # comment after triple quote
y = 2  # real comment
'''
        expected = '''"""
# Full-line comment that is actually inside a string
x = 1  # should not remove this
"""
y = 2
'''
        result = remove_single_line_comments_preserving_triple_quotes(code)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
