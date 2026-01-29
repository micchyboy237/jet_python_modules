import unittest
from jet.transformers.formatters import format_html


class TestFormatHtml(unittest.TestCase):
    def test_basic_html(self):
        """Test basic HTML with nested elements and default indent."""
        input_html = '<!DOCTYPE html><html><head><title>Test</title></head><body><p>Hello</p></body></html>'
        expected = (
            '<html>\n'
            '  <head>\n'
            '    <title>\n'
            '      Test\n'
            '    </title>\n'
            '  </head>\n'
            '  <body>\n'
            '    <p>\n'
            '      Hello\n'
            '    </p>\n'
            '  </body>\n'
            '</html>'
        )
        result = format_html(input_html, indent=2)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
