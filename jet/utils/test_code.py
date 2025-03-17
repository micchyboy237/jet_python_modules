import unittest
from textwrap import dedent
from jet.utils.code import shorten_functions


class TestShortenFunctions(unittest.TestCase):
    def test_single_line_function(self):
        sample = "def hello():\n    pass"
        expected = "def hello():"
        result = shorten_functions(sample)
        self.assertEqual(result, expected)

    def test_multiline_function(self):
        sample = dedent("""\
            def add(a,
                    b):
                return a + b
        """)
        expected = dedent("""\
            def add(a,
                    b):""")
        result = shorten_functions(sample)
        self.assertEqual(result, expected)

    def test_class_with_methods(self):
        sample = dedent("""\
            class Example:
                def method(self, x,
                           y):
                    print(x, y)

                async def async_method(self, a: int,
                                       b: str):
                    pass
        """)
        expected = dedent("""\
            class Example:
                def method(self, x,
                           y):
                async def async_method(self, a: int,
                                       b: str):""")
        result = shorten_functions(sample)
        self.assertEqual(result, expected)

    def test_nested_class(self):
        sample = dedent("""\
            class Outer:
                class Inner:
                    def inner_method(self):
                        pass
        """)
        expected = dedent("""\
            class Outer:
                class Inner:
                    def inner_method(self):""")
        result = shorten_functions(sample)
        self.assertEqual(result, expected)

    def test_async_function(self):
        sample = dedent("""\
            async def fetch_data(url):
                return await fetch(url)
        """)
        expected = "async def fetch_data(url):"
        result = shorten_functions(sample)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
