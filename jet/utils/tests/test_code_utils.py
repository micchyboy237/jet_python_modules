import unittest
from textwrap import dedent
from jet.utils.code_utils import shorten_functions


class TestShortenFunctions(unittest.TestCase):
    def test_single_line_function(self):
        sample = "def hello():\n    pass"
        expected = "def hello()"
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
                    b)""")
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
        expected = """
class Example:
    def method(self, x,
                   y)
    async def async_method(self, a: int,
                               b: str)
        """.strip()
        result = shorten_functions(sample)
        self.assertEqual(result, expected)

    def test_nested_class(self):
        sample = dedent("""\
            class Outer:
                class Inner:
                    def inner_method(self):
                        pass
        """)
        expected = """
class Outer:
    class Inner:
        def inner_method(self)
        """.strip()
        result = shorten_functions(sample)
        self.assertEqual(result, expected)

    def test_async_function(self):
        sample = dedent("""\
            async def fetch_data(url: str) -> dict:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        return await response.json()
        """)
        expected = "async def fetch_data(url: str) -> dict"
        result = shorten_functions(sample)
        self.assertEqual(result, expected)

    def test_class_with_variables(self):
        """
        Test that shorten_functions can include class variables when requested.
        Given: A class with variables and methods
        When: shorten_functions is called with include_class_vars=True
        Then: The output includes the class signature, variables, and method signatures
        """
        sample = dedent("""\
            class ResourceContents(BaseModel):
                id: int = 42
                name: str
                def get_id(self) -> int:
                    return self.id
                async def fetch_name(self, url: str) -> str:
                    return await some_call(url)
        """)
        expected = """
class ResourceContents(BaseModel):
    id: int = 42
    name: str
    def get_id(self) -> int
    async def fetch_name(self, url: str) -> str
        """.strip()
        result = shorten_functions(sample)
        self.assertEqual(
            result, expected, f"shorten_functions result did not match expected.\nResult:\n{result!r}")

    def test_class_without_variables(self):
        """
        Test that shorten_functions excludes class variables when requested.
        Given: A class with variables and methods
        When: shorten_functions is called with remove_class_vars=True
        Then: The output includes only the class and method signatures, excluding variables
        """
        sample = dedent("""\
            class ResourceContents(BaseModel):
                id: int = 42
                name: str
                def get_id(self) -> int:
                    return self.id
                async def fetch_name(self, url: str) -> str:
                    return await some_call(url)
        """)
        expected = """
class ResourceContents(BaseModel):
    def get_id(self) -> int
    async def fetch_name(self, url: str) -> str
        """.strip()
        result = shorten_functions(sample, remove_class_vars=True)
        self.assertEqual(
            result, expected, f"shorten_functions result did not match expected.\nResult:\n{result!r}")


if __name__ == "__main__":
    unittest.main()
