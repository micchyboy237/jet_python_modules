import unittest

from jet.transformers.text import to_snake_case


class TestToSnakeCase(unittest.TestCase):
    def test_basic_url(self):
        self.assertEqual(to_snake_case(
            "https://example.com/path/to/resource"), "example_com_path_to_resource")

    def test_url_with_special_characters(self):
        self.assertEqual(to_snake_case(
            "https://example.com/path/to/res!ource"), "example_com_path_to_res_ource")

    def test_url_with_trailing_slash(self):
        self.assertEqual(to_snake_case(
            "https://example.com/path/to/resource/"), "example_com_path_to_resource")

    def test_empty_path(self):
        self.assertEqual(to_snake_case("https://example.com/"), "example_com")

    def test_root_url(self):
        self.assertEqual(to_snake_case("https://example.com"), "example_com")

    def test_multiple_slashes(self):
        self.assertEqual(to_snake_case(
            "https://example.com/path//to///resource"), "example_com_path_to_resource")

    def test_mixed_case_path(self):
        self.assertEqual(to_snake_case(
            "https://example.com/Path/To/Resource"), "example_com_path_to_resource")

    def test_path_with_numbers(self):
        self.assertEqual(to_snake_case(
            "https://example.com/path123/to456/resource789"), "example_com_path123_to456_resource789")

    def test_path_with_non_alphanumeric(self):
        self.assertEqual(to_snake_case(
            "https://example.com/path@#$%^&*()/to/resource!"), "example_com_path_to_resource")

    def test_different_domain(self):
        self.assertEqual(to_snake_case(
            "https://sub.example.co.uk/path/to/resource"), "sub_example_co_uk_path_to_resource")


if __name__ == "__main__":
    unittest.main()
