import os
import unittest
from tempfile import TemporaryDirectory
from jet.file import traverse_directory


class TestTraverseDirectory(unittest.TestCase):
    def setUp(self):
        # Set up a temporary directory structure for testing
        self.temp_dir = TemporaryDirectory()
        base = self.temp_dir.name

        # Create directories
        os.makedirs(os.path.join(base, "dir1/subdir1"))
        os.makedirs(os.path.join(base, "dir1/subdir2"))
        os.makedirs(os.path.join(base, "dir2/subdir3"))
        os.makedirs(os.path.join(base, "dir3"))

        # Create some files
        open(os.path.join(base, "dir1/file1.txt"), "w").close()
        open(os.path.join(base, "dir1/subdir1/file2.txt"), "w").close()
        open(os.path.join(base, "dir2/file3.txt"), "w").close()

        self.base_dir = base

    def tearDown(self):
        # Clean up the temporary directory after tests
        self.temp_dir.cleanup()

    def test_traverse_forward_include_pattern(self):
        includes = ["subdir1", "subdir2"]
        results = list(traverse_directory(self.base_dir, includes=includes))
        expected = [
            os.path.join(self.base_dir, "dir1/subdir1"),
            os.path.join(self.base_dir, "dir1/subdir2"),
        ]
        self.assertCountEqual(results, expected)

    def test_traverse_forward_exclude_pattern(self):
        includes = ["dir1"]
        excludes = ["subdir2"]
        results = list(traverse_directory(
            self.base_dir, includes=includes, excludes=excludes))
        expected = [os.path.join(self.base_dir, "dir1/subdir1")]
        self.assertCountEqual(results, expected)

    def test_limit_results(self):
        includes = ["dir"]
        results = list(traverse_directory(
            self.base_dir, includes=includes, limit=2))
        self.assertEqual(len(results), 2)

    def test_backward_traversal(self):
        current_dir = os.path.join(self.base_dir, "dir1/subdir1")
        includes = ["dir1"]
        results = list(traverse_directory(
            current_dir, includes=includes, direction="backward"))
        expected = [os.path.join(self.base_dir, "dir1")]
        self.assertCountEqual(results, expected)

    def test_backward_with_max_depth(self):
        current_dir = os.path.join(self.base_dir, "dir1/subdir1")
        includes = ["dir"]
        results = list(
            traverse_directory(
                current_dir, includes=includes, direction="backward", max_backward_depth=1
            )
        )
        expected = [os.path.join(self.base_dir, "dir1")]
        self.assertCountEqual(results, expected)

    def test_forward_and_backward_traversal(self):
        includes = ["dir1", "subdir3"]
        results = list(traverse_directory(
            self.base_dir, includes=includes, direction="both"))
        expected = [
            os.path.join(self.base_dir, "dir1"),
            os.path.join(self.base_dir, "dir2/subdir3"),
        ]
        self.assertCountEqual(results, expected)


if __name__ == "__main__":
    unittest.main()
