import os
import pytest
import tempfile
from jet.utils.file_utils.search import find_files


class TestFindFiles:
    def setup_method(self):
        """Create a temporary directory structure for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.sub_dir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(self.sub_dir)

        # Create test files
        self.file1 = os.path.join(self.temp_dir, "test1.txt")
        self.file2 = os.path.join(self.sub_dir, "test2.py")
        self.file3 = os.path.join(self.temp_dir, "package.json")

        with open(self.file1, 'w') as f:
            f.write("Hello, world!")
        with open(self.file2, 'w') as f:
            f.write("print('test')")
        with open(self.file3, 'w') as f:
            f.write('{"name": "test"}')

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_find_files_with_include_pattern(self):
        """Test finding files with a specific include pattern."""
        # Given: A directory with files and an include pattern for *.txt
        base_dir = self.temp_dir
        include = ["*.txt"]
        exclude = []
        include_content_patterns = []
        exclude_content_patterns = []
        expected = [os.path.relpath(self.file1, self.temp_dir)]

        # When: We call find_files with the include pattern
        result = find_files(base_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)

        # Then: Only the .txt file should be returned
        assert sorted(result) == sorted(expected)

    def test_find_files_with_exclude_pattern(self):
        """Test excluding files with a specific pattern."""
        # Given: A directory with files and an exclude pattern for *.py
        base_dir = self.temp_dir
        include = ["*"]
        exclude = ["*.py"]
        include_content_patterns = []
        exclude_content_patterns = []
        expected = [
            os.path.relpath(self.file1, self.temp_dir),
            os.path.relpath(self.file3, self.temp_dir)
        ]

        # When: We call find_files with the exclude pattern
        result = find_files(base_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)

        # Then: The .py file should be excluded
        assert sorted(result) == sorted(expected)

    def test_find_files_with_content_patterns(self):
        """Test finding files based on content patterns."""
        # Given: A directory with files and a content include pattern
        base_dir = self.temp_dir
        include = ["*"]
        exclude = []
        include_content_patterns = ["Hello*"]
        exclude_content_patterns = []
        expected = [os.path.relpath(self.file1, self.temp_dir)]

        # When: We call find_files with content patterns
        result = find_files(base_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)

        # Then: Only files matching the content pattern should be returned
        assert sorted(result) == sorted(expected)

    def test_find_files_case_insensitive_content(self):
        """Test content matching with case-insensitive patterns."""
        # Given: A directory with files and a case-insensitive content pattern
        base_dir = self.temp_dir
        include = ["*"]
        exclude = []
        include_content_patterns = ["HELLO*"]
        exclude_content_patterns = []
        expected = [os.path.relpath(self.file1, self.temp_dir)]

        # When: We call find_files with case-insensitive flag
        result = find_files(base_dir, include, exclude, include_content_patterns,
                            exclude_content_patterns, case_sensitive=False)

        # Then: Files should match case-insensitive content
        assert sorted(result) == sorted(expected)

    def test_find_files_with_extensions(self):
        """Test finding files with specific extensions."""
        # Given: A directory with mixed file types and an extension filter
        base_dir = self.temp_dir
        include = ["*"]
        exclude = []
        include_content_patterns = []
        exclude_content_patterns = []
        extensions = ["txt", ".txt", "*.txt"]  # Test all supported formats
        expected = [os.path.relpath(self.file1, self.temp_dir)]
        # When: Finding files with the extension filter
        result = find_files(
            base_dir, include, exclude, include_content_patterns, exclude_content_patterns, extensions=extensions
        )
        # Then: Only files with .txt extension are returned
        assert sorted(result) == sorted(expected)

    def test_find_files_with_extensions_no_match(self):
        """Test finding files with extensions that don't match any files."""
        # Given: A directory with files and a non-matching extension filter
        base_dir = self.temp_dir
        include = ["*"]
        exclude = []
        include_content_patterns = []
        exclude_content_patterns = []
        extensions = ["docx"]  # Non-existent extension
        expected = []
        # When: Finding files with the extension filter
        result = find_files(
            base_dir, include, exclude, include_content_patterns, exclude_content_patterns, extensions=extensions
        )
        # Then: No files are returned
        assert sorted(result) == sorted(expected)
