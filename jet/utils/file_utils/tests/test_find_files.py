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
        # Given: A directory with mixed file types and an include pattern
        base_dir = self.temp_dir
        include = ["*.txt"]
        exclude = []
        include_content_patterns = []
        exclude_content_patterns = []
        expected = [os.path.join(self.temp_dir, "test1.txt")]
        # When: Finding files with the include pattern
        result = find_files(base_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)
        # Then: Only files matching the include pattern are returned
        assert sorted(result) == sorted(expected)

    def test_find_files_with_exclude_pattern(self):
        """Test excluding files with a specific pattern."""
        # Given: A directory with mixed file types and an exclude pattern
        base_dir = self.temp_dir
        include = ["*"]
        exclude = ["*.py"]
        include_content_patterns = []
        exclude_content_patterns = []
        expected = [
            os.path.join(self.temp_dir, "test1.txt"),
            os.path.join(self.temp_dir, "package.json")
        ]
        # When: Finding files with the exclude pattern
        result = find_files(base_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)
        # Then: Files matching the exclude pattern are not returned
        assert sorted(result) == sorted(expected)

    def test_find_files_with_content_patterns(self):
        """Test finding files based on content patterns."""
        # Given: A directory with files and a content pattern
        base_dir = self.temp_dir
        include = ["*"]
        exclude = []
        include_content_patterns = ["Hello*"]
        exclude_content_patterns = []
        expected = [os.path.join(self.temp_dir, "test1.txt")]
        # When: Finding files with the content pattern
        result = find_files(base_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)
        # Then: Only files matching the content pattern are returned
        assert sorted(result) == sorted(expected)

    def test_find_files_case_insensitive_content(self):
        """Test content matching with case-insensitive patterns."""
        # Given: A directory with files and a case-insensitive content pattern
        base_dir = self.temp_dir
        include = ["*"]
        exclude = []
        include_content_patterns = ["HELLO*"]
        exclude_content_patterns = []
        expected = [os.path.join(self.temp_dir, "test1.txt")]
        # When: Finding files with the case-insensitive content pattern
        result = find_files(base_dir, include, exclude, include_content_patterns,
                            exclude_content_patterns, case_sensitive=False)
        # Then: Only files matching the content pattern are returned
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
        expected = [os.path.join(self.temp_dir, "test1.txt")]
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

    def test_find_files_with_only_extensions(self):
        """Test finding files using only the extensions parameter."""
        # Given: A directory with mixed file types and only an extension filter
        base_dir = self.temp_dir
        include = []
        exclude = []
        include_content_patterns = []
        exclude_content_patterns = []
        extensions = [".txt", "txt", "*.txt"]  # Test all supported formats
        expected = [os.path.join(self.temp_dir, "test1.txt")]
        # When: Finding files with only the extension filter
        result = find_files(
            base_dir,
            include,
            exclude,
            include_content_patterns,
            exclude_content_patterns,
            case_sensitive=False,
            extensions=extensions
        )
        # Then: Only files with .txt extension are returned
        assert sorted(result) == sorted(expected)


class TestDirectoryMatching:
    def setup_method(self):
        """Create a temporary directory structure for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.sub_dir = os.path.join(self.temp_dir, "subdir")
        self.zh_dir = os.path.join(self.temp_dir, "zh")
        os.makedirs(self.sub_dir)
        os.makedirs(self.zh_dir)
        self.file1 = os.path.join(self.sub_dir, "test_sub.txt")
        self.file2 = os.path.join(self.zh_dir, "test_zh.txt")
        self.file3 = os.path.join(self.temp_dir, "root.txt")
        with open(self.file1, 'w') as f:
            f.write("Subdir content")
        with open(self.file2, 'w') as f:
            f.write("ZH content")
        with open(self.file3, 'w') as f:
            f.write("Root content")

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_absolute_directory_path(self):
        """Test finding files in a specific directory using an absolute path."""
        # Given: A directory with a file in a subdirectory
        base_dir = self.temp_dir
        abs_path = os.path.join(self.temp_dir, "subdir")
        include = [abs_path]
        exclude = []
        include_content_patterns = []
        exclude_content_patterns = []
        expected = [os.path.join(self.temp_dir, "subdir", "test_sub.txt")]

        # When: Searching with an absolute directory path
        result = find_files(base_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)

        # Then: Only files in the specified directory should be found
        assert sorted(result) == sorted(expected)

    def test_wildcard_directory_pattern(self):
        """Test finding files in directories matching a wildcard pattern."""
        # Given: A directory with a zh subdirectory containing a file
        include = ["*/zh/*"]
        exclude = []
        include_content_patterns = []
        exclude_content_patterns = []
        expected = [os.path.join(self.temp_dir, "zh", "test_zh.txt")]

        # When: Searching with a wildcard directory pattern
        result = find_files(self.temp_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)

        # Then: Only files in the matching directory should be found
        assert sorted(result) == sorted(expected)

    def test_slash_ending_directory_pattern(self):
        """Test finding files in directories with slash-ending patterns."""
        # Given: A directory with a zh subdirectory containing a file
        include = ["zh/", "/zh/"]
        exclude = []
        include_content_patterns = []
        exclude_content_patterns = []
        expected = [os.path.join(self.temp_dir, "zh", "test_zh.txt")]

        # When: Searching with slash-ending directory patterns
        result = find_files(self.temp_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)

        # Then: Files in the matching directory should be found for both patterns
        assert sorted(result) == sorted(expected)

    def test_exclude_directory_pattern(self):
        """Test excluding files in a specific directory."""
        # Given: A directory with multiple subdirectories and files
        include = ["*"]
        exclude = ["subdir/*"]
        include_content_patterns = []
        exclude_content_patterns = []
        expected = [
            os.path.join(self.temp_dir, "zh", "test_zh.txt"),
            os.path.join(self.temp_dir, "root.txt")
        ]

        # When: Searching with a directory exclude pattern
        result = find_files(self.temp_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)

        # Then: Files in the excluded directory should not be included
        assert sorted(result) == sorted(expected)

    def test_nested_directory_pattern(self):
        """Test finding files in a nested directory structure."""
        # Given: A nested directory structure with files
        nested_dir = os.path.join(self.temp_dir, "zh", "nested")
        os.makedirs(nested_dir)
        nested_file = os.path.join(nested_dir, "nested.txt")
        with open(nested_file, 'w') as f:
            f.write("Nested content")
        include = ["zh/**/*"]
        exclude = []
        include_content_patterns = []
        exclude_content_patterns = []
        expected = [
            os.path.join(self.temp_dir, "zh", "test_zh.txt"),
            os.path.join(self.temp_dir, "zh", "nested", "nested.txt")
        ]

        # When: Searching with a recursive directory pattern
        result = find_files(self.temp_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)

        # Then: Files in the nested directories should be found
        assert sorted(result) == sorted(expected)

    def test_multiple_directory_patterns(self):
        """Test finding files with multiple directory include patterns."""
        # Given: Multiple directories with files
        include = ["zh/*", "subdir/*"]
        exclude = []
        include_content_patterns = []
        exclude_content_patterns = []
        expected = [
            os.path.join(self.temp_dir, "zh", "test_zh.txt"),
            os.path.join(self.temp_dir, "subdir", "test_sub.txt")
        ]

        # When: Searching with multiple directory patterns
        result = find_files(self.temp_dir, include, exclude,
                            include_content_patterns, exclude_content_patterns)

        # Then: Files in all specified directories should be found
        assert sorted(result) == sorted(expected)

    def test_include_directory_with_extensions(self):
        """Test finding files in directories with 'en' parent or ancestor with specific extensions."""
        # Given: A directory structure with 'en' and other directories containing .py and .ipynb files
        en_dir = os.path.join(self.temp_dir, "en")
        en_nested_dir = os.path.join(en_dir, "nested")
        os.makedirs(en_nested_dir)
        other_dir = os.path.join(self.temp_dir, "other")
        os.makedirs(other_dir)

        en_file = os.path.join(en_dir, "config.py")
        en_nested_file = os.path.join(en_nested_dir, "config.ipynb")
        other_file = os.path.join(other_dir, "config.py")

        with open(en_file, 'w') as f:
            f.write("print('en config')")
        with open(en_nested_file, 'w') as f:
            f.write("{'notebook': 'en nested'}")
        with open(other_file, 'w') as f:
            f.write("print('other config')")

        # When: Searching for files with 'en' parent/ancestor and specific extensions
        include = ["en/"]
        exclude = []
        include_content_patterns = []
        exclude_content_patterns = []
        extensions = [".py", "py", "*.py", ".ipynb"]

        result = find_files(
            self.temp_dir,
            include,
            exclude,
            include_content_patterns,
            exclude_content_patterns,
            case_sensitive=False,
            extensions=extensions
        )

        # Then: Only files under 'en' directory with .py or .ipynb extensions are returned
        expected = [
            os.path.join(self.temp_dir, "en", "config.py"),
            os.path.join(self.temp_dir, "en", "nested", "config.ipynb")
        ]
        assert sorted(result) == sorted(expected)

    def test_exclude_directory_with_extensions(self):
        """Test excluding files in directories with 'zh' parent or ancestor with specific extensions."""
        # Given: A directory structure with 'zh' and other directories containing .py and .ipynb files
        zh_dir = os.path.join(self.temp_dir, "zh")
        zh_nested_dir = os.path.join(zh_dir, "nested")
        en_dir = os.path.join(self.temp_dir, "en")
        os.makedirs(zh_nested_dir)
        os.makedirs(en_dir)

        zh_file = os.path.join(zh_dir, "config.py")
        zh_nested_file = os.path.join(zh_nested_dir, "config.ipynb")
        en_file = os.path.join(en_dir, "config.py")

        with open(zh_file, 'w') as f:
            f.write("print('zh config')")
        with open(zh_nested_file, 'w') as f:
            f.write("{'notebook': 'zh nested'}")
        with open(en_file, 'w') as f:
            f.write("print('en config')")

        # When: Searching for files excluding 'zh' parent/ancestor with specific extensions
        include = []
        exclude = ["zh/"]
        include_content_patterns = []
        exclude_content_patterns = []
        extensions = [".py", "py", "*.py", ".ipynb"]

        result = find_files(
            self.temp_dir,
            include,
            exclude,
            include_content_patterns,
            exclude_content_patterns,
            case_sensitive=False,
            extensions=extensions
        )

        # Then: Only files not under 'zh' directory with .py or .ipynb extensions are returned
        expected = [
            os.path.join(self.temp_dir, "en", "config.py")
        ]
        assert sorted(result) == sorted(expected)

    def test_exclude_directory_patterns_recursive_and_non_recursive(self):
        """Test exclude patterns for recursive and non-recursive directory exclusion."""
        # Given: A directory structure with zh, zh/nested, and en directories
        zh_nested_dir = os.path.join(self.temp_dir, "zh", "nested")
        en_dir = os.path.join(self.temp_dir, "en")
        os.makedirs(zh_nested_dir)
        os.makedirs(en_dir)
        zh_file = os.path.join(self.zh_dir, "config.py")
        zh_nested_file = os.path.join(zh_nested_dir, "nested_config.py")
        en_file = os.path.join(en_dir, "config.py")
        with open(zh_file, 'w') as f:
            f.write("print('zh config')")
        with open(zh_nested_file, 'w') as f:
            f.write("print('zh nested config')")
        with open(en_file, 'w') as f:
            f.write("print('en config')")

        # Given: Expected files when excluding zh recursively
        expected_recursive = [
            os.path.join(self.temp_dir, "en", "config.py"),
            os.path.join(self.temp_dir, "root.txt"),
            os.path.join(self.temp_dir, "subdir", "test_sub.txt")
        ]
        # Given: Expected files when excluding zh non-recursively
        expected_non_recursive = [
            os.path.join(self.temp_dir, "en", "config.py"),
            os.path.join(self.temp_dir, "root.txt"),
            os.path.join(self.temp_dir, "subdir", "test_sub.txt"),
            os.path.join(self.temp_dir, "zh", "nested", "nested_config.py")
        ]

        # When: Testing recursive exclude patterns
        for exclude_pattern in [
            "zh/",
            "/zh/",
            os.path.join(self.temp_dir, "zh")
        ]:
            result = find_files(
                base_dir=self.temp_dir,
                include=["*"],
                exclude=[exclude_pattern],
                include_content_patterns=[],
                exclude_content_patterns=[],
                case_sensitive=False,
                extensions=[".py"]
            )
            # Then: Only files outside zh directory are included
            assert sorted(result) == sorted(
                expected_recursive), f"Failed for recursive exclude pattern: {exclude_pattern}"

        # When: Testing non-recursive exclude pattern
        result = find_files(
            base_dir=self.temp_dir,
            include=["*"],
            exclude=["zh/*"],
            include_content_patterns=[],
            exclude_content_patterns=[],
            case_sensitive=False,
            extensions=[".py"]
        )
        # Then: Files in zh/nested are included, but files directly in zh are excluded
        assert sorted(result) == sorted(
            expected_non_recursive), "Failed for non-recursive exclude pattern: zh/*"
