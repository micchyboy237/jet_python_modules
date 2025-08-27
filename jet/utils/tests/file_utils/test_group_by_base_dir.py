import pytest
from jet.utils.file import group_by_base_dir
from pathlib import Path


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory structure for testing."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "docs").mkdir()
    (base / "docs" / "text").mkdir()
    (base / "images").mkdir()

    # Create some test files
    (base / "file1.txt").touch()
    (base / "docs" / "doc1.txt").touch()
    (base / "docs" / "text" / "text1.txt").touch()
    (base / "images" / "img1.png").touch()

    return base


class TestGroupByBaseDir:
    def test_groups_files_by_base_directory(self, temp_dir):
        # Given: A list of file paths and a base directory
        base = str(temp_dir)
        paths = [
            str(temp_dir / "file1.txt"),
            str(temp_dir / "docs" / "doc1.txt"),
            str(temp_dir / "docs" / "text" / "text1.txt"),
            str(temp_dir / "images" / "img1.png")
        ]
        expected = {
            "": [str(temp_dir / "file1.txt")],
            "docs": [str(temp_dir / "docs" / "doc1.txt")],
            "docs/text": [str(temp_dir / "docs" / "text" / "text1.txt")],
            "images": [str(temp_dir / "images" / "img1.png")]
        }

        # When: We group the paths by base directory
        result = group_by_base_dir(paths, base)

        # Then: The paths are correctly grouped by their relative base directories
        assert result == expected

    def test_handles_empty_path_list(self, temp_dir):
        # Given: An empty list of paths and a base directory
        base = str(temp_dir)
        paths = []
        expected = {}

        # When: We group the empty path list
        result = group_by_base_dir(paths, base)

        # Then: An empty dictionary is returned
        assert result == expected

    def test_ignores_paths_outside_base_dir(self, temp_dir):
        # Given: A mix of paths inside and outside the base directory
        base = str(temp_dir)
        outside_path = str(temp_dir.parent / "outside.txt")
        paths = [
            str(temp_dir / "file1.txt"),
            outside_path,
            str(temp_dir / "docs" / "doc1.txt")
        ]
        expected = {
            "": [str(temp_dir / "file1.txt")],
            "docs": [str(temp_dir / "docs" / "doc1.txt")]
        }

        # When: We group the paths
        result = group_by_base_dir(paths, base)

        # Then: Only paths under base_dir are grouped, others are ignored
        assert result == expected

    def test_handles_relative_paths(self, temp_dir):
        # Given: Relative paths and a base directory
        base = str(temp_dir)
        paths = [
            "file1.txt",
            "docs/doc1.txt",
            "docs/text/text1.txt"
        ]
        expected = {
            "": [str(temp_dir / "file1.txt")],
            "docs": [str(temp_dir / "docs" / "doc1.txt")],
            "docs/text": [str(temp_dir / "docs" / "text" / "text1.txt")]
        }

        # When: We group the paths relative to base_dir
        result = group_by_base_dir(paths, base)

        # Then: The relative paths are correctly resolved and grouped
        assert result == expected
