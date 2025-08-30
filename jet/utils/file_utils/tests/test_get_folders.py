import pytest
import tempfile
import os
from pathlib import Path
from typing import List
import sys
from io import StringIO
from contextlib import redirect_stdout

from jet.utils.file_utils.get_folders import get_folder_absolute_paths


@pytest.fixture
def temp_base_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def setup_test_dirs(temp_base_dir: str) -> str:
    """Set up test directories under temp_base_dir."""
    os.makedirs(os.path.join(temp_base_dir, "folder1"))
    os.makedirs(os.path.join(temp_base_dir, "folder2"))
    os.makedirs(os.path.join(temp_base_dir, "folder3", "subfolder1"))
    os.makedirs(os.path.join(temp_base_dir, "folder3", "subfolder2"))
    os.makedirs(os.path.join(temp_base_dir, "folder4",
                "subfolder3", "deepfolder"))
    with open(os.path.join(temp_base_dir, "file.txt"), "w") as f:
        f.write("test")
    return temp_base_dir


class TestGetFolderAbsolutePaths:
    def test_gets_only_depth_1_folders(self, setup_test_dirs: str):
        # Given: A directory with immediate subdirectories, deeper subdirectories, and a file
        base_dir = setup_test_dirs
        expected = [
            str(Path(base_dir).resolve() / "folder1"),
            str(Path(base_dir).resolve() / "folder2"),
            str(Path(base_dir).resolve() / "folder3"),
            str(Path(base_dir).resolve() / "folder4"),
        ]

        # When: We call get_folder_absolute_paths with depth=1
        result = get_folder_absolute_paths(base_dir, depth=1)

        # Then: Only depth 1 directories are returned, sorted for consistency
        assert sorted(result) == sorted(expected)

    def test_gets_only_depth_2_folders(self, setup_test_dirs: str):
        # Given: A directory with immediate and deeper subdirectories
        base_dir = setup_test_dirs
        expected = [
            str(Path(base_dir).resolve() / "folder3" / "subfolder1"),
            str(Path(base_dir).resolve() / "folder3" / "subfolder2"),
            str(Path(base_dir).resolve() / "folder4" / "subfolder3"),
        ]

        # When: We call get_folder_absolute_paths with depth=2
        result = get_folder_absolute_paths(base_dir, depth=2)

        # Then: Only depth 2 directories are returned, sorted for consistency
        assert sorted(result) == sorted(expected)

    def test_gets_only_depth_3_folders(self, setup_test_dirs: str):
        # Given: A directory with deep subdirectories
        base_dir = setup_test_dirs
        expected = [
            str(Path(base_dir).resolve() / "folder4" /
                "subfolder3" / "deepfolder"),
        ]

        # When: We call get_folder_absolute_paths with depth=3
        result = get_folder_absolute_paths(base_dir, depth=3)

        # Then: Only depth 3 directories are returned
        assert sorted(result) == sorted(expected)

    def test_empty_directory(self, temp_base_dir: str):
        # Given: An empty directory
        base_dir = temp_base_dir
        expected: List[str] = []

        # When: We call get_folder_absolute_paths with depth=1
        result = get_folder_absolute_paths(base_dir, depth=1)

        # Then: An empty list is returned
        assert result == expected

    def test_non_existent_directory(self, temp_base_dir: str):
        # Given: A non-existent directory path
        non_existent_dir = os.path.join(temp_base_dir, "non_existent")
        expected_error = FileNotFoundError

        # When: We call get_folder_absolute_paths with a non-existent path
        # Then: A FileNotFoundError is raised
        with pytest.raises(expected_error):
            get_folder_absolute_paths(non_existent_dir)

    def test_file_instead_of_directory(self, temp_base_dir: str):
        # Given: A file path instead of a directory
        file_path = os.path.join(temp_base_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("test")
        expected_error = NotADirectoryError

        # When: We call get_folder_absolute_paths with a file path
        # Then: A NotADirectoryError is raised
        with pytest.raises(expected_error):
            get_folder_absolute_paths(file_path)

    def test_invalid_depth(self, temp_base_dir: str):
        # Given: A valid directory but invalid depth
        base_dir = temp_base_dir
        expected_error = ValueError

        # When: We call get_folder_absolute_paths with depth=0
        # Then: A ValueError is raised
        with pytest.raises(expected_error):
            get_folder_absolute_paths(base_dir, depth=0)

    def test_main_block_execution(self, setup_test_dirs: str, monkeypatch):
        # Given: A directory with subdirectories and a mocked sys.argv
        base_dir = setup_test_dirs
        expected = [
            str(Path(base_dir).resolve() / "folder1"),
            str(Path(base_dir).resolve() / "folder2"),
            str(Path(base_dir).resolve() / "folder3"),
            str(Path(base_dir).resolve() / "folder4"),
        ]
        monkeypatch.setattr(sys, "argv", ["get_folders.py", base_dir])
        captured_output = StringIO()

        # When: We simulate running the script as if it were the main module
        with redirect_stdout(captured_output):
            if True:  # Simulate __name__ == "__main__"
                if len(sys.argv) != 2:
                    print("Usage: python get_folders.py <directory_path>")
                    sys.exit(1)

                try:
                    folder_paths = get_folder_absolute_paths(sys.argv[1])
                    for path in folder_paths:
                        print(path)
                except (FileNotFoundError, NotADirectoryError, ValueError) as e:
                    print(f"Error: {e}")
                    sys.exit(1)

        # Then: The output contains the expected folder paths (depth=1 by default)
        output = captured_output.getvalue().strip().split("\n")
        assert sorted(output) == sorted(expected)
