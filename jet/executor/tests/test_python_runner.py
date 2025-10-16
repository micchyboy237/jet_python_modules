import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock
from jet.executor.python_runner import run_python_files_in_directory, sort_key


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture to create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def setup_test_files(temp_dir: Path) -> Path:
    """Fixture to create test Python files and directory structure."""
    # Create main test directory
    test_dir = temp_dir / "test_python_files"
    test_dir.mkdir()

    # Create subdirectories
    sub_dir1 = test_dir / "sub1"
    sub_dir2 = test_dir / "sub2"
    sub_dir1.mkdir()
    sub_dir2.mkdir()

    # Create test Python files
    (test_dir / "1_success.py").write_text("print('Success')")
    (test_dir / "2_fail.py").write_text("raise Exception('Failed')")
    (test_dir / "3_another_success.py").write_text("print('Another Success')")
    (sub_dir1 / "4_sub_success.py").write_text("print('Sub Success')")
    (sub_dir2 / "5_sub_fail.py").write_text("raise Exception('Sub Failed')")

    return test_dir


@pytest.fixture
def output_dir(temp_dir: Path) -> Path:
    """Fixture to create output directory."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    return output_dir


class TestSortKey:
    """Test suite for sort_key function."""

    def test_given_numeric_prefix_files_when_sorting_then_orders_correctly(self):
        """Test sort_key orders files with numeric prefixes correctly."""
        given_files = ["10_file.py", "2_file.py", "1_file.py", "file.py"]
        expected = ["1_file.py", "2_file.py", "10_file.py", "file.py"]
        result = sorted(given_files, key=sort_key)
        assert result == expected, f"Expected {expected}, but got {result}"


class TestRunPythonFilesInDirectory:
    """Test suite for run_python_files_in_directory function."""
    def test_given_non_recursive_search_when_running_files_then_processes_only_top_level(
        self, setup_test_files: Path, output_dir: Path
    ):
        """Test non-recursive search only processes top-level files."""
        target_dir = setup_test_files
        expected_files = [
            str(target_dir / "1_success.py"),
            str(target_dir / "2_fail.py"),
            str(target_dir / "3_another_success.py")
        ]
        with patch("subprocess.Popen") as mock_popen, patch("jet.utils.file.search_files") as mock_search:
            mock_search.return_value = expected_files
            mock_process = Mock()
            mock_process.stdout = ["Success\n"]
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process
            run_python_files_in_directory(
                target_dir=target_dir,
                output_dir=output_dir,
                recursive=False,
                rerun_mode="all",
                include_contents=[],
                exclude_contents=[],
            )
        status_file = output_dir / "files_status.json"
        assert status_file.exists(), "Status file was not created"
        with status_file.open("r") as f:
            result = json.load(f)
        result_files = [str(target_dir / entry["file"]) for entry in result]
        assert sorted(result_files) == sorted(expected_files), \
            f"Expected {expected_files}, but got {result_files}"
    def test_given_recursive_search_when_running_files_then_processes_all_files(
        self, setup_test_files: Path, output_dir: Path
    ):
        """Test recursive search processes all files including subdirectories."""
        target_dir = setup_test_files
        expected_files = [
            str(target_dir / "1_success.py"),
            str(target_dir / "2_fail.py"),
            str(target_dir / "3_another_success.py"),
            str(target_dir / "sub1" / "4_sub_success.py"),
            str(target_dir / "sub2" / "5_sub_fail.py")
        ]
        with patch("subprocess.Popen") as mock_popen, patch("jet.utils.file.search_files") as mock_search:
            mock_search.return_value = expected_files
            mock_process = Mock()
            mock_process.stdout = ["Success\n"]
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process
            run_python_files_in_directory(
                target_dir=target_dir,
                output_dir=output_dir,
                recursive=True,
                rerun_mode="all",
                include_contents=[],
                exclude_contents=[],
            )
        status_file = output_dir / "files_status.json"
        assert status_file.exists(), "Status file was not created"
        with status_file.open("r") as f:
            result = json.load(f)
        result_files = [str(target_dir / entry["file"]) for entry in result]
        assert sorted(result_files) == sorted(expected_files), \
            f"Expected {expected_files}, but got {result_files}"
    def test_given_include_patterns_when_running_files_then_only_matching_files_run(
        self, setup_test_files: Path, output_dir: Path
    ):
        """Test include patterns filter files correctly."""
        target_dir = setup_test_files
        includes = ["*success*.py"]
        expected_files = [
            str(target_dir / "1_success.py"),
            str(target_dir / "3_another_success.py"),
            str(target_dir / "sub1" / "4_sub_success.py")
        ]
        with patch("subprocess.Popen") as mock_popen, patch("jet.utils.file.search_files") as mock_search:
            mock_search.return_value = expected_files
            mock_process = Mock()
            mock_process.stdout = ["Success\n"]
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process
            run_python_files_in_directory(
                target_dir=target_dir,
                output_dir=output_dir,
                includes=includes,
                recursive=True,
                rerun_mode="all",
                include_contents=[],
                exclude_contents=[],
            )
        status_file = output_dir / "files_status.json"
        assert status_file.exists(), "Status file was not created"
        with status_file.open("r") as f:
            result = json.load(f)
        result_files = [str(target_dir / entry["file"]) for entry in result]
        assert sorted(result_files) == sorted(expected_files), \
            f"Expected {expected_files}, but got {result_files}"
    def test_given_failed_rerun_mode_when_running_files_then_only_failed_files_run(
        self, setup_test_files: Path, output_dir: Path
    ):
        """Test failed rerun mode only processes previously failed files."""
        target_dir = setup_test_files
        status_file = output_dir / "files_status.json"
        initial_status = [
            {"file": "1_success.py", "status": "Success",
                "return_code": "0", "timestamp": "2025-09-08T00:00:00"},
            {"file": "2_fail.py",
                "status": "Failed (code 1)", "return_code": "1", "timestamp": "2025-09-08T00:00:00"},
            {"file": "sub1/4_sub_success.py", "status": "Success",
                "return_code": "0", "timestamp": "2025-09-08T00:00:00"}
        ]
        status_file.write_text(json.dumps(initial_status))
        expected_files = [str(target_dir / "2_fail.py")]
        with patch("subprocess.Popen") as mock_popen, patch("jet.utils.file.search_files") as mock_search:
            mock_search.return_value = expected_files
            mock_process = Mock()
            mock_process.stdout = ["Success\n"]
            mock_process.wait.return_value = 0
            mock_popen.side_effect = lambda cmd, *args, **kwargs: mock_process \
                if cmd[-1].endswith("2_fail.py") else Mock(stdout=[], wait=lambda: 0)
            run_python_files_in_directory(
                target_dir=target_dir,
                output_dir=output_dir,
                recursive=True,
                rerun_mode="failed",
                include_contents=[],
                exclude_contents=[],
            )
        with status_file.open("r") as f:
            result = json.load(f)
        result_files = [
            str(target_dir / entry["file"]) for entry in result
            if entry["timestamp"] > "2025-09-08T00:00:00"
        ]
        assert sorted(result_files) == sorted(expected_files), \
            f"Expected {expected_files}, but got {result_files}"
    def test_given_content_filters_when_running_files_then_only_matching_content_files_run(
        self, setup_test_files: Path, output_dir: Path
    ):
        """Test content include/exclude patterns filter files correctly."""
        target_dir = setup_test_files
        include_contents = ["*Success*"]
        exclude_contents = ["*Failed*"]
        expected_files = [
            str(target_dir / "1_success.py"),
            str(target_dir / "3_another_success.py"),
            str(target_dir / "sub1" / "4_sub_success.py")
        ]
        with patch("subprocess.Popen") as mock_popen, patch("jet.utils.file.search_files") as mock_search:
            mock_search.return_value = expected_files
            mock_process = Mock()
            mock_process.stdout = ["Success\n"]
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process
            run_python_files_in_directory(
                target_dir=target_dir,
                output_dir=output_dir,
                recursive=True,
                rerun_mode="all",
                include_contents=include_contents,
                exclude_contents=exclude_contents,
            )
        status_file = output_dir / "files_status.json"
        assert status_file.exists(), "Status file was not created"
        with status_file.open("r") as f:
            result = json.load(f)
        result_files = [str(target_dir / entry["file"]) for entry in result]
        assert sorted(result_files) == sorted(expected_files), \
            f"Expected {expected_files}, but got {result_files}"
