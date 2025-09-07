import pytest
import os
import json
import shutil
from pathlib import Path
from jet.executor.python_runner import run_python_files_in_directory


@pytest.fixture
def setup_test_dir(tmp_path: Path):
    """Set up a temporary directory with test Python files."""
    test_dir = tmp_path / "test_scripts"
    test_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create test files
    (test_dir / "1_success.py").write_text('print("Success")')
    (test_dir / "2_fail.py").write_text('print("Fail"); exit(1)')
    (test_dir / "3_unrun.py").write_text('print("Unrun")')

    yield test_dir, output_dir

    # Cleanup
    shutil.rmtree(tmp_path, ignore_errors=True)


class TestPythonRunner:
    """Test suite for run_python_files_in_directory function."""

    def test_run_all_files(self, setup_test_dir):
        """Test running all files in the directory."""
        test_dir, output_dir = setup_test_dir

        # Given: A directory with Python files and an output directory
        # When: Running all files
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="all"
        )
        # Then: Verify status file contains all files with correct statuses
        status_file = output_dir / "files_status.json"
        assert status_file.exists()
        with status_file.open('r') as f:
            status_data = json.load(f)
        expected = [
            {"file": "1_success.py", "status": "Success", "return_code": "0"},
            {"file": "2_fail.py",
                "status": "Failed (code 1)", "return_code": "1"},
            {"file": "3_unrun.py", "status": "Success", "return_code": "0"}
        ]
        result = [
            {k: entry[k] for k in ["file", "status", "return_code"]}
            for entry in status_data
        ]
        assert sorted(result, key=lambda x: x["file"]) == sorted(
            expected, key=lambda x: x["file"])

    def test_run_failed_files(self, setup_test_dir):
        """Test running only previously failed files."""
        test_dir, output_dir = setup_test_dir

        # Given: Run all files first to create status file with a failed file
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="all"
        )
        # When: Rerun only failed files
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="failed"
        )
        # Then: Verify only failed file was rerun, others preserved
        status_file = output_dir / "files_status.json"
        assert status_file.exists()
        with status_file.open('r') as f:
            status_data = json.load(f)
        expected = [
            {"file": "1_success.py", "status": "Success", "return_code": "0"},
            {"file": "2_fail.py",
                "status": "Failed (code 1)", "return_code": "1"},
            {"file": "3_unrun.py", "status": "Success", "return_code": "0"}
        ]
        result = [
            {k: entry[k] for k in ["file", "status", "return_code"]}
            for entry in status_data
        ]
        assert sorted(result, key=lambda x: x["file"]) == sorted(
            expected, key=lambda x: x["file"])
        # Verify only 2_fail.py log was updated
        failed_log = output_dir / "failed" / "2_fail.log"
        assert failed_log.exists()

    def test_run_unrun_files(self, setup_test_dir):
        """Test running only unrun files."""
        test_dir, output_dir = setup_test_dir

        # Given: Run some files to create status file
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="all",
            includes=["1_success.py", "2_fail.py"]
        )
        # When: Run only unrun files
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="unrun"
        )
        # Then: Verify only 3_unrun.py was run, others preserved
        status_file = output_dir / "files_status.json"
        assert status_file.exists()
        with status_file.open('r') as f:
            status_data = json.load(f)
        expected = [
            {"file": "1_success.py", "status": "Success", "return_code": "0"},
            {"file": "2_fail.py",
                "status": "Failed (code 1)", "return_code": "1"},
            {"file": "3_unrun.py", "status": "Success", "return_code": "0"}
        ]
        result = [
            {k: entry[k] for k in ["file", "status", "return_code"]}
            for entry in status_data
        ]
        assert sorted(result, key=lambda x: x["file"]) == sorted(
            expected, key=lambda x: x["file"])
        # Verify 3_unrun.py log exists
        success_log = output_dir / "success" / "3_unrun.log"
        assert success_log.exists()

    def test_run_failed_and_unrun_files(self, setup_test_dir):
        """Test running only previously failed and unrun files."""
        test_dir, output_dir = setup_test_dir

        # Given: Run some files to create status file with a failed file
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="all",
            includes=["1_success.py", "2_fail.py"]
        )
        # When: Run only failed and unrun files
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="failed_and_unrun"
        )
        # Then: Verify failed (2_fail.py) and unrun (3_unrun.py) files were run, others preserved
        status_file = output_dir / "files_status.json"
        assert status_file.exists()
        with status_file.open('r') as f:
            status_data = json.load(f)
        expected = [
            {"file": "1_success.py", "status": "Success", "return_code": "0"},
            {"file": "2_fail.py",
                "status": "Failed (code 1)", "return_code": "1"},
            {"file": "3_unrun.py", "status": "Success", "return_code": "0"}
        ]
        result = [
            {k: entry[k] for k in ["file", "status", "return_code"]}
            for entry in status_data
        ]
        assert sorted(result, key=lambda x: x["file"]) == sorted(
            expected, key=lambda x: x["file"])
        # Verify 2_fail.py and 3_unrun.py logs exist
        failed_log = output_dir / "failed" / "2_fail.log"
        success_log = output_dir / "success" / "3_unrun.log"
        assert failed_log.exists()
        assert success_log.exists()

    def test_run_empty_directory(self, setup_test_dir):
        """Test running in an empty directory."""
        test_dir, output_dir = setup_test_dir
        # Given: An empty directory with no Python files
        for file in test_dir.glob("*.py"):
            file.unlink()
        # When: Running all files
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="all"
        )
        # Then: Verify status file is created but empty
        status_file = output_dir / "files_status.json"
        assert status_file.exists()
        with status_file.open('r') as f:
            status_data = json.load(f)
        expected = []
        result = status_data
        assert result == expected

    def test_run_with_includes_filter(self, setup_test_dir):
        """Test running files with an include filter."""
        test_dir, output_dir = setup_test_dir
        # Given: A directory with Python files and an include filter for specific files
        includes = ["1_success.py"]
        # When: Running files with the include filter
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="all",
            includes=includes
        )
        # Then: Verify only included file was run
        status_file = output_dir / "files_status.json"
        assert status_file.exists()
        with status_file.open('r') as f:
            status_data = json.load(f)
        expected = [
            {"file": "1_success.py", "status": "Success", "return_code": "0"}
        ]
        result = [
            {k: entry[k] for k in ["file", "status", "return_code"]}
            for entry in status_data
        ]
        assert sorted(result, key=lambda x: x["file"]) == sorted(
            expected, key=lambda x: x["file"])

    def test_preserve_existing_status(self, setup_test_dir):
        """Test preserving status of files not rerun."""
        test_dir, output_dir = setup_test_dir
        # Given: Run all files to create status file
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="all"
        )
        # When: Rerun only unrun files (none exist)
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="unrun"
        )
        # Then: Verify all previous statuses are preserved
        status_file = output_dir / "files_status.json"
        assert status_file.exists()
        with status_file.open('r') as f:
            status_data = json.load(f)
        expected = [
            {"file": "1_success.py", "status": "Success", "return_code": "0"},
            {"file": "2_fail.py",
                "status": "Failed (code 1)", "return_code": "1"},
            {"file": "3_unrun.py", "status": "Success", "return_code": "0"}
        ]
        result = [
            {k: entry[k] for k in ["file", "status", "return_code"]}
            for entry in status_data
        ]
        assert sorted(result, key=lambda x: x["file"]) == sorted(
            expected, key=lambda x: x["file"])

    def test_invalid_rerun_mode(self, setup_test_dir):
        """Test handling of invalid rerun mode."""
        test_dir, output_dir = setup_test_dir
        # Given: A directory with Python files and an invalid rerun mode
        invalid_mode = "invalid_mode"
        # When: Attempting to run with an invalid rerun mode
        with pytest.raises(ValueError):
            run_python_files_in_directory(
                target_dir=test_dir,
                output_dir=output_dir,
                rerun_mode=invalid_mode  # type: ignore
            )
        # Then: Verify no status file is created
        status_file = output_dir / "files_status.json"
        assert not status_file.exists()

    def test_recursive_file_discovery(self, setup_test_dir):
        """Test recursive discovery of Python files."""
        test_dir, output_dir = setup_test_dir
        # Given: A directory with a nested subdirectory containing a Python file
        sub_dir = test_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "4_nested.py").write_text('print("Nested Success")')
        # When: Running files with recursive=True
        run_python_files_in_directory(
            target_dir=test_dir,
            output_dir=output_dir,
            rerun_mode="all",
            recursive=True
        )
        # Then: Verify nested file was run
        status_file = output_dir / "files_status.json"
        assert status_file.exists()
        with status_file.open('r') as f:
            status_data = json.load(f)
        expected = [
            {"file": "1_success.py", "status": "Success", "return_code": "0"},
            {"file": "2_fail.py",
                "status": "Failed (code 1)", "return_code": "1"},
            {"file": "3_unrun.py", "status": "Success", "return_code": "0"},
            {"file": f"subdir{os.sep}4_nested.py",
                "status": "Success", "return_code": "0"}
        ]
        result = [
            {k: entry[k] for k in ["file", "status", "return_code"]}
            for entry in status_data
        ]
        assert sorted(result, key=lambda x: x["file"]) == sorted(
            expected, key=lambda x: x["file"])

    def test_run_no_status_file(self, setup_test_dir):
        """Test running all files when no files_status.json exists regardless of rerun_mode."""
        # Given: A directory with test scripts and an output directory without a status file
        test_dir, output_dir = setup_test_dir

        # When: Running python files with any rerun_mode but no status file
        for mode in ["failed", "unrun", "failed_and_unrun", "all"]:
            run_python_files_in_directory(
                target_dir=test_dir,
                output_dir=output_dir,
                rerun_mode=mode  # Should default to "all" since no status file
            )

            # Then: All files should be run and status file should be created
            status_file = output_dir / "files_status.json"
            assert status_file.exists()
            with status_file.open('r') as f:
                status_data = json.load(f)
            expected = [
                {"file": "1_success.py", "status": "Success", "return_code": "0"},
                {"file": "2_fail.py", "status": "Failed (code 1)", "return_code": "1"},
                {"file": "3_unrun.py", "status": "Success", "return_code": "0"}
            ]
            result = [
                {k: entry[k] for k in ["file", "status", "return_code"]}
                for entry in status_data
            ]
            assert sorted(result, key=lambda x: x["file"]) == sorted(
                expected, key=lambda x: x["file"])

            # Clean up status file for next iteration
            status_file.unlink()

    def test_run_empty_status_file(self, setup_test_dir):
        """Test running all files when files_status.json exists but is empty."""
        # Given: A directory with test scripts and an output directory with an empty status file
        test_dir, output_dir = setup_test_dir
        status_file = output_dir / "files_status.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with status_file.open('w') as f:
            json.dump([], f)  # Create an empty status file

        # When: Running python files with any rerun_mode and an empty status file
        for mode in ["failed", "unrun", "failed_and_unrun", "all"]:
            run_python_files_in_directory(
                target_dir=test_dir,
                output_dir=output_dir,
                rerun_mode=mode  # Should default to "all" since status file is empty
            )

            # Then: All files should be run and status file should be updated
            assert status_file.exists()
            with status_file.open('r') as f:
                status_data = json.load(f)
            expected = [
                {"file": "1_success.py", "status": "Success", "return_code": "0"},
                {"file": "2_fail.py", "status": "Failed (code 1)", "return_code": "1"},
                {"file": "3_unrun.py", "status": "Success", "return_code": "0"}
            ]
            result = [
                {k: entry[k] for k in ["file", "status", "return_code"]}
                for entry in status_data
            ]
            assert sorted(result, key=lambda x: x["file"]) == sorted(
                expected, key=lambda x: x["file"])

            # Clean up status file for next iteration
            status_file.unlink()
