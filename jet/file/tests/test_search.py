from jet.file.search import traverse_directory


class TestTraverseDirectory:
    def test_traverse_directory_depth_zero(self, tmp_path):
        # Setup: Create a temporary directory structure
        base_dir = tmp_path / "test_base"
        base_dir.mkdir()
        (base_dir / "dir1").mkdir()
        (base_dir / "dir2").mkdir()
        (base_dir / "dir1" / "subdir1").mkdir()
        (base_dir / "dir2" / "subdir2").mkdir()

        # Expected: Only immediate subdirectories of base_dir
        expected = [
            (str(base_dir / "dir1"), 0),
            (str(base_dir / "dir2"), 0),
        ]

        # Result: Collect directories from traverse_directory
        result = []
        for folder, depth in traverse_directory(
            base_dir=str(base_dir),
            includes=["*"],
            excludes=[],
            max_forward_depth=0,
            direction="forward"
        ):
            result.append((folder, depth))

        # Assert: Compare result with expected, ensuring only immediate subdirs are returned
        assert sorted(result) == sorted(
            expected), "Should only return immediate subdirectories at depth 0"

    def test_traverse_directory_depth_zero_excludes_base(self, tmp_path):
        # Setup: Create a temporary directory structure
        base_dir = tmp_path / "test_base"
        base_dir.mkdir()
        (base_dir / "dir1").mkdir()

        # Expected: Only immediate subdirectory, excluding base_dir
        expected = [(str(base_dir / "dir1"), 0)]

        # Result: Collect directories from traverse_directory
        result = []
        for folder, depth in traverse_directory(
            base_dir=str(base_dir),
            includes=["*"],
            excludes=[],
            max_forward_depth=0,
            direction="forward"
        ):
            result.append((folder, depth))

        # Assert: Compare result with expected, ensuring base_dir is excluded
        assert sorted(result) == sorted(
            expected), "Should exclude base directory and return only immediate subdirectory"

    def test_traverse_directory_depth_non_zero(self, tmp_path):
        # Setup: Create a temporary directory structure
        base_dir = tmp_path / "test_base"
        base_dir.mkdir()
        (base_dir / "dir1").mkdir()
        (base_dir / "dir2").mkdir()
        (base_dir / "dir1" / "subdir1").mkdir()

        # Expected: Directories up to depth 1, excluding base_dir
        expected = [
            (str(base_dir / "dir1"), 0),
            (str(base_dir / "dir2"), 0),
            (str(base_dir / "dir1" / "subdir1"), 1),
        ]

        # Result: Collect directories from traverse_directory
        result = []
        for folder, depth in traverse_directory(
            base_dir=str(base_dir),
            includes=["*"],
            excludes=[],
            max_forward_depth=1,
            direction="forward"
        ):
            result.append((folder, depth))

        # Assert: Compare result with expected
        assert sorted(result) == sorted(
            expected), "Should return directories up to depth 1, excluding base_dir"
