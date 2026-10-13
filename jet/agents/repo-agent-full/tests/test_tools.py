"""
Tests for the RepoTools module.
"""

from tools import RepoTools


class TestRepoTools:
    def test_read_file_exists(self, temp_repo_dir):
        """Test reading an existing file."""
        tools = RepoTools(str(temp_repo_dir))
        content = tools.read_file("main.py")

        assert "main.py" in content
        assert "calculate_sum" in content

    def test_read_file_not_found(self, temp_repo_dir):
        """Test reading a non-existent file."""
        tools = RepoTools(str(temp_repo_dir))
        content = tools.read_file("nonexistent.py")

        assert "Error" in content
        assert "not found" in content

    def test_read_file_with_lines(self, temp_repo_dir):
        """Test reading specific lines of a file."""
        tools = RepoTools(str(temp_repo_dir))
        content = tools.read_file("main.py", start_line=2, end_line=3)

        assert "calculate_sum" in content
        assert "calculate_product" not in content  # Should be on line 6+

    def test_grep_search_found(self, temp_repo_dir):
        """Test grep search for an existing pattern."""
        tools = RepoTools(str(temp_repo_dir))
        results = tools.grep_search("calculate_sum")

        assert len(results) > 0
        assert any("main.py" in r for r in results)

    def test_grep_search_not_found(self, temp_repo_dir):
        """Test grep search for a non-existing pattern."""
        tools = RepoTools(str(temp_repo_dir))
        results = tools.grep_search("xyz_nonexistent_function")

        assert len(results) == 0

    def test_list_dir(self, temp_repo_dir):
        """Test listing directory contents."""
        tools = RepoTools(str(temp_repo_dir))
        content = tools.list_dir(".")

        assert "[FILE] main.py" in content
        assert "[FILE] utils.py" in content

    def test_list_dir_invalid(self, temp_repo_dir):
        """Test listing a non-existent directory."""
        tools = RepoTools(str(temp_repo_dir))
        content = tools.list_dir("nonexistent_dir")

        assert "Error" in content
