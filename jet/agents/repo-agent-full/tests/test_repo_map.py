"""
Tests for the Repository Map generator.
"""

from repo_map import RepoMap


class TestRepoMap:
    def test_generate_map(self, temp_repo_dir):
        """Test that the map is generated correctly."""
        mapper = RepoMap(str(temp_repo_dir))
        map_data = mapper.generate()

        assert "structure" in map_data
        assert "symbols" in map_data

        # Check if main.py is in the symbols
        assert "main.py" in map_data["symbols"]

        # Check if calculate_sum is in the symbols for main.py
        assert any("calculate_sum" in s for s in map_data["symbols"]["main.py"])

    def test_get_summary_for_query(self, temp_repo_dir):
        """Test that the summary string is generated."""
        mapper = RepoMap(str(temp_repo_dir))
        mapper.generate()

        summary = mapper.get_summary_for_query("Test query")

        assert "Project Structure" in summary
        assert "Key Symbols by File" in summary
        assert "main.py" in summary

    def test_empty_repo_map(self, tmp_path):
        """Test map generation for an empty directory."""
        empty_dir = tmp_path / "empty_repo"
        empty_dir.mkdir()

        mapper = RepoMap(str(empty_dir))
        map_data = mapper.generate()

        assert map_data["symbols"] == {}
        assert map_data["structure"] == {}
