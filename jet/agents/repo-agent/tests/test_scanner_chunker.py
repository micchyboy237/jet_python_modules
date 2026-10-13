"""
Tests for RepositoryScanner and CodeChunker.
"""


class TestScannerChunker:
    def test_scan_valid_repo(self, temp_repo_dir, scanner):
        files = scanner.scan(str(temp_repo_dir))
        assert len(files) == 3  # main.py, utils.py, README.md

        # Check content of one file
        main_file = [f for f in files if "main.py" in f[0]][0]
        assert "calculate_sum" in main_file[2]
        assert main_file[1] == "python"

    def test_chunking_logic(self, temp_repo_dir, scanner, chunker):
        files = scanner.scan(str(temp_repo_dir))
        chunks = chunker.chunk_files(files)

        assert len(chunks) > 0

        # Verify chunk structure
        first_chunk = chunks[0]
        assert "content" in first_chunk
        assert "file_path" in first_chunk
        assert "id" in first_chunk
        assert "start_line" in first_chunk

    def test_chunk_overlap(self, temp_repo_dir, scanner, chunker):
        # Create a file large enough to trigger multiple chunks
        # 800 tokens ~ 3200 chars. Let's make it 5000 chars to be safe.
        large_file = temp_repo_dir / "large.py"
        # Each line is ~30 chars. 200 lines = 6000 chars.
        content = "\n".join(
            [
                f"def func_{i}():\n    return {i} # This is a comment to add length to the line"
                for i in range(200)
            ]
        )
        large_file.write_text(content)

        files = scanner.scan(str(temp_repo_dir))
        chunks = chunker.chunk_files(files)

        large_chunks = [c for c in chunks if "large.py" in c["file_path"]]
        assert len(large_chunks) > 1, (
            f"Expected multiple chunks, got {len(large_chunks)}"
        )

        # Check that IDs are unique
        ids = [c["id"] for c in large_chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"
