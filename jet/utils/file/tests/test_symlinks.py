from pathlib import Path
from jet.utils.file import resolve_symlinks


class TestResolveSymlinks:
    def test_resolve_file_symlink(self, tmp_path: Path):
        # Given: a real file and a symlink to it
        real_file = tmp_path / "data.txt"
        real_file.write_text("hello world")
        symlink = tmp_path / "link.txt"
        symlink.symlink_to(real_file)

        # When: resolving the single symlink path
        resolve_symlinks(symlink)

        # Then: symlink is replaced with real file content
        assert symlink.exists()
        assert not symlink.is_symlink()
        assert symlink.read_text() == "hello world"

    def test_resolve_directory_symlink(self, tmp_path: Path):
        # Given: a real directory with files and a symlink to it
        real_dir = tmp_path / "assets"
        real_dir.mkdir()
        (real_dir / "a.txt").write_text("A")
        (real_dir / "b.txt").write_text("B")

        symlink = tmp_path / "assets_link"
        symlink.symlink_to(real_dir, target_is_directory=True)

        # When
        resolve_symlinks(symlink)

        # Then
        replaced_dir = tmp_path / "assets_link"
        assert replaced_dir.exists()
        assert not replaced_dir.is_symlink()
        assert (replaced_dir / "a.txt").read_text() == "A"
        assert (replaced_dir / "b.txt").read_text() == "B"

    def test_resolve_all_symlinks_in_directory(self, tmp_path: Path):
        # Given: a directory with multiple symlinks
        base = tmp_path / "project"
        base.mkdir()

        real1 = base / "file1.txt"
        real2 = base / "file2.txt"
        real1.write_text("X")
        real2.write_text("Y")

        link1 = base / "l1.txt"
        link2 = base / "l2.txt"
        link1.symlink_to(real1)
        link2.symlink_to(real2)

        # When
        resolve_symlinks(base)

        # Then
        assert link1.exists() and not link1.is_symlink()
        assert link2.exists() and not link2.is_symlink()
        assert link1.read_text() == "X"
        assert link2.read_text() == "Y"

    def test_broken_symlink_raises(self, tmp_path: Path):
        # Given: broken symlink
        target = tmp_path / "missing.txt"
        symlink = tmp_path / "badlink"
        symlink.symlink_to(target)

        # When / Then
        try:
            resolve_symlinks(symlink)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError as e:
            assert "Broken symlink" in str(e)
