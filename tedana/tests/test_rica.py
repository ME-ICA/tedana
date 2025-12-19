"""Tests for tedana.rica module."""

import platform
import stat
from pathlib import Path
from unittest.mock import patch

from tedana import rica


class TestGetRicaCacheDir:
    """Tests for get_rica_cache_dir function."""

    def test_returns_path_object(self):
        """Test that function returns a Path object."""
        result = rica.get_rica_cache_dir()
        assert isinstance(result, Path)

    def test_creates_directory(self, tmp_path):
        """Test that the cache directory is created if it doesn't exist."""
        with patch.object(Path, "home", return_value=tmp_path):
            cache_dir = rica.get_rica_cache_dir()
            assert cache_dir.exists()
            assert cache_dir.is_dir()

    def test_platform_specific_paths(self, tmp_path):
        """Test platform-specific cache directory paths."""
        with patch.object(Path, "home", return_value=tmp_path):
            with patch("platform.system", return_value="Linux"):
                cache_dir = rica.get_rica_cache_dir()
                assert ".cache" in str(cache_dir)

            with patch("platform.system", return_value="Darwin"):
                cache_dir = rica.get_rica_cache_dir()
                assert "Library" in str(cache_dir) or "Caches" in str(cache_dir)


class TestGenerateRicaLauncherScript:
    """Tests for generate_rica_launcher_script function."""

    def test_creates_launcher_script(self, tmp_path):
        """Test that launcher script is created in output directory."""
        launcher_path = rica.generate_rica_launcher_script(tmp_path)

        assert launcher_path.exists()
        assert launcher_path.name == "open_rica_report.py"
        assert launcher_path.parent == tmp_path

    def test_launcher_script_is_executable_on_unix(self, tmp_path):
        """Test that launcher script is executable on Unix systems."""
        launcher_path = rica.generate_rica_launcher_script(tmp_path)

        if platform.system() != "Windows":
            # Check that user execute bit is set
            mode = launcher_path.stat().st_mode
            assert mode & stat.S_IXUSR

    def test_launcher_script_contains_required_elements(self, tmp_path):
        """Test that launcher script contains essential code."""
        launcher_path = rica.generate_rica_launcher_script(tmp_path)
        content = launcher_path.read_text()

        # Check for essential elements
        assert "#!/usr/bin/env python3" in content
        assert "RicaHandler" in content
        assert "http.server" in content
        assert "webbrowser" in content
        assert "def main():" in content
        assert "RICA_FILE_PATTERNS" in content

    def test_custom_port(self, tmp_path):
        """Test that custom port is included in the script."""
        custom_port = 9999
        launcher_path = rica.generate_rica_launcher_script(tmp_path, port=custom_port)
        content = launcher_path.read_text()

        assert str(custom_port) in content


class TestIsCachedFunctions:
    """Tests for cache checking functions."""

    def test_get_cached_rica_version_no_file(self, tmp_path):
        """Test that None is returned when VERSION file doesn't exist."""
        result = rica.get_cached_rica_version(tmp_path)
        assert result is None

    def test_get_cached_rica_version_with_file(self, tmp_path):
        """Test that version is returned when VERSION file exists."""
        version_file = tmp_path / "VERSION"
        version_file.write_text("v2.0.0\n")

        result = rica.get_cached_rica_version(tmp_path)
        assert result == "v2.0.0"

    def test_is_rica_cached_false_when_files_missing(self, tmp_path):
        """Test that is_rica_cached returns False when files are missing."""
        result = rica.is_rica_cached(tmp_path)
        assert result is False

    def test_is_rica_cached_true_when_all_files_present(self, tmp_path):
        """Test that is_rica_cached returns True when all files exist."""
        for filename in rica.RICA_FILES:
            (tmp_path / filename).write_text("dummy content")

        result = rica.is_rica_cached(tmp_path)
        assert result is True

    def test_is_rica_cached_false_when_partial_files(self, tmp_path):
        """Test that is_rica_cached returns False when some files are missing."""
        # Only create one file
        (tmp_path / rica.RICA_FILES[0]).write_text("dummy content")

        result = rica.is_rica_cached(tmp_path)
        assert result is False


class TestSetupRicaReport:
    """Tests for setup_rica_report function."""

    def test_creates_rica_directory(self, tmp_path):
        """Test that rica subdirectory is created."""
        # Create mock cached files
        mock_cache_dir = tmp_path / "cache"
        mock_cache_dir.mkdir()
        for filename in rica.RICA_FILES:
            (mock_cache_dir / filename).write_text("dummy content")
        (mock_cache_dir / "VERSION").write_text("v2.0.0")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch.object(rica, "download_rica", return_value=mock_cache_dir):
            rica.setup_rica_report(output_dir)

        rica_dir = output_dir / "rica"
        assert rica_dir.exists()
        assert rica_dir.is_dir()

    def test_copies_rica_files(self, tmp_path):
        """Test that Rica files are copied to output directory."""
        # Create mock cached files
        mock_cache_dir = tmp_path / "cache"
        mock_cache_dir.mkdir()
        for filename in rica.RICA_FILES:
            (mock_cache_dir / filename).write_text(f"content of {filename}")
        (mock_cache_dir / "VERSION").write_text("v2.0.0")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch.object(rica, "download_rica", return_value=mock_cache_dir):
            rica.setup_rica_report(output_dir)

        rica_dir = output_dir / "rica"
        for filename in rica.RICA_FILES:
            assert (rica_dir / filename).exists()
            assert (rica_dir / filename).read_text() == f"content of {filename}"

    def test_creates_launcher_script(self, tmp_path):
        """Test that launcher script is created in output directory."""
        # Create mock cached files
        mock_cache_dir = tmp_path / "cache"
        mock_cache_dir.mkdir()
        for filename in rica.RICA_FILES:
            (mock_cache_dir / filename).write_text("dummy content")
        (mock_cache_dir / "VERSION").write_text("v2.0.0")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch.object(rica, "download_rica", return_value=mock_cache_dir):
            launcher_path = rica.setup_rica_report(output_dir)

        assert launcher_path is not None
        assert launcher_path.exists()
        assert launcher_path.name == "open_rica_report.py"
        assert launcher_path.parent == output_dir

    def test_returns_none_on_failure(self, tmp_path):
        """Test that None is returned when setup fails."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch.object(rica, "download_rica", side_effect=RuntimeError("Download failed")):
            result = rica.setup_rica_report(output_dir)

        assert result is None


class TestGetRicaVersion:
    """Tests for get_rica_version function."""

    def test_returns_cached_version(self, tmp_path):
        """Test that cached version is returned."""
        # Create VERSION file in cache dir
        with patch.object(rica, "get_rica_cache_dir", return_value=tmp_path):
            (tmp_path / "VERSION").write_text("v2.0.0")
            result = rica.get_rica_version()
            assert result == "v2.0.0"

    def test_returns_none_when_not_cached(self, tmp_path):
        """Test that None is returned when not cached."""
        with patch.object(rica, "get_rica_cache_dir", return_value=tmp_path):
            result = rica.get_rica_version()
            assert result is None
