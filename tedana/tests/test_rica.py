"""Tests for tedana.rica module."""

import json
import platform
import stat
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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

    def test_windows_with_localappdata(self, tmp_path, monkeypatch):
        """Test Windows path using LOCALAPPDATA environment variable."""
        local_app_data = tmp_path / "AppData" / "Local"
        local_app_data.mkdir(parents=True)

        with patch("platform.system", return_value="Windows"):
            monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
            cache_dir = rica.get_rica_cache_dir()

            assert cache_dir.exists()
            assert "tedana" in str(cache_dir)
            assert "rica" in str(cache_dir)
            assert str(local_app_data) in str(cache_dir)

    def test_windows_without_localappdata(self, tmp_path, monkeypatch):
        """Test Windows fallback when LOCALAPPDATA is not set."""
        with patch.object(Path, "home", return_value=tmp_path):
            with patch("platform.system", return_value="Windows"):
                monkeypatch.delenv("LOCALAPPDATA", raising=False)
                cache_dir = rica.get_rica_cache_dir()

                assert cache_dir.exists()
                assert "AppData" in str(cache_dir) and "Local" in str(cache_dir)


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

    def test_creates_launcher_script(self, tmp_path):
        """Test that launcher script is created in output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        launcher_path = rica.setup_rica_report(output_dir)

        assert launcher_path is not None
        assert launcher_path.exists()
        assert launcher_path.name == "open_rica_report.py"
        assert launcher_path.parent == output_dir

    def test_launcher_script_contains_download_logic(self, tmp_path):
        """Test that launcher script contains Rica download functionality."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        launcher_path = rica.setup_rica_report(output_dir)
        content = launcher_path.read_text()

        # Check for download-related functions
        assert "def download_rica" in content
        assert "def setup_rica" in content
        assert "def get_rica_cache_dir" in content
        assert "RICA_GITHUB_API" in content

    def test_launcher_script_contains_server_logic(self, tmp_path):
        """Test that launcher script contains server functionality."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        launcher_path = rica.setup_rica_report(output_dir)
        content = launcher_path.read_text()

        # Check for server-related elements
        assert "RicaHandler" in content
        assert "http.server" in content
        assert "webbrowser" in content
        assert "def main():" in content

    def test_launcher_script_has_force_download_option(self, tmp_path):
        """Test that launcher script supports --force-download flag."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        launcher_path = rica.setup_rica_report(output_dir)
        content = launcher_path.read_text()

        assert "--force-download" in content


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


class TestGetLatestRicaVersion:
    """Tests for get_latest_rica_version function."""

    def test_successful_api_response_with_valid_assets(self):
        """Test successful API response with valid assets."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "tag_name": "v2.0.0",
                "assets": [
                    {
                        "name": "index.html",
                        "browser_download_url": "http://example.com/index.html",
                    },
                    {
                        "name": "rica_server.py",
                        "browser_download_url": "http://example.com/rica_server.py",
                    },
                ],
            }
        ).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            version, assets = rica.get_latest_rica_version()

            assert version == "v2.0.0"
            assert "index.html" in assets
            assert "rica_server.py" in assets
            assert assets["index.html"] == "http://example.com/index.html"
            assert assets["rica_server.py"] == "http://example.com/rica_server.py"

    def test_filters_non_rica_assets(self):
        """Test that only Rica files are included in assets dict."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "tag_name": "v2.1.0",
                "assets": [
                    {
                        "name": "index.html",
                        "browser_download_url": "http://example.com/index.html",
                    },
                    {
                        "name": "rica_server.py",
                        "browser_download_url": "http://example.com/rica_server.py",
                    },
                    {
                        "name": "some_other_file.txt",
                        "browser_download_url": "http://example.com/other.txt",
                    },
                ],
            }
        ).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            version, assets = rica.get_latest_rica_version()

            assert version == "v2.1.0"
            assert len(assets) == 2
            assert "some_other_file.txt" not in assets

    def test_value_error_when_no_assets_found(self):
        """Test ValueError when no assets found in release."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"tag_name": "v2.0.0", "assets": []}).encode(
            "utf-8"
        )
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(ValueError, match="No Rica assets found"):
                rica.get_latest_rica_version()

    def test_value_error_when_no_matching_assets(self):
        """Test ValueError when release has assets but none match Rica files."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "tag_name": "v2.0.0",
                "assets": [
                    {
                        "name": "other_file.txt",
                        "browser_download_url": "http://example.com/other.txt",
                    }
                ],
            }
        ).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(ValueError, match="No Rica assets found"):
                rica.get_latest_rica_version()

    def test_url_error_when_network_fails(self):
        """Test URLError when network fails."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network error")):
            with pytest.raises(urllib.error.URLError):
                rica.get_latest_rica_version()

    def test_timeout_error(self):
        """Test URLError on timeout."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Timeout")):
            with pytest.raises(urllib.error.URLError):
                rica.get_latest_rica_version()


class TestDownloadRicaFile:
    """Tests for download_rica_file function."""

    def test_successful_file_download(self, tmp_path):
        """Test successful file download."""
        file_content = b"This is mock Rica file content"
        dest_path = tmp_path / "test_file.html"

        mock_response = MagicMock()
        mock_response.read.return_value = file_content
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            rica.download_rica_file("http://example.com/file.html", dest_path)

            assert dest_path.exists()
            assert dest_path.read_bytes() == file_content

    def test_url_error_on_download_failure(self, tmp_path):
        """Test URLError on download failure."""
        dest_path = tmp_path / "test_file.html"

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Download failed"),
        ):
            with pytest.raises(urllib.error.URLError):
                rica.download_rica_file("http://example.com/file.html", dest_path)

        # File should not be created
        assert not dest_path.exists()

    def test_http_error_on_404(self, tmp_path):
        """Test HTTPError on 404 response."""
        dest_path = tmp_path / "test_file.html"

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "http://example.com/file.html", 404, "Not Found", {}, None
            ),
        ):
            with pytest.raises(urllib.error.HTTPError):
                rica.download_rica_file("http://example.com/file.html", dest_path)

        assert not dest_path.exists()


class TestDownloadRica:
    """Tests for download_rica function."""

    def test_uses_cached_version_when_up_to_date(self, tmp_path):
        """Test using cached version when already cached and up-to-date."""
        # Set up cache with all files and version
        for filename in rica.RICA_FILES:
            (tmp_path / filename).write_text(f"content of {filename}")
        (tmp_path / "VERSION").write_text("v2.0.0")

        with patch.object(rica, "get_rica_cache_dir", return_value=tmp_path):
            # Mock get_latest_rica_version to not be called (since we shouldn't check)
            with patch.object(
                rica,
                "get_latest_rica_version",
                side_effect=RuntimeError("Should not be called"),
            ):
                result = rica.download_rica()

                assert result == tmp_path
                assert (tmp_path / "VERSION").read_text() == "v2.0.0"

    def test_fallback_to_cache_when_network_fails(self, tmp_path):
        """Test fallback to cache when network fails."""
        # Set up cache with all files and version
        for filename in rica.RICA_FILES:
            (tmp_path / filename).write_text(f"content of {filename}")
        (tmp_path / "VERSION").write_text("v1.5.0")

        with patch.object(rica, "get_rica_cache_dir", return_value=tmp_path):
            with patch.object(
                rica,
                "get_latest_rica_version",
                side_effect=urllib.error.URLError("Network error"),
            ):
                result = rica.download_rica()

                assert result == tmp_path
                # Should still have old version
                assert (tmp_path / "VERSION").read_text() == "v1.5.0"

    def test_skips_download_when_already_have_latest_version(self, tmp_path):
        """Test skipping download when already have latest version."""
        # Set up cache with all files and version
        for filename in rica.RICA_FILES:
            (tmp_path / filename).write_text(f"content of {filename}")
        (tmp_path / "VERSION").write_text("v2.0.0")

        mock_assets = {
            "index.html": "http://example.com/index.html",
            "rica_server.py": "http://example.com/rica_server.py",
        }

        with patch.object(rica, "get_rica_cache_dir", return_value=tmp_path):
            with patch.object(
                rica, "get_latest_rica_version", return_value=("v2.0.0", mock_assets)
            ):
                # Mock download_rica_file to fail if called
                with patch.object(
                    rica,
                    "download_rica_file",
                    side_effect=RuntimeError("Should not download"),
                ):
                    result = rica.download_rica()

                    assert result == tmp_path
                    assert (tmp_path / "VERSION").read_text() == "v2.0.0"

    def test_successful_download_flow(self, tmp_path):
        """Test successful download flow."""
        mock_assets = {
            "index.html": "http://example.com/index.html",
            "rica_server.py": "http://example.com/rica_server.py",
        }

        def mock_download_file(_url, dest_path):
            """Mock download that creates files with content."""
            dest_path.write_text(f"Downloaded from {_url}")

        with patch.object(rica, "get_rica_cache_dir", return_value=tmp_path):
            with patch.object(
                rica, "get_latest_rica_version", return_value=("v2.1.0", mock_assets)
            ):
                with patch.object(rica, "download_rica_file", side_effect=mock_download_file):
                    result = rica.download_rica()

                    assert result == tmp_path
                    assert (tmp_path / "VERSION").read_text() == "v2.1.0"
                    for filename in rica.RICA_FILES:
                        assert (tmp_path / filename).exists()

    def test_warning_when_asset_missing_from_release(self, tmp_path):
        """Test warning when asset missing from release."""
        # Only provide one of the two required assets
        mock_assets = {"index.html": "http://example.com/index.html"}

        def mock_download_file(_url, dest_path):
            """Mock download that creates files with content."""
            dest_path.write_text(f"Downloaded from {_url}")

        with patch.object(rica, "get_rica_cache_dir", return_value=tmp_path):
            with patch.object(
                rica, "get_latest_rica_version", return_value=("v2.1.0", mock_assets)
            ):
                with patch.object(rica, "download_rica_file", side_effect=mock_download_file):
                    result = rica.download_rica()

                    assert result == tmp_path
                    # Only index.html should be downloaded
                    assert (tmp_path / "index.html").exists()
                    # rica_server.py should not be downloaded
                    assert not (tmp_path / "rica_server.py").exists()

    def test_runtime_error_when_download_fails(self, tmp_path):
        """Test RuntimeError when download fails."""
        mock_assets = {
            "index.html": "http://example.com/index.html",
            "rica_server.py": "http://example.com/rica_server.py",
        }

        with patch.object(rica, "get_rica_cache_dir", return_value=tmp_path):
            with patch.object(
                rica, "get_latest_rica_version", return_value=("v2.1.0", mock_assets)
            ):
                with patch.object(
                    rica,
                    "download_rica_file",
                    side_effect=urllib.error.URLError("Download failed"),
                ):
                    with pytest.raises(RuntimeError, match="Failed to download"):
                        rica.download_rica()

    def test_force_redownloads_when_cached(self, tmp_path):
        """Test force=True re-downloads even when cached."""
        # Set up cache with all files and version
        for filename in rica.RICA_FILES:
            (tmp_path / filename).write_text("old content")
        (tmp_path / "VERSION").write_text("v2.0.0")

        mock_assets = {
            "index.html": "http://example.com/index.html",
            "rica_server.py": "http://example.com/rica_server.py",
        }

        def mock_download_file(_url, dest_path):  # noqa: U101
            """Mock download that creates files with new content."""
            dest_path.write_text("new content")

        with patch.object(rica, "get_rica_cache_dir", return_value=tmp_path):
            with patch.object(
                rica, "get_latest_rica_version", return_value=("v2.0.0", mock_assets)
            ):
                with patch.object(rica, "download_rica_file", side_effect=mock_download_file):
                    result = rica.download_rica(force=True)

                    assert result == tmp_path
                    # Files should have new content
                    for filename in rica.RICA_FILES:
                        assert (tmp_path / filename).read_text() == "new content"

    def test_runtime_error_when_network_fails_and_no_cache(self, tmp_path):
        """Test RuntimeError when network fails and nothing is cached."""
        with patch.object(rica, "get_rica_cache_dir", return_value=tmp_path):
            with patch.object(
                rica,
                "get_latest_rica_version",
                side_effect=urllib.error.URLError("Network error"),
            ):
                with pytest.raises(RuntimeError, match="Failed to download Rica"):
                    rica.download_rica()


class TestLocalRicaPath:
    """Tests for local Rica path support."""

    def _create_mock_rica_files(self, rica_dir):
        """Create mock Rica files in a directory.

        Parameters
        ----------
        rica_dir : Path
            Directory to create mock Rica files in.
        """
        rica_dir.mkdir(parents=True, exist_ok=True)
        for filename in rica.RICA_FILES:
            (rica_dir / filename).write_text(f"mock content for {filename}")

    def test_validate_rica_path_valid(self, tmp_path):
        """Test that validation passes when all required files exist."""
        rica_dir = tmp_path / "rica"
        self._create_mock_rica_files(rica_dir)

        result = rica.validate_rica_path(rica_dir)

        assert result is True

    def test_validate_rica_path_missing_files(self, tmp_path):
        """Test that validation fails when files are missing."""
        rica_dir = tmp_path / "rica"
        rica_dir.mkdir()
        # Only create one file, not all required files
        (rica_dir / rica.RICA_FILES[0]).write_text("mock content")

        result = rica.validate_rica_path(rica_dir)

        assert result is False

    def test_validate_rica_path_not_directory(self, tmp_path):
        """Test that validation fails for non-existent or file paths."""
        # Test non-existent path
        non_existent = tmp_path / "does_not_exist"
        result = rica.validate_rica_path(non_existent)
        assert result is False

        # Test path that is a file, not a directory
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("I am a file")
        result = rica.validate_rica_path(file_path)
        assert result is False

    def test_get_rica_from_local_valid(self, tmp_path):
        """Test getting Rica from a valid local path."""
        rica_dir = tmp_path / "rica"
        self._create_mock_rica_files(rica_dir)

        result = rica.get_rica_from_local(rica_dir)

        assert result == rica_dir
        assert result.is_dir()

    def test_get_rica_from_local_invalid(self, tmp_path):
        """Test that an error is raised for invalid paths."""
        # Test non-existent path
        non_existent = tmp_path / "does_not_exist"
        with pytest.raises(ValueError, match="does not exist"):
            rica.get_rica_from_local(non_existent)

        # Test path that is a file, not a directory
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("I am a file")
        with pytest.raises(ValueError, match="is not a directory"):
            rica.get_rica_from_local(file_path)

        # Test directory missing required files
        incomplete_dir = tmp_path / "incomplete"
        incomplete_dir.mkdir()
        (incomplete_dir / rica.RICA_FILES[0]).write_text("mock content")
        with pytest.raises(ValueError, match="missing required files"):
            rica.get_rica_from_local(incomplete_dir)

    def test_setup_rica_report_creates_launcher(self, tmp_path):
        """Test that setup_rica_report creates launcher script.

        Note: The launcher script handles environment variable detection
        and Rica file setup at runtime, not during tedana execution.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        launcher_path = rica.setup_rica_report(output_dir)

        # Verify launcher script was created
        assert launcher_path is not None
        assert launcher_path.exists()
        assert launcher_path.name == "open_rica_report.py"

        # Verify launcher script contains env var detection logic
        content = launcher_path.read_text()
        assert "TEDANA_RICA_PATH" in content
