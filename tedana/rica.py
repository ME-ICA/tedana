"""Rica integration module for tedana.

This module provides functionality to:
1. Download and cache Rica (ICA component visualization tool) from GitHub releases
2. Generate Rica report files in tedana output directories
3. Create cross-platform launcher scripts to open Rica reports

Rica is an interactive web-based visualization tool for exploring ICA components
from tedana analysis. For more information, see: https://github.com/ME-ICA/rica
"""

import json
import logging
import os
import platform
import shutil
import stat
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

LGR = logging.getLogger("GENERAL")

# Rica GitHub repository information
RICA_REPO_OWNER = "ME-ICA"
RICA_REPO_NAME = "rica"
RICA_GITHUB_API = (
    f"https://api.github.com/repos/{RICA_REPO_OWNER}/{RICA_REPO_NAME}/releases/latest"
)

# Files to download from Rica releases
RICA_FILES = ["index.html", "rica_server.py"]

# Environment variable for local Rica path
RICA_PATH_ENV_VAR = "TEDANA_RICA_PATH"

# Bundled Rica path (relative to tedana package)
RICA_BUNDLED_PATH = Path(__file__).parent / "resources" / "rica"


def validate_rica_path(rica_path: Union[str, Path]) -> bool:
    """Check if a local path contains the required Rica files.

    Parameters
    ----------
    rica_path : str or Path
        Path to a local Rica directory to validate.

    Returns
    -------
    bool
        True if the path exists and contains all required Rica files,
        False otherwise.
    """
    rica_path = Path(rica_path)

    if not rica_path.exists():
        LGR.debug(f"Rica path does not exist: {rica_path}")
        return False

    if not rica_path.is_dir():
        LGR.debug(f"Rica path is not a directory: {rica_path}")
        return False

    missing_files = []
    for filename in RICA_FILES:
        if not (rica_path / filename).exists():
            missing_files.append(filename)

    if missing_files:
        LGR.debug(f"Rica path {rica_path} is missing files: {missing_files}")
        return False

    return True


def get_rica_from_local(rica_path: Union[str, Path]) -> Path:
    """Validate and return the path to a local Rica directory.

    Parameters
    ----------
    rica_path : str or Path
        Path to a local Rica directory.

    Returns
    -------
    Path
        Validated path to the local Rica directory.

    Raises
    ------
    ValueError
        If the path does not exist, is not a directory, or is missing
        required Rica files.
    """
    rica_path = Path(rica_path)

    if not rica_path.exists():
        raise ValueError(f"Local Rica path does not exist: {rica_path}")

    if not rica_path.is_dir():
        raise ValueError(f"Local Rica path is not a directory: {rica_path}")

    missing_files = []
    for filename in RICA_FILES:
        if not (rica_path / filename).exists():
            missing_files.append(filename)

    if missing_files:
        raise ValueError(
            f"Local Rica path {rica_path} is missing required files: {missing_files}. "
            f"Required files are: {RICA_FILES}"
        )

    LGR.debug(f"Validated local Rica path: {rica_path}")
    return rica_path


def get_rica_cache_dir() -> Path:
    """Get the platform-specific cache directory for Rica files.

    Returns
    -------
    Path
        Path to Rica cache directory.
        - Linux: ~/.cache/tedana/rica
        - macOS: ~/Library/Caches/tedana/rica
        - Windows: %LOCALAPPDATA%/tedana/rica
    """
    system = platform.system()

    if system == "Linux":
        base_cache = Path.home() / ".cache"
    elif system == "Darwin":  # macOS
        base_cache = Path.home() / "Library" / "Caches"
    elif system == "Windows":
        base_cache = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        # Fallback for other systems
        base_cache = Path.home() / ".cache"

    rica_cache = base_cache / "tedana" / "rica"
    rica_cache.mkdir(parents=True, exist_ok=True)

    return rica_cache


def get_latest_rica_version() -> Tuple[str, Dict]:
    """Fetch the latest Rica version information from GitHub.

    Returns
    -------
    version : str
        The version tag (e.g., "v2.0.0").
    assets : dict
        Dictionary mapping filename to download URL.

    Raises
    ------
    urllib.error.URLError
        If unable to connect to GitHub API.
    ValueError
        If the release has no assets.
    """
    req = urllib.request.Request(
        RICA_GITHUB_API,
        headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "tedana"},
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        release_info = json.loads(response.read().decode("utf-8"))

    version = release_info["tag_name"]
    assets = {}

    for asset in release_info.get("assets", []):
        name = asset["name"]
        if name in RICA_FILES:
            assets[name] = asset["browser_download_url"]

    if not assets:
        raise ValueError(f"No Rica assets found in release {version}")

    return version, assets


def download_rica_file(url: str, dest_path: Path) -> None:
    """Download a single file from a URL.

    Parameters
    ----------
    url : str
        URL to download from.
    dest_path : Path
        Destination file path.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "tedana"})

    with urllib.request.urlopen(req, timeout=60) as response:
        dest_path.write_bytes(response.read())


def get_cached_rica_version(cache_dir: Path) -> Optional[str]:
    """Get the version of Rica currently cached.

    Parameters
    ----------
    cache_dir : Path
        Path to Rica cache directory.

    Returns
    -------
    str or None
        Version string if version file exists, None otherwise.
    """
    version_file = cache_dir / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return None


def is_rica_cached(cache_dir: Path) -> bool:
    """Check if Rica files are cached and complete.

    Parameters
    ----------
    cache_dir : Path
        Path to Rica cache directory.

    Returns
    -------
    bool
        True if all required files exist in cache.
    """
    for filename in RICA_FILES:
        if not (cache_dir / filename).exists():
            return False
    return True


def download_rica(force: bool = False) -> Path:
    """Download Rica files from GitHub releases.

    Downloads the latest Rica release to the platform-specific cache directory.
    If Rica is already cached and up-to-date, skips the download unless force=True.

    Parameters
    ----------
    force : bool, optional
        Force re-download even if Rica is cached. Default: False.

    Returns
    -------
    Path
        Path to Rica cache directory containing the downloaded files.

    Raises
    ------
    RuntimeError
        If download fails after retries.
    """
    cache_dir = get_rica_cache_dir()
    cached_version = get_cached_rica_version(cache_dir)

    # Check if we need to download
    if not force and is_rica_cached(cache_dir) and cached_version:
        LGR.debug(f"Rica {cached_version} already cached at {cache_dir}")
        return cache_dir

    LGR.info("Downloading Rica from GitHub...")

    try:
        latest_version, assets = get_latest_rica_version()
    except (urllib.error.URLError, ValueError) as e:
        if is_rica_cached(cache_dir):
            LGR.warning(
                f"Could not check for Rica updates: {e}. Using cached version {cached_version}."
            )
            return cache_dir
        raise RuntimeError(f"Failed to download Rica: {e}") from e

    # Skip if already have latest version
    if not force and cached_version == latest_version and is_rica_cached(cache_dir):
        LGR.debug(f"Rica {latest_version} is already cached")
        return cache_dir

    LGR.info(f"Downloading Rica {latest_version}...")

    # Download each file
    for filename in RICA_FILES:
        if filename not in assets:
            LGR.warning(f"Rica asset {filename} not found in release")
            continue

        dest_path = cache_dir / filename
        try:
            download_rica_file(assets[filename], dest_path)
            LGR.debug(f"Downloaded {filename}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to download {filename}: {e}") from e

    # Write version file
    (cache_dir / "VERSION").write_text(latest_version)

    LGR.info(f"Rica {latest_version} downloaded to {cache_dir}")
    return cache_dir


def generate_rica_launcher_script(out_dir: Union[str, Path], port: int = 8000) -> Path:
    """Generate the Rica launcher script for a tedana output directory.

    This creates a Python script that, when executed, starts a local server
    and opens Rica in the user's browser to visualize the tedana output.

    Parameters
    ----------
    out_dir : str or Path
        The tedana output directory.
    port : int, optional
        Default port for the local server. Default: 8000.

    Returns
    -------
    Path
        Path to the generated launcher script.
    """
    out_dir = Path(out_dir)
    script_path = out_dir / "open_rica_report.py"

    # Cross-platform launcher script
    script_content = (
        '''#!/usr/bin/env python3
"""
Rica Report Launcher - Opens tedana output in Rica visualization

Usage:
    python open_rica_report.py [--port PORT] [--no-open]

This script starts a local server and opens Rica in your default browser
to visualize the ICA component analysis from this tedana output directory.

Press Ctrl+C to stop the server when done.
"""

import argparse
import http.server
import json
import mimetypes
import sys
import webbrowser
from pathlib import Path
from urllib.parse import unquote

# File patterns that Rica needs
RICA_FILE_PATTERNS = [
    "_metrics.tsv",
    "_mixing.tsv",
    "stat-z_components.nii.gz",
    "_mask.nii",
    "report.txt",
    "comp_",
    ".svg",
    "tedana_",
]

# Ensure proper MIME types
mimetypes.add_type("application/gzip", ".gz")
mimetypes.add_type("text/tab-separated-values", ".tsv")


class RicaHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS support and file listing endpoint."""

    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        """Handle GET requests with special endpoint for file listing."""
        path = unquote(self.path)

        if path == "/api/files":
            self.send_file_list()
        else:
            super().do_GET()

    def send_file_list(self):
        """Return JSON list of Rica-relevant files in current directory."""
        files = []
        cwd = Path.cwd()

        for f in cwd.rglob("*"):
            if f.is_file():
                if any(pattern in f.name for pattern in RICA_FILE_PATTERNS):
                    rel_path = str(f.relative_to(cwd)).replace("\\", "/")
                    files.append(rel_path)

        response_data = {
            "files": sorted(files),
            "path": str(cwd),
            "count": len(files),
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_data, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        """Custom log format - suppress noisy output."""
        try:
            msg = str(args[0]) if args else ""
            if "/api/files" in msg:
                print("[Rica] File list requested")
            elif "GET" in msg and len(args) > 1:
                status = str(args[1])
                if not status.startswith("2"):
                    print(f"[Rica] {msg} - {status}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Open Rica report to visualize tedana ICA components"
    )
    parser.add_argument(
        "--port", type=int, default='''
        + str(port)
        + """, help="Port to serve on"
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Don't auto-open browser"
    )
    args = parser.parse_args()

    # Change to script directory (tedana output folder)
    script_dir = Path(__file__).parent.resolve()

    # Check for Rica files
    rica_index = script_dir / "rica" / "index.html"
    if not rica_index.exists():
        print(f"Error: Rica not found at {rica_index}")
        print("Rica files may not have been downloaded during tedana execution.")
        print("Try re-running tedana with --rica-report flag.")
        sys.exit(1)

    # Change to output directory
    import os
    os.chdir(script_dir)

    # Find a free port if default is busy
    port = args.port
    import socket
    for attempt in range(10):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", port))
            sock.close()
            break
        except OSError:
            port += 1
    else:
        print(f"Error: Could not find free port starting from {args.port}")
        sys.exit(1)

    # Start server
    try:
        with http.server.HTTPServer(("", port), RicaHandler) as httpd:
            url = f"http://localhost:{port}/rica/index.html"
            print()
            print("=" * 60)
            print("Rica - ICA Component Visualization")
            print("=" * 60)
            print()
            print(f"Server running at: http://localhost:{port}")
            print(f"Rica interface:    {url}")
            print(f"Serving files from: {script_dir}")
            print()
            print("Press Ctrl+C to stop the server")
            print()

            if not args.no_open:
                webbrowser.open(url)

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\\nServer stopped.")
        sys.exit(0)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Error: Port {port} is already in use.")
            print(f"Try: python open_rica_report.py --port {port + 1}")
        else:
            raise


if __name__ == "__main__":
    main()
"""
    )

    script_path.write_text(script_content)

    # Make script executable on Unix systems
    if platform.system() != "Windows":
        script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return script_path


def setup_rica_report(out_dir: Union[str, Path]) -> Optional[Path]:
    """Set up Rica report files in a tedana output directory.

    This function:
    1. Gets Rica files from environment variable, bundled resources, or downloads
    2. Copies Rica files to the output directory
    3. Generates the launcher script

    The priority order for obtaining Rica files is:
    1. Check for TEDANA_RICA_PATH environment variable (for developers)
    2. Check for bundled Rica in tedana/resources/rica/
    3. Check for cached Rica
    4. Fall back to downloading from GitHub

    Parameters
    ----------
    out_dir : str or Path
        The tedana output directory.

    Returns
    -------
    Path or None
        Path to the launcher script if successful, None if Rica setup failed.
    """
    out_dir = Path(out_dir)
    source_dir: Optional[Path] = None
    source_description: str = ""

    try:
        # Priority 1: Check TEDANA_RICA_PATH environment variable (for developers)
        env_rica_path = os.environ.get(RICA_PATH_ENV_VAR)
        if env_rica_path:
            if validate_rica_path(env_rica_path):
                source_dir = Path(env_rica_path)
                source_description = f"environment variable ({RICA_PATH_ENV_VAR}): {source_dir}"
                LGR.info(f"Using Rica from {source_description}")
            else:
                LGR.warning(
                    f"{RICA_PATH_ENV_VAR} is set to '{env_rica_path}' but path is invalid "
                    "or missing required files. Falling back to other sources."
                )

        # Priority 2: Check for bundled Rica in tedana/resources/rica/
        if source_dir is None:
            if validate_rica_path(RICA_BUNDLED_PATH):
                source_dir = RICA_BUNDLED_PATH
                source_description = f"bundled resources: {source_dir}"
                LGR.info(f"Using Rica from {source_description}")

        # Priority 3: Check for cached Rica
        if source_dir is None:
            cache_dir = get_rica_cache_dir()
            if is_rica_cached(cache_dir):
                source_dir = cache_dir
                cached_version = get_cached_rica_version(cache_dir)
                source_description = f"cached: {source_dir} (version {cached_version})"
                LGR.info(f"Using Rica from {source_description}")

        # Priority 4: Download from GitHub
        if source_dir is None:
            LGR.info("No local Rica found. Downloading from GitHub...")
            source_dir = download_rica()
            source_description = f"GitHub download (cached): {source_dir}"

        # Create rica subdirectory in output
        rica_dir = out_dir / "rica"
        rica_dir.mkdir(exist_ok=True)

        # Copy Rica files to output
        for filename in RICA_FILES:
            src = source_dir / filename
            dst = rica_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
                LGR.debug(f"Copied {filename} to {rica_dir}")
            else:
                LGR.warning(f"Rica file {filename} not found in {source_description}")

        # Generate launcher script
        launcher_path = generate_rica_launcher_script(out_dir)

        LGR.info(f"Rica report setup complete. Run '{launcher_path.name}' to visualize results.")

        return launcher_path

    except Exception as e:
        LGR.warning(f"Failed to set up Rica report: {e}")
        LGR.warning("You can still view the standard HTML report.")
        return None


def get_rica_version() -> Optional[str]:
    """Get the version of Rica that is cached.

    Returns
    -------
    str or None
        Version string if Rica is cached, None otherwise.
    """
    cache_dir = get_rica_cache_dir()
    return get_cached_rica_version(cache_dir)
