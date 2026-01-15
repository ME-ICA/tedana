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

    This creates a Python script that, when executed:
    1. Checks for Rica files (env var, bundled, cached, or downloads)
    2. Copies Rica files to output directory if needed
    3. Starts a local server and opens Rica in the browser

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

    # Cross-platform launcher script with embedded download logic
    script_content = (
        '''#!/usr/bin/env python3
"""
Rica Report Launcher - Opens tedana output in Rica visualization

Usage:
    python open_rica_report.py [--port PORT] [--no-open] [--force-download]

This script checks for Rica files, downloads them if necessary, and then
starts a local server to visualize the ICA component analysis from this
tedana output directory.

Press Ctrl+C to stop the server when done.
"""

import argparse
import http.server
import json
import mimetypes
import os
import platform
import shutil
import socket
import sys
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import unquote

# Rica configuration
RICA_REPO_OWNER = "ME-ICA"
RICA_REPO_NAME = "rica"
RICA_GITHUB_API = (
    f"https://api.github.com/repos/{RICA_REPO_OWNER}/{RICA_REPO_NAME}/releases/latest"
)
RICA_FILES = ["index.html", "rica_server.py"]
RICA_PATH_ENV_VAR = "TEDANA_RICA_PATH"

# File patterns that Rica needs from tedana output
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


def get_rica_cache_dir():
    """Get platform-specific cache directory for Rica files."""
    system = platform.system()
    if system == "Linux":
        base_cache = Path.home() / ".cache"
    elif system == "Darwin":
        base_cache = Path.home() / "Library" / "Caches"
    elif system == "Windows":
        base_cache = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base_cache = Path.home() / ".cache"
    rica_cache = base_cache / "tedana" / "rica"
    rica_cache.mkdir(parents=True, exist_ok=True)
    return rica_cache


def validate_rica_path(rica_path):
    """Check if path contains required Rica files."""
    rica_path = Path(rica_path)
    if not rica_path.exists() or not rica_path.is_dir():
        return False
    return all((rica_path / f).exists() for f in RICA_FILES)


def get_cached_rica_version(cache_dir):
    """Get cached Rica version if available."""
    version_file = cache_dir / "VERSION"
    return version_file.read_text().strip() if version_file.exists() else None


def download_rica(force=False):
    """Download Rica from GitHub releases."""
    cache_dir = get_rica_cache_dir()
    cached_version = get_cached_rica_version(cache_dir)

    # Check if already cached
    if not force and validate_rica_path(cache_dir) and cached_version:
        print(f"[Rica] Using cached version {cached_version}")
        return cache_dir

    print("[Rica] Downloading from GitHub...")

    try:
        req = urllib.request.Request(
            RICA_GITHUB_API,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "tedana-rica-launcher",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            release_info = json.loads(response.read().decode("utf-8"))

        version = release_info["tag_name"]
        assets = {}
        for asset in release_info.get("assets", []):
            if asset["name"] in RICA_FILES:
                assets[asset["name"]] = asset["browser_download_url"]

        if not assets:
            raise ValueError(f"No Rica assets found in release {version}")

        # Skip if already have latest
        if not force and cached_version == version and validate_rica_path(cache_dir):
            print(f"[Rica] Already have latest version {version}")
            return cache_dir

        print(f"[Rica] Downloading version {version}...")
        for filename, url in assets.items():
            dest_path = cache_dir / filename
            req = urllib.request.Request(url, headers={"User-Agent": "tedana-rica-launcher"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest_path.write_bytes(resp.read())
            print(f"[Rica] Downloaded {filename}")

        (cache_dir / "VERSION").write_text(version)
        print(f"[Rica] Successfully downloaded Rica {version}")
        return cache_dir

    except (urllib.error.URLError, ValueError) as e:
        if validate_rica_path(cache_dir):
            print(f"[Rica] Warning: Could not check for updates ({e})")
            print(f"[Rica] Using cached version {cached_version}")
            return cache_dir
        raise RuntimeError(f"Failed to download Rica: {e}") from e


def setup_rica(output_dir, force_download=False):
    """Set up Rica files in the output directory."""
    output_dir = Path(output_dir)
    rica_dir = output_dir / "rica"

    # Check if Rica already exists in output
    if not force_download and validate_rica_path(rica_dir):
        print("[Rica] Files already present in output directory")
        return rica_dir

    source_dir = None

    # Priority 1: Environment variable
    env_path = os.environ.get(RICA_PATH_ENV_VAR)
    if env_path and validate_rica_path(env_path):
        source_dir = Path(env_path)
        print(f"[Rica] Using path from {RICA_PATH_ENV_VAR}: {source_dir}")

    # Priority 2: Check cache
    if source_dir is None:
        cache_dir = get_rica_cache_dir()
        if validate_rica_path(cache_dir):
            source_dir = cache_dir
            version = get_cached_rica_version(cache_dir)
            print(f"[Rica] Using cached version {version}")

    # Priority 3: Download
    if source_dir is None or force_download:
        source_dir = download_rica(force=force_download)

    # Copy files to output
    rica_dir.mkdir(exist_ok=True)
    for filename in RICA_FILES:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, rica_dir / filename)

    print(f"[Rica] Files ready in {rica_dir}")
    return rica_dir


class RicaHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS support and file listing endpoint."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        path = unquote(self.path)
        if path == "/api/files":
            self.send_file_list()
        else:
            super().do_GET()

    def send_file_list(self):
        files = []
        cwd = Path.cwd()
        for f in cwd.rglob("*"):
            if f.is_file() and any(p in f.name for p in RICA_FILE_PATTERNS):
                files.append(f.relative_to(cwd).as_posix())
        response_data = {"files": sorted(files), "path": str(cwd), "count": len(files)}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_data, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        try:
            msg = str(args[0]) if args else ""
            if "/api/files" in msg:
                print("[Rica] File list requested")
            elif "GET" in msg and len(args) > 1 and not str(args[1]).startswith("2"):
                print(f"[Rica] {msg} - {args[1]}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Open Rica report to visualize tedana ICA components"
    )
    parser.add_argument("--port", type=int, default='''
        + str(port)
        + """, help="Port to serve on")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--force-download", action="store_true", help="Force re-download Rica")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()

    # Set up Rica (download if needed)
    try:
        setup_rica(script_dir, force_download=args.force_download)
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)

    # Verify Rica is ready
    rica_index = script_dir / "rica" / "index.html"
    if not rica_index.exists():
        print(f"Error: Rica not found at {rica_index}")
        sys.exit(1)

    os.chdir(script_dir)

    # Find free port
    port = args.port
    for _ in range(10):
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


def setup_rica_report(out_dir: Union[str, Path]) -> Path:
    """Generate the Rica launcher script in a tedana output directory.

    This function generates a launcher script (open_rica_report.py) that handles
    all Rica setup when executed by the user, including:
    - Checking for Rica files (environment variable, cache)
    - Downloading Rica from GitHub if not available
    - Starting a local server to visualize results

    Parameters
    ----------
    out_dir : str or Path
        The tedana output directory.

    Returns
    -------
    Path
        Path to the generated launcher script.
    """
    out_dir = Path(out_dir)

    # Generate launcher script (all Rica setup happens when user runs the script)
    launcher_path = generate_rica_launcher_script(out_dir)

    LGR.info(f"Rica launcher created. Run 'python {launcher_path}' to visualize results.")

    return launcher_path


def get_rica_version() -> Optional[str]:
    """Get the version of Rica that is cached.

    Returns
    -------
    str or None
        Version string if Rica is cached, None otherwise.
    """
    cache_dir = get_rica_cache_dir()
    return get_cached_rica_version(cache_dir)
