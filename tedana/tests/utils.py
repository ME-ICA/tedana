"""Utility functions for testing tedana."""

import json
import logging
import shutil
import tarfile
import tempfile
from functools import lru_cache
from os import listdir, makedirs, rename
from os.path import abspath, dirname, exists, join, sep

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tedana.workflows import tedana as tedana_cli

# Added a testing logger to output whether or not testing data were downlaoded
TestLGR = logging.getLogger("TESTING")

# File written into each extracted dataset, recording the OSF "last updated" date
# of the archive it came from. Tarball member mtimes do not match that date
# (e.g., the reclassify archive), so it is the only reliable freshness check.
OSF_DATE_FILE = ".osf_updated_date"

# OSF intermittently returns 5xx (and 403 from its storage backend) errors,
# so retry those along with connection and read failures.
N_DOWNLOAD_ATTEMPTS = 3
OSF_RETRY = Retry(
    total=10,
    connect=5,
    read=5,
    status=5,
    backoff_factor=2,
    status_forcelist=(403, 429, 500, 502, 503, 504),
)
OSF_TIMEOUT = (30, 300)


def get_test_data_path():
    """
    Returns the path to test datasets, terminated with separator.

    Test-related
    data are kept in tests folder in "data".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return abspath(join(dirname(__file__), "data") + sep)


def data_for_testing_info(test_dataset=str):
    """
    Get the path and download link for each dataset used for testing.

    Also creates the base directories into which the data and output
    directories are written

    Parameters
    ----------
    test_dataset : str
       References one of the datasets to download. It can be:
        three-echo
        three-echo-reclassify
        four-echo
        five-echo
       To only return the base path without downloading data,
       you can supply:
        path

    Returns
    -------
    test_data_path : str
       The path to the local directory where the data will be downloaded
       If "path" is specified, returns the base directory path.
    osf_id : str
       The ID for the OSF file.
       Data download link would be https://osf.io/osf_id/download
       Metadata download link would be https://osf.io/metadata/osf_id?format=datacite-json
       If "path" is specified, this value is not returned.
    """

    tedana_path = dirname(tedana_cli.__file__)
    base_data_path = abspath(join(tedana_path, "../../.testing_data_cache"))
    makedirs(base_data_path, exist_ok=True)
    makedirs(join(base_data_path, "outputs"), exist_ok=True)
    if test_dataset == "three-echo":
        test_data_path = join(base_data_path, "three-echo/TED.three-echo")
        osf_id = "rqhfc"
        makedirs(join(base_data_path, "three-echo"), exist_ok=True)
        makedirs(join(base_data_path, "outputs/three-echo"), exist_ok=True)
    elif test_dataset == "three-echo-reclassify":
        test_data_path = join(base_data_path, "reclassify")
        osf_id = "f6g45"
        makedirs(join(base_data_path, "outputs/reclassify"), exist_ok=True)
    elif test_dataset == "four-echo":
        test_data_path = join(base_data_path, "four-echo/TED.four-echo")
        osf_id = "gnj73"
        makedirs(join(base_data_path, "four-echo"), exist_ok=True)
        makedirs(join(base_data_path, "outputs/four-echo"), exist_ok=True)
    elif test_dataset == "five-echo":
        test_data_path = join(base_data_path, "five-echo/TED.five-echo")
        osf_id = "9c42e"
        makedirs(join(base_data_path, "five-echo"), exist_ok=True)
        makedirs(join(base_data_path, "outputs/five-echo"), exist_ok=True)
    elif test_dataset == "path":
        return base_data_path
    else:
        raise ValueError(f"{test_dataset} is not a valid dataset string for data_for_testing_info")

    return test_data_path, osf_id


def download_test_data(osf_id, test_data_path):
    """If current data is not already available, downloads tar.gz data.

    Data are stored at `https://osf.io/osf_id/download`.
    It unpacks into `out_path`.

    Parameters
    ----------
    osf_id : str
       The ID for the OSF file.
    test_data_path : str
        Path to directory where OSF data should be extracted
    """
    osf_date = _get_osf_updated_date(osf_id)
    local_date = _get_local_osf_date(test_data_path)
    local_data_exists = exists(test_data_path) and len(listdir(test_data_path)) > 0

    if local_data_exists:
        if osf_date is None:
            TestLGR.warning(
                f"Using local copy of testing data in {test_data_path} "
                "but cannot validate that local copy is up-to-date"
            )
            return

        if local_date == osf_date:
            TestLGR.info(
                f"Downloaded and up-to-date data already in {test_data_path}. Not redownloading"
            )
            return

        TestLGR.info(
            f"Downloaded data in {test_data_path} are from {local_date or 'an unknown date'}. "
            f"Data on https://osf.io/{osf_id} were last updated on {osf_date}. Redownloading"
        )

    _download_and_extract(osf_id, test_data_path, osf_date)


def _osf_session():
    """Create a requests session that retries transient OSF failures."""
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=OSF_RETRY))
    return session


@lru_cache(maxsize=None)
def _get_osf_updated_date(osf_id):
    """Get the date an OSF file was last updated, or None if OSF cannot be reached.

    The result is cached, so each test session queries OSF at most once per dataset.
    """
    try:
        with _osf_session() as session:
            datainfo = session.get(
                f"https://osf.io/metadata/{osf_id}/?format=datacite-json",
                timeout=OSF_TIMEOUT,
            )
            datainfo.raise_for_status()
    except requests.RequestException as err:
        TestLGR.warning(f"Cannot access metadata for https://osf.io/{osf_id}: {err}")
        return None

    metadata = json.loads(datainfo.content)
    # 'dates' is a list with all updates to the file, the last item in the list
    # is the most recent and the 'date' field in the list is the date of the last
    # update.
    return metadata["dates"][-1]["date"]


def _get_local_osf_date(test_data_path):
    """Get the OSF date recorded when the local data were downloaded, if any."""
    date_file = join(test_data_path, OSF_DATE_FILE)
    if not exists(date_file):
        return None

    with open(date_file) as fobj:
        return fobj.read().strip()


def _download_and_extract(osf_id, test_data_path, osf_date):
    """Download an OSF tarball and extract it into test_data_path.

    The archive is extracted into a temporary directory and only moved into
    place once it is complete, so an interrupted download never leaves a
    partial dataset behind.
    """
    parent_dir = dirname(abspath(test_data_path))
    makedirs(parent_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=".download-", dir=parent_dir)
    extract_dir = join(tmp_dir, "data")
    try:
        for attempt in range(1, N_DOWNLOAD_ATTEMPTS + 1):
            try:
                with _osf_session() as session, tempfile.TemporaryFile() as archive:
                    with session.get(
                        f"https://osf.io/{osf_id}/download",
                        stream=True,
                        timeout=OSF_TIMEOUT,
                    ) as req:
                        req.raise_for_status()
                        for chunk in req.iter_content(chunk_size=2**20):
                            archive.write(chunk)

                    archive.seek(0)
                    with tarfile.open(fileobj=archive, mode="r:gz") as tar:
                        tar.extractall(extract_dir)
                break
            except (requests.RequestException, tarfile.TarError, EOFError) as err:
                shutil.rmtree(extract_dir, ignore_errors=True)
                if attempt == N_DOWNLOAD_ATTEMPTS:
                    raise

                TestLGR.warning(
                    f"Attempt {attempt} to download https://osf.io/{osf_id} failed: {err}. "
                    "Retrying"
                )

        if osf_date is not None:
            with open(join(extract_dir, OSF_DATE_FILE), "w") as fobj:
                fobj.write(osf_date)

        if exists(test_data_path):
            shutil.rmtree(test_data_path)

        rename(extract_dir, test_data_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
