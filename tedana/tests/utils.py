"""Utility functions for testing tedana."""

import glob
import json
import logging
import shutil
import tarfile
from datetime import datetime
from gzip import GzipFile
from io import BytesIO
from os import listdir, makedirs
from os.path import abspath, dirname, exists, getctime, getmtime, join, sep

import requests

from tedana.workflows import tedana as tedana_cli

# Added a testing logger to output whether or not testing data were downlaoded
TestLGR = logging.getLogger("TESTING")


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
       Metadata download link would be https://osf.io/osf_id/metadata/?format=datacite-json
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
    out_path : str
        Path to directory where OSF data should be extracted
    """

    try:
        datainfo = requests.get(f"https://osf.io/{osf_id}/metadata/?format=datacite-json")
    except Exception:
        if len(listdir(test_data_path)) == 0:
            raise ConnectionError(
                f"Cannot access https://osf.io/{osf_id} and testing data " "are not yet downloaded"
            )
        else:
            TestLGR.warning(
                f"Cannot access https://osf.io/{osf_id}. "
                f"Using local copy of testing data in {test_data_path} "
                "but cannot validate that local copy is up-to-date"
            )
            return
    datainfo.raise_for_status()
    metadata = json.loads(datainfo.content)
    # 'dates' is a list with all udpates to the file, the last item in the list
    # is the most recent and the 'date' field in the list is the date of the last
    # update.
    osf_filedate = metadata["dates"][-1]["date"]

    # File the file with the most recent date for comparision with
    # the lsst updated date for the osf file
    if exists(test_data_path):
        filelist = glob.glob(f"{test_data_path}/*")
        most_recent_file = max(filelist, key=getctime)
        if exists(most_recent_file):
            local_filedate = getmtime(most_recent_file)
            local_filedate_str = str(datetime.fromtimestamp(local_filedate).date())
            local_data_exists = True
        else:
            local_data_exists = False
    else:
        local_data_exists = False
    if local_data_exists:
        if local_filedate_str == osf_filedate:
            TestLGR.info(
                f"Downloaded and up-to-date data already in {test_data_path}. Not redownloading"
            )
            return
        else:
            TestLGR.info(
                f"Downloaded data in {test_data_path} was last modified on "
                f"{local_filedate_str}. Data on https://osf.io/{osf_id} "
                f" was last updated on {osf_filedate}. Deleting and redownloading"
            )
            shutil.rmtree(test_data_path)
    req = requests.get(f"https://osf.io/{osf_id}/download")
    req.raise_for_status()
    t = tarfile.open(fileobj=GzipFile(fileobj=BytesIO(req.content)))
    makedirs(test_data_path, exist_ok=True)
    t.extractall(test_data_path)
