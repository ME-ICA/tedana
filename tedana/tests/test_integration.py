"""
Integration tests for "real" data
"""

import os
import re
import glob
import shutil
import tarfile
from io import BytesIO
from gzip import GzipFile
from pkg_resources import resource_filename

import pytest
import requests
import pandas as pd

from tedana.workflows import tedana as tedana_cli
from tedana import io


def check_integration_outputs(fname, outpath):
    """
    Checks outputs of integration tests

    Parameters
    ----------
    fname : str
        Path to file with expected outputs
    outpath : str
        Path to output directory generated from integration tests
    """

    # Gets filepaths generated by integration test
    existing = [os.path.relpath(f, outpath) for f in
                glob.glob(os.path.join(outpath, '**'), recursive=True)[1:]]

    # Checks for log file
    log_regex = ('^tedana_'
                 '[12][0-9]{3}-[0-9]{2}-[0-9]{2}T[0-9]{2}:'
                 '[0-9]{2}:[0-9]{2}.tsv$')
    logfiles = [out for out in existing if re.match(log_regex, out)]
    assert len(logfiles) == 1

    # Removes logfile from list of existing files
    existing.remove(logfiles[0])

    # Compares remaining files with those expected
    with open(fname, 'r') as f:
        tocheck = f.read().splitlines()
    assert sorted(tocheck) == sorted(existing)


def download_test_data(osf, outpath):
    """
    Downloads tar.gz data stored at `osf` and unpacks into `outpath`

    Parameters
    ----------
    osf : str
        URL to OSF file that contains data to be downloaded
    outpath : str
        Path to directory where OSF data should be extracted
    """

    req = requests.get(osf)
    req.raise_for_status()
    t = tarfile.open(fileobj=GzipFile(fileobj=BytesIO(req.content)))
    os.makedirs(outpath, exist_ok=True)
    t.extractall(outpath)


def test_integration_five_echo(skip_integration):
    """ Integration test of the full tedana workflow using five-echo test data
    """

    if skip_integration:
        pytest.skip('Skipping five-echo integration test')
    out_dir = '/tmp/data/five-echo/TED.five-echo'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # download data and run the test
    download_test_data('https://osf.io/9c42e/download',
                       os.path.dirname(out_dir))
    prepend = '/tmp/data/five-echo/p06.SBJ01_S09_Task11_e'
    suffix = '.sm.nii.gz'
    datalist = [prepend + str(i + 1) + suffix for i in range(5)]
    echo_times = [15.4, 29.7, 44.0, 58.3, 72.6]
    tedana_cli.tedana_workflow(
        data=datalist,
        tes=echo_times,
        out_dir=out_dir,
        tedpca='aic',
        fittype='curvefit',
        tedort=True,
        verbose=True)

    # Just a check on the component table pending a unit test of load_comptable
    comptable = os.path.join(out_dir, 'ica_decomposition.json')
    df = io.load_comptable(comptable)
    assert isinstance(df, pd.DataFrame)

    # Test re-running, but use the CLI
    out_dir2 = '/tmp/data/five-echo/TED.five-echo-manual'
    acc_comps = df.loc[df['classification'] == 'accepted'].index.values
    mixing = os.path.join(out_dir, 'ica_mixing.tsv')
    args = (['-d'] + datalist + ['-e'] + echo_times +
            ['--out-dir', out_dir2, '--debug', '--verbose',
            '--manacc', ','.join(acc_comps.astype(str),
            '--ctab', comptable, '--mixm', mixing])
    tedana_cli._main(args)

    # compare the generated output files
    fn = resource_filename('tedana',
                           'tests/data/nih_five_echo_outputs_verbose.txt')
    check_integration_outputs(fn, out_dir)


def test_integration_four_echo(skip_integration):
    """ Integration test of the full tedana workflow using four-echo test data
    """

    if skip_integration:
        pytest.skip('Skipping four-echo integration test')
    out_dir = '/tmp/data/four-echo/TED.four-echo'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # download data and run the test
    download_test_data('https://osf.io/gnj73/download',
                       os.path.dirname(out_dir))
    prepend = '/tmp/data/four-echo/'
    prepend += 'sub-PILOT_ses-01_task-localizerDetection_run-01_echo-'
    suffix = '_space-sbref_desc-preproc_bold+orig.HEAD'
    datalist = [prepend + str(i + 1) + suffix for i in range(4)]
    tedana_cli.tedana_workflow(
        data=datalist,
        tes=[11.8, 28.04, 44.28, 60.52],
        out_dir=out_dir,
        tedpca='kundu-stabilize',
        gscontrol=['gsr', 't1c'],
        png_cmap='bone',
        debug=True,
        verbose=True)

    # compare the generated output files
    fn = resource_filename('tedana', 'tests/data/fiu_four_echo_outputs.txt')
    check_integration_outputs(fn, out_dir)


def test_integration_three_echo(skip_integration):
    """ Integration test of the full tedana workflow using three-echo test data
    """

    if skip_integration:
        pytest.skip('Skipping three-echo integration test')
    out_dir = '/tmp/data/three-echo/TED.three-echo'
    out_dir2 = '/tmp/data/three-echo/TED.three-echo-rerun'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # download data and run the test
    download_test_data('https://osf.io/rqhfc/download',
                       os.path.dirname(out_dir))
    tedana_cli.tedana_workflow(
        data='/tmp/data/three-echo/three_echo_Cornell_zcat.nii.gz',
        tes=[14.5, 38.5, 62.5],
        out_dir=out_dir,
        low_mem=True,
        tedpca='mdl')

    # test rerunning the workflow
    tedana_cli.tedana_workflow(
        data='/tmp/data/three-echo/three_echo_Cornell_zcat.nii.gz',
        tes=[14.5, 38.5, 62.5],
        out_dir=out_dir2,
        mixm=os.path.join(out_dir, 'ica_mixing.tsv'),
        ctab=os.path.join(out_dir, 'ica_decomposition.json'),
        no_png=True)

    # compare the generated output files
    fn = resource_filename('tedana',
                           'tests/data/cornell_three_echo_outputs.txt')
    check_integration_outputs(fn, out_dir)
