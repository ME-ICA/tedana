"""
Quick hack for tedana, to prepare a ready-to-read FSLeyes folder.

"""

import logging

import sys
import argparse
import shutil
import gzip

import numpy as np
import pandas as pd

import os.path as op

from os import mkdir

LGR = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_parser():
    """
    Parses command line inputs for this function

    Returns
    -------
    parser.parse_args() : argparse dict

    """
    parser = argparse.ArgumentParser()
    # Argument parser follow template provided by RalphyZ, also used by tedana.
    # https://stackoverflow.com/a/43456577
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-td', '--ted-dir',
                          dest='tedana_dir',
                          type=str,
                          help=('Output directory of tedana, i.e. where '
                                'the mixing matrix and the component'
                                'table are located.'),
                          required=True)
    optional.add_argument('-o', '--out-dir',
                          dest='out_dir',
                          type=str,
                          help=('Output folder, i.e. where'
                                'the meica.ica folder should be.'),
                          default=None)
    parser._action_groups.append(optional)
    return parser


def create_melview_folder(tedana_dir, outdir):
    """
    Function to create a melodic view folder.
    To be used for FSLeyes use.

    Parameters
    ----------
    tedana_dir: string
        Full path to the tedana output folder.
    outdir: string
        Full path to where the melodic folder should be

    Note
    ----
    Output will be a me.ica folder. See:
    https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/userdoc/latest/ic_classification.html#loading-a-melodic-analysis

    """

    ctab = op.join(tedana_dir, 'comp_table_ica.txt')
    mmat = op.join(tedana_dir, 'meica_mix.1D')
    fvol = op.join(tedana_dir, 'betas_OC.nii')
    out_fvol = op.join(outdir, 'melodic_IC.nii.gz')
    out_mmat = op.join(outdir, 'melodic_mix')
    out_ftmat = op.join(outdir, 'melodic_FTmix')
    out_ctab = op.join(outdir, 'melodic_class')

    if not op.exists(mmat):
        LGR.error('%s not found. Check tedana\'s output folder' % mmat)
        sys.exit()
    elif not op.exists(ctab):
        LGR.error('%s not found. Check tedana\'s output folder' % ctab)
        sys.exit()

    if op.exists(outdir):
        shutil.rmtree(outdir)
        LGR.warning('Removing %s' % outdir)

    mkdir(outdir)
    LGR.info('The files will be stored in %s' % outdir)

    if not op.exists(fvol):
        fvol = op.join(tedana_dir, 'betas_OC.nii.gz')

        if not op.exists(fvol):
            LGR.error('%s not found. Check tedana\'s output folder' % fvol)
            sys.exit()

        LGR.info('Copying betas_OC in me.ica folder')
        shutil.copyfile(fvol,out_fvol)
    else:
        LGR.info('Compressing and copying betas_OC in me.ica folder')
        with open(fvol, 'rb') as f_in:
            with gzip.open(out_fvol, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    LGR.info('Copying meica_mix in me.ica folder and generating melodic_FTmix')
    ts = np.genfromtxt(mmat)
    power_spectrum = np.abs(np.fft.rfft(ts.T)) ** 2

    np.savetxt(out_ftmat, power_spectrum.T, fmt='%.08e')
    np.savetxt(out_mmat, ts, fmt='%.08e')

    LGR.info('Reading comp_table_ica and extracting labels')
    comptable = pd.read_csv(ctab, sep='\t')

    comptable = comptable.drop(labels=['kappa', 'rho', 'variance explained',
                                       'normalized variance explained',
                                       'countsigFR2', 'countsigFS0',
                                       'dice_FR2', 'dice_FS0', 'countnoise',
                                       'signal-noise_t', 'signal-noise_p',
                                       'd_table_score', 'd_table_score_scrub'], axis=1)
    comptable['true'] = 'True'
    comptable.component = comptable.component + 1

    compstr = str(list(comptable.component))

    f = open(out_ctab, 'w+')
    f.write('.\n')
    comptable.to_csv(f, header=False, index=False)
    f.write('%s' % compstr)
    f.close()
    LGR.info('Folder ready. Open with \"fsleyes -ad -s melodic %s\"' % outdir)


def _main(argv=None):
    options = _get_parser().parse_args(argv)

    if options.out_dir is None:
        outdir = op.join(options.tedana_dir,'me.ica')
    else:
        outdir = op.join(options.out_dir,'me.ica')

    create_melview_folder(options.tedana_dir, outdir)


if __name__ == '__main__':
    _main()
