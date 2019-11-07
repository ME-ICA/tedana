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
import os


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
    required.add_argument('-d', '--dir',
                          dest='tedana_dir',
                          type=str,
                          help=('Output directory of tedana, i.e. where '
                                'the mixing matrix and the component'
                                'table are located.'),
                          required=True)
    optional.add_argument('-o', '--out-dir',
                          dest='out_dir',
                          type=str,
                          help=('Output folder, i.e. where '
                                'the me.ica folder should be found.'
                                '\n Default: tedana directory'),
                          default=None)
    optional.add_argument('-a', '--anat',
                          dest='anat',
                          type=str,
                          help=('Anatomical volume, in the same worldspace as '
                                'tedana\'s outputs'),
                          default=None)
    parser._action_groups.append(optional)
    return parser


def check_gzip_and_link_volume(vol, out_vol, name=''):
    """
    Function to link a .nii.gz volume from source to destination.
    Alternatively, it copies and compress a .nii file
    from source to destination.
    It also check that the file exists.

    Parameters
    ----------
    vol: string
        File with full path to source.
    out_vol: string
        File with full path to destination.
    name: string
        Name of volume for messages

    Note
    ----
    Output will be a me.ica folder. See:
    https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/userdoc/latest/ic_classification.html#loading-a-melodic-analysis

    """

    if vol[-3:] == '.gz':
        vol = vol[:-3]

    if not os.path.exists(vol):
        vol = vol + '.gz'

        if not os.path.exists(vol):
            LGR.error('%s not found. Check the relevant folder', vol)
            sys.exit()

        LGR.info('Linking %s in new folder', name)
        os.symlink(vol, out_vol)
    else:
        LGR.info('Compressing and copying %s in new folder', name)
        with open(vol, 'rb') as f_in:
            with gzip.open(out_vol, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def create_melview_folder(tedana_dir, outdir, anat):
    """
    Function to create a melodic view folder.
    To be used for FSLeyes use.

    Parameters
    ----------
    tedana_dir: string
        Full path to the tedana output folder.
    outdir: string
        Full path to where the melodic folder should be.
    anat: string
        Full path to the anatomical file to move to the melodic folder.

    Note
    ----
    Output will be a me.ica folder. See:
    https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/userdoc/latest/ic_classification.html#loading-a-melodic-analysis

    """

    ctab = os.path.join(tedana_dir, 'comp_table_ica.txt')
    mmat = os.path.join(tedana_dir, 'meica_mix.1D')
    fvol = os.path.join(tedana_dir, 'betas_OC.nii')
    dvol = os.path.join(tedana_dir, 'dn_ts_OC.nii')
    out_fvol = os.path.join(outdir, 'filtered_func_data.ica', 'melodic_IC.nii.gz')
    out_dvol = os.path.join(outdir, 'filtered_func_data.nii.gz')
    out_mmat = os.path.join(outdir, 'filtered_func_data.ica', 'melodic_mix')
    out_ftmat = os.path.join(outdir, 'filtered_func_data.ica', 'melodic_FTmix')
    out_ctab = os.path.join(outdir, 'filtered_func_data.ica', 'melodic_class')

    if not os.path.exists(mmat):
        LGR.error('%s not found. Check tedana\'s output folder', mmat)
        sys.exit()
    elif not os.path.exists(ctab):
        LGR.error('%s not found. Check tedana\'s output folder', ctab)
        sys.exit()

    if os.path.exists(outdir):
        shutil.rmtree(outdir)
        LGR.warning('Removing %s', outdir)

    os.makedirs(os.path.join(outdir, 'filtered_func_data.ica'))
    LGR.info('The files will be stored in %s', outdir)

    check_gzip_and_link_volume(fvol, out_fvol, 'betas_OC')
    check_gzip_and_link_volume(dvol, out_dvol, 'dn_ts_OC')

    if anat:
        if anat[-7:] != '.nii.gz':
            if anat[-4:] != '.nii':
                anat = anat + '.nii'
    else:
        anat = os.path.join(tedana_dir, 't2sv.nii')

        out_anat = os.path.join(outdir, 'filtered_func_data.ica', 'mean.nii.gz')
        check_gzip_and_link_volume(anat, out_anat, 'anat')

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
                                       'd_table_score', 'kappa ratio',
                                       'd_table_score_scrub', 'rationale'], axis=1)

    comptable['true'] = ' True'
    comptable['classification'] = ' ' + comptable['classification']

    comptable.component = comptable.component + 1

    compstr = str(list(comptable.component))

    f = open(out_ctab, 'w+')
    first_line = os.path.join(outdir, 'filtered_func_data.ica')
    f.write(f'{first_line}\n')
    comptable.to_csv(f, header=False, index=False, sep=',')
    f.write('%s' % compstr)
    f.close()
    LGR.info('Folder ready. Open with \"fsleyes -ad -s melodic %s\"', outdir)


def _main(argv=None):
    options = _get_parser().parse_args(argv)

    if options.out_dir:
        outdir = os.path.join(options.out_dir, 'me.ica')
    else:
        outdir = os.path.join(options.tedana_dir, 'me.ica')

    create_melview_folder(options.tedana_dir, outdir, options.anat)


if __name__ == '__main__':
    _main()
