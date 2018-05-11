import os
import shutil
import numpy as np
import os.path as op
from scipy import stats
from tedana import (decomp, io, model, select, utils)

import logging
logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)
lgr = logging.getLogger(__name__)

"""
PROCEDURE 2 : Computes ME-PCA and ME-ICA
-Computes T2* map
-Computes PCA of concatenated ME data, then computes TE-dependence of PCs
-Computes ICA of TE-dependence PCs
-Identifies TE-dependent ICs, outputs high-\kappa (BOLD) component
   and denoised time series
-or- Computes TE-dependence of each component of a general linear model
   specified by input (includes MELODIC FastICA mixing matrix)
PROCEDURE 2a: Model fitting and component selection routines
"""


def main(data, tes, mixm=None, ctab=None, manacc=None, strict=False,
         no_gscontrol=False, kdaw=10., rdaw=1., conv=2.5e-5, ste=-1,
         combmode='t2s', dne=False, initcost='tanh', finalcost='tanh',
         stabilize=False, fout=False, filecsdata=False, label=None,
         fixed_seed=42):
    """
    Parameters
    ----------
    data : :obj:`list` of :obj:`str`
        Either a single z-concatenated file (single-entry list) or a
        list of echo-specific files, in ascending order.
    tes : :obj:`list`
        List of echo times associated with data in milliseconds.
    mixm : :obj:`str`, optional
        File containing mixing matrix. If not provided, ME-PCA and ME-ICA are
        done.
    ctab : :obj:`str`, optional
        File containing component table from which to extract pre-computed
        classifications.
    manacc : :obj:`str`, optional
        Comma separated list of manually accepted components in string form.
        Default is None.
    strict : :obj:`bool`, optional
        Ignore low-variance ambiguous components. Default is False.
    no_gzcontrol : :obj:`bool`, optional
        Control global signal using spatial approach. Default is False.
    kdaw : :obj:`float`, optional
        Dimensionality augmentation weight (Kappa). Default is 10.
        -1 for low-dimensional ICA.
    rdaw : :obj:`float`, optional
        Dimensionality augmentation weight (Rho). Default is 1.
        -1 for low-dimensional ICA.
    conv : :obj:`float`, optional
        Convergence limit. Default is 2.5e-5.
    ste : :obj:`int`, optional
        Source TEs for models. 0 for all, -1 for optimal combination.
        Default is -1.
    combmode : {'t2s', 'ste'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default), 'ste' (Poser).
    dne : :obj:`bool`, optional
        Denoise each TE dataset separately. Default is False.
    initcost : {'tanh', 'pow3', 'gaus', 'skew'}, optional
        Initial cost function for ICA. Default is 'tanh'.
    finalcost : {'tanh', 'pow3', 'gaus', 'skew'}, optional
        Final cost function. Default is 'tanh'.
    stabilize : :obj:`bool`, optional
        Stabilize convergence by reducing dimensionality, for low quality data.
        Default is False.
    fout : :obj:`bool`, optional
        Save output TE-dependence Kappa/Rho SPMs. Default is False.
    filecsdata : :obj:`bool`, optional
        Save component selection data to file. Default is False.
    label : :obj:`str` or :obj:`None`, optional
        Label for output directory. Default is None.
    fixed_seed : :obj:`int`, optional
        Seeded value for ICA, for reproducibility.
    """

    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    n_echos = len(tes)

    # coerce data to samples x echos x time array
    lgr.info('++ Loading input data: {}'.format(data))
    catd, ref_img = utils.load_data(data, n_echos=n_echos)
    n_samp, n_echos, n_vols = catd.shape

    if fout:
        fout = ref_img
    else:
        fout = None

    kdaw, rdaw = float(kdaw), float(rdaw)

    if label is not None:
        out_dir = 'TED.{0}'.format(label)
    else:
        out_dir = 'TED'
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    if mixm is not None and op.isfile(mixm):
        shutil.copyfile(mixm, op.join(out_dir, 'meica_mix.1D'))
        shutil.copyfile(mixm, op.join(out_dir, op.basename(mixm)))
    elif mixm is not None:
        raise IOError('Argument "mixm" must be an existing file.')

    if ctab is not None and op.isfile(ctab):
        shutil.copyfile(ctab, op.join(out_dir, 'comp_table.txt'))
        shutil.copyfile(ctab, op.join(out_dir, op.basename(ctab)))
    elif ctab is not None:
        raise IOError('Argument "ctab" must be an existing file.')

    os.chdir(out_dir)

    lgr.info('++ Computing Mask')
    mask, masksum = utils.make_adaptive_mask(catd, minimum=False, getsum=True)

    lgr.info('++ Computing T2* map')
    t2s, s0, t2ss, s0s, t2sG, s0G = model.t2sadmap(catd, tes, mask, masksum,
                                                   start_echo=1)

    # set a hard cap for the T2* map
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s.flatten(), 99.5,
                                      interpolation_method='lower')
    t2s[t2s > cap_t2s * 10] = cap_t2s
    utils.filewrite(t2s, op.join(out_dir, 't2sv'), ref_img)
    utils.filewrite(s0, op.join(out_dir, 's0v'), ref_img)
    utils.filewrite(t2ss, op.join(out_dir, 't2ss'), ref_img)
    utils.filewrite(s0s, op.join(out_dir, 's0vs'), ref_img)
    utils.filewrite(t2sG, op.join(out_dir, 't2svG'), ref_img)
    utils.filewrite(s0G, op.join(out_dir, 's0vG'), ref_img)

    # optimally combine data
    OCcatd = model.make_optcom(catd, t2sG, tes, mask, combmode)

    # regress out global signal unless explicitly not desired
    if not no_gscontrol:
        catd, OCcatd = model.gscontrol_raw(catd, OCcatd, n_echos, ref_img)

    if mixm is None:
        lgr.info("++ Doing ME-PCA and ME-ICA")
        n_components, dd = decomp.tedpca(catd, OCcatd, combmode, mask, t2s, t2sG,
                                         stabilize, ref_img,
                                         tes=tes, kdaw=kdaw, rdaw=rdaw, ste=ste)
        mmix_orig = decomp.tedica(n_components, dd, conv, fixed_seed, cost=initcost,
                                  final_cost=finalcost)
        np.savetxt(op.join(out_dir, '__meica_mix.1D'), mmix_orig)
        seldict, comptable, betas, mmix = model.fitmodels_direct(catd, mmix_orig,
                                                                 mask, t2s, t2sG,
                                                                 tes, combmode,
                                                                 ref_img,
                                                                 fout=fout,
                                                                 reindex=True)
        np.savetxt(op.join(out_dir, 'meica_mix.1D'), mmix)

        acc, rej, midk, empty = select.selcomps(seldict, mmix, mask, ref_img, manacc,
                                                n_echos, t2s, s0, strict_mode=strict,
                                                filecsdata=filecsdata)
    else:
        mmix_orig = np.loadtxt(op.join(out_dir, 'meica_mix.1D'))
        seldict, comptable, betas, mmix = model.fitmodels_direct(catd, mmix_orig,
                                                                 mask, t2s, t2sG,
                                                                 tes, combmode,
                                                                 ref_img,
                                                                 fout=fout)
        if ctab is None:
            acc, rej, midk, empty = select.selcomps(seldict, mmix, mask, ref_img, manacc,
                                                    n_echos, t2s, s0,
                                                    filecsdata=filecsdata,
                                                    strict_mode=strict)
        else:
            acc, rej, midk, empty = io.ctabsel(ctab)

    if len(acc) == 0:
        lgr.warning('++ No BOLD components detected!!! Please check data and results!')

    io.writeresults(OCcatd, mask, comptable, mmix, n_vols, acc, rej, midk, empty, ref_img)
    io.gscontrol_mmix(OCcatd, mmix, mask, acc, rej, midk, ref_img)
    if dne:
        io.writeresults_echoes(catd, mmix, mask, acc, rej, midk, ref_img)
