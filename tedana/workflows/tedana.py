"""
Run the "canonical" TE-Dependent ANAlysis workflow.
"""
import os
import os.path as op
import shutil
import logging

import numpy as np
from scipy import stats
from tedana import (decomposition, model, selection, utils)

LGR = logging.getLogger(__name__)

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


def tedana(data, tes, mixm=None, ctab=None, manacc=None, strict=False,
           gscontrol=True, kdaw=10., rdaw=1., conv=2.5e-5, ste=-1,
           combmode='t2s', dne=False, cost='logcosh',
           stabilize=False, fout=False, filecsdata=False, wvpca=False,
           label=None, fixed_seed=42, debug=False, quiet=False):
    """
    Run the "canonical" TE-Dependent ANAlysis workflow.

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
    gscontrol : :obj:`bool`, optional
        Control global signal using spatial approach. Default is True.
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
    cost : {'tanh', 'pow3', 'gaus', 'skew'}, optional
        Initial cost function for ICA. Default is 'tanh'.
    stabilize : :obj:`bool`, optional
        Stabilize convergence by reducing dimensionality, for low quality data.
        Default is False.
    fout : :obj:`bool`, optional
        Save output TE-dependence Kappa/Rho SPMs. Default is False.
    filecsdata : :obj:`bool`, optional
        Save component selection data to file. Default is False.
    wvpca : :obj:`bool`, optional
        Whether or not to perform PCA on wavelet-transformed data.
        Default is False.
    label : :obj:`str` or :obj:`None`, optional
        Label for output directory. Default is None.

    Other Parameters
    ----------------
    fixed_seed : :obj:`int`, optional
        Seeded value for ICA, for reproducibility.
    debug : :obj:`bool`, optional
        Whether to run in debugging mode or not. Default is False.
    quiet : :obj:`bool`, optional
        If True, suppresses logging/printing of messages. Default is False.

    Notes
    -----
    This workflow writes out several files, which are written out to a folder
    named TED.[ref_label].[label] if ``label`` is provided and TED.[ref_label]
    if not. ``ref_label`` is determined based on the name of the first ``data``
    file.

    Files are listed below:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    t2sv.nii                  Limited estimated T2* 3D map.
                              The difference between the limited and full maps
                              is that, for voxels affected by dropout where
                              only one echo contains good data, the full map
                              uses the single echo's value while the limited
                              map has a NaN.
    s0v.nii                   Limited S0 3D map.
                              The difference between the limited and full maps
                              is that, for voxels affected by dropout where
                              only one echo contains good data, the full map
                              uses the single echo's value while the limited
                              map has a NaN.
    t2ss.nii                  ???
    s0vs.nii                  ???
    t2svG.nii                 Full T2* map/timeseries. The difference between
                              the limited and full maps is that, for voxels
                              affected by dropout where only one echo contains
                              good data, the full map uses the single echo's
                              value while the limited map has a NaN.
    s0vG.nii                  Full S0 map/timeseries.
    __meica_mix.1D            A mixing matrix
    meica_mix.1D              Another mixing matrix
    ts_OC.nii                 Optimally combined timeseries.
    betas_OC.nii              Full ICA coefficient feature set.
    betas_hik_OC.nii          Denoised ICA coefficient feature set
    feats_OC2.nii             Z-normalized spatial component maps
    comp_table.txt            Component table
    sphis_hik.nii             T1-like effect
    hik_ts_OC_T1c.nii         T1 corrected time series by regression
    dn_ts_OC_T1c.nii          ME-DN version of T1 corrected time series
    betas_hik_OC_T1c.nii      T1-GS corrected components
    meica_mix_T1c.1D          T1-GS corrected mixing matrix
    ======================    =================================================

    If ``dne`` is set to True:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    hik_ts_e[echo].nii        High-Kappa timeseries for echo number ``echo``
    midk_ts_e[echo].nii       Mid-Kappa timeseries for echo number ``echo``
    lowk_ts_e[echo].nii       Low-Kappa timeseries for echo number ``echo``
    dn_ts_e[echo].nii         Denoised timeseries for echo number ``echo``
    ======================    =================================================
    """

    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    n_echos = len(tes)

    # coerce data to samples x echos x time array
    LGR.info('Loading input data: {}'.format([f for f in data]))
    catd, ref_img = utils.load_data(data, n_echos=n_echos)
    n_samp, n_echos, n_vols = catd.shape
    LGR.debug('Resulting data shape: {}'.format(catd.shape))

    if fout:
        fout = ref_img
    else:
        fout = None

    kdaw, rdaw = float(kdaw), float(rdaw)

    try:
        ref_label = os.path.basename(ref_img).split('.')[0]
    except TypeError:
        ref_label = os.path.basename(str(data[0])).split('.')[0]

    if label is not None:
        out_dir = 'TED.{0}.{1}'.format(ref_label, label)
    else:
        out_dir = 'TED.{0}'.format(ref_label)
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        LGR.info('Creating output directory: {}'.format(out_dir))
        os.mkdir(out_dir)
    else:
        LGR.info('Using output directory: {}'.format(out_dir))

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

    LGR.info('Computing adaptive mask')
    mask, masksum = utils.make_adaptive_mask(catd, minimum=False, getsum=True)
    LGR.debug('Retaining {}/{} samples'.format(mask.sum(), n_samp))

    LGR.info('Computing T2* map')
    t2s, s0, t2ss, s0s, t2sG, s0G = model.fit_decay(catd, tes, mask, masksum,
                                                    start_echo=1)

    # set a hard cap for the T2* map
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s.flatten(), 99.5,
                                      interpolation_method='lower')
    LGR.debug('Setting cap on T2* map at {:.5f}'.format(cap_t2s * 10))
    t2s[t2s > cap_t2s * 10] = cap_t2s
    utils.filewrite(t2s, op.join(out_dir, 't2sv.nii'), ref_img)
    utils.filewrite(s0, op.join(out_dir, 's0v.nii'), ref_img)
    utils.filewrite(t2ss, op.join(out_dir, 't2ss.nii'), ref_img)
    utils.filewrite(s0s, op.join(out_dir, 's0vs.nii'), ref_img)
    utils.filewrite(t2sG, op.join(out_dir, 't2svG.nii'), ref_img)
    utils.filewrite(s0G, op.join(out_dir, 's0vG.nii'), ref_img)

    # optimally combine data
    OCcatd = model.make_optcom(catd, tes, mask, t2s=t2sG, combmode=combmode)

    # regress out global signal unless explicitly not desired
    if gscontrol:
        catd, OCcatd = model.gscontrol_raw(catd, OCcatd, n_echos, ref_img)

    if mixm is None:
        n_components, dd = decomposition.tedpca(catd, OCcatd, combmode, mask,
                                                t2s, t2sG, stabilize, ref_img,
                                                tes=tes, kdaw=kdaw, rdaw=rdaw,
                                                ste=ste, wvpca=wvpca)
        mmix_orig = decomposition.tedica(n_components, dd, conv, fixed_seed,
                                         cost=cost)
        np.savetxt(op.join(out_dir, '__meica_mix.1D'), mmix_orig)

        LGR.info('Making second component selection guess from ICA results')
        seldict, comptable, betas, mmix = model.fitmodels_direct(catd, mmix_orig,
                                                                 mask, t2s, t2sG,
                                                                 tes, combmode,
                                                                 ref_img,
                                                                 fout=fout,
                                                                 reindex=True)
        np.savetxt(op.join(out_dir, 'meica_mix.1D'), mmix)

        acc, rej, midk, empty = selection.selcomps(seldict, mmix, mask, ref_img, manacc,
                                                   n_echos, t2s, s0, strict_mode=strict,
                                                   filecsdata=filecsdata)
    else:
        LGR.info('Using supplied mixing matrix from ICA')
        mmix_orig = np.loadtxt(op.join(out_dir, 'meica_mix.1D'))
        seldict, comptable, betas, mmix = model.fitmodels_direct(catd, mmix_orig,
                                                                 mask, t2s, t2sG,
                                                                 tes, combmode,
                                                                 ref_img,
                                                                 fout=fout)
        if ctab is None:
            acc, rej, midk, empty = selection.selcomps(seldict, mmix, mask,
                                                       ref_img, manacc,
                                                       n_echos, t2s, s0,
                                                       filecsdata=filecsdata,
                                                       strict_mode=strict)
        else:
            acc, rej, midk, empty = utils.ctabsel(ctab)

    if len(acc) == 0:
        LGR.warning('No BOLD components detected! Please check data and '
                    'results!')

    utils.writeresults(OCcatd, mask, comptable, mmix, n_vols, acc, rej, midk,
                       empty, ref_img)
    utils.gscontrol_mmix(OCcatd, mmix, mask, acc, rej, midk, ref_img)
    if dne:
        utils.writeresults_echoes(catd, mmix, mask, acc, rej, midk, ref_img)
