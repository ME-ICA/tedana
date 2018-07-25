"""
Estimate T2 and S0, and optimally combine data across TEs.
"""
import os
import os.path as op
import logging

import numpy as np
from scipy import stats

from tedana import model, utils

LGR = logging.getLogger(__name__)


def t2smap(data, tes, fitmode='all', combmode='t2s', label=None):
    """
    Estimate T2 and S0, and optimally combine data across TEs.

    Parameters
    ----------
    data : :obj:`list` of :obj:`str`
        Either a single z-concatenated file (single-entry list) or a
        list of echo-specific files, in ascending order.
    tes : :obj:`list`
        List of echo times associated with data in milliseconds.
    fitmode : {'all', 'ts'}, optional
        Monoexponential model fitting scheme.
        'all' means that the model is fit, per voxel, across all timepoints.
        'ts' means that the model is fit, per voxel and per timepoint.
        Default is 'all'.
    combmode : {'t2s', 'ste'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default), 'ste' (Poser).
    label : :obj:`str` or :obj:`None`, optional
        Label for output directory. Default is None.

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
    t2sv.nii                  Limited estimated T2* 3D map or 4D timeseries.
                              Will be a 3D map if ``fitmode`` is 'all' and a
                              4D timeseries if it is 'ts'.
    s0v.nii                   Limited S0 3D map or 4D timeseries.
    t2svG.nii                 Full T2* map/timeseries. The difference between
                              the limited and full maps is that, for voxels
                              affected by dropout where only one echo contains
                              good data, the full map uses the single echo's
                              value while the limited map has a NaN.
    s0vG.nii                  Full S0 map/timeseries.
    ts_OC.nii                 Optimally combined timeseries.
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

    LGR.info('Computing adaptive mask')
    mask, masksum = utils.make_adaptive_mask(catd, minimum=False, getsum=True)

    LGR.info('Computing adaptive T2* map')
    if fitmode == 'all':
        (t2s_limited, s0_limited,
         t2ss, s0s,
         t2s_full, s0_full) = model.fit_decay(catd, tes, mask,
                                              masksum, start_echo=1)
    else:
        (t2s_limited, s0_limited,
         t2s_full, s0_full) = model.fit_decay_ts(catd, tes, mask, masksum,
                                                 start_echo=1)

    # set a hard cap for the T2* map/timeseries
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s_limited.flatten(), 99.5,
                                      interpolation_method='lower')
    LGR.debug('Setting cap on T2* map at {:.5f}'.format(cap_t2s * 10))
    t2s_limited[t2s_limited > cap_t2s * 10] = cap_t2s

    LGR.info('Computing optimal combination')
    # optimally combine data
    OCcatd = model.make_optcom(catd, tes, mask, t2s=t2s_full,
                               combmode=combmode)

    # clean up numerical errors
    for arr in (OCcatd, s0_limited, t2s_limited):
        np.nan_to_num(arr, copy=False)

    s0_limited[s0_limited < 0] = 0
    t2s_limited[t2s_limited < 0] = 0

    utils.filewrite(t2s_limited, op.join(out_dir, 't2sv.nii'), ref_img)
    utils.filewrite(s0_limited, op.join(out_dir, 's0v.nii'), ref_img)
    utils.filewrite(t2s_full, op.join(out_dir, 't2svG.nii'), ref_img)
    utils.filewrite(s0_full, op.join(out_dir, 's0vG.nii'), ref_img)
    utils.filewrite(OCcatd, op.join(out_dir, 'ts_OC.nii'), ref_img)
