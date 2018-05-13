import logging
import numpy as np
from tedana import model, utils

lgr = logging.getLogger(__name__)


def main(options):
    """
    Estimate T2 and S0, and optimally combine data across TEs.

    Parameters
    ----------
    options
        label
        tes
        data
    """
    if options.label is not None:
        suf = '_%s' % str(options.label)
    else:
        suf = ''
    tes, data, combmode = options.tes, options.data, options.combmode

    tes = [float(te) for te in tes]
    n_echos = len(tes)

    catd = utils.load_data(data, n_echos=n_echos)
    n_samp, n_echos, n_trs = catd.shape

    ref_img = data[0] if isinstance(data, list) else data

    lgr.info('Computing adaptive mask')
    mask, masksum = utils.make_adaptive_mask(catd, minimum=False, getsum=True)
    utils.filewrite(masksum, 'masksum%s' % suf, ref_img, copy_header=False)

    lgr.info('Computing adaptive T2* map')
    t2s, s0, t2ss, s0vs, t2saf, s0vaf = model.t2sadmap(catd, tes, mask, masksum, 2)
    utils.filewrite(t2ss, 't2ss%s' % suf, ref_img, copy_header=False)
    utils.filewrite(s0vs, 's0vs%s' % suf, ref_img, copy_header=False)

    lgr.info('Computing optimal combination')
    tsoc = np.array(model.make_optcom(catd, t2s, tes, mask, combmode),
                    dtype=float)

    # Clean up numerical errors
    t2sm = t2s.copy()
    for n in (tsoc, s0, t2s, t2sm):
        np.nan_to_num(n, copy=False)

    s0[s0 < 0] = 0
    t2s[t2s < 0] = 0
    t2sm[t2sm < 0] = 0

    utils.filewrite(tsoc, 'ocv%s' % suf, ref_img, copy_header=False)
    utils.filewrite(s0, 's0v%s' % suf, ref_img, copy_header=False)
    utils.filewrite(t2s, 't2sv%s' % suf, ref_img, copy_header=False)
    utils.filewrite(t2sm, 't2svm%s' % suf, ref_img, copy_header=False)
