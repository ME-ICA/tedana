"""
Fit models.
"""
import logging
import os.path as op

import numpy as np
import pandas as pd
from scipy import stats
import nilearn.image as niimg
from nilearn._utils import check_niimg
from nilearn.regions import connected_regions

from tedana import (combine, io, utils)

LGR = logging.getLogger(__name__)

F_MAX = 500
Z_MAX = 8


def dependence_metrics(catd, tsoc, mmix, mask, t2s, tes, ref_img,
                       reindex=False, mmixN=None, method=None, label=None,
                       out_dir='.', verbose=False):
    """
    Fit TE-dependence and -independence models to components.

    Parameters
    ----------
    catd : (S x E x T) array_like
        Input data, where `S` is samples, `E` is echos, and `T` is time
    tsoc : (S x T) array_like
        Optimally combined data
    mmix : (T x C) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `catd`
    mask : (S [x E]) array_like
        Boolean mask array
    t2s : (S [x T]) array_like
        Limited T2* map or timeseries.
    tes : list
        List of echo times associated with `catd`, in milliseconds
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    reindex : bool, optional
        Whether to sort components in descending order by Kappa. Default: False
    mmixN : (T x C) array_like, optional
        Z-scored mixing matrix. Default: None
    method : {'kundu_v2', 'kundu_v3', None}, optional
        Decision tree to be applied to metrics. Determines which maps will be
        generated and stored in seldict. Default: None

    Returns
    -------
    comptab : (C x M) :obj:`pandas.DataFrame`
        Component metrics to be used for component selection.
    seldict : :obj:`dict` or None
        Dictionary containing component-specific metric maps to be used for
        component selection. If `method` is None, then seldict will be None as
        well.
    betas : :obj:`numpy.ndarray`
    mmix_new : :obj:`numpy.ndarray`
    """
    if not (catd.shape[0] == t2s.shape[0] == mask.shape[0]):
        raise ValueError('First dimensions (number of samples) of catd ({0}), '
                         't2s ({1}), and mask ({2}) do not '
                         'match'.format(catd.shape[0], t2s.shape[0],
                                        mask.shape[0]))
    elif catd.shape[1] != len(tes):
        raise ValueError('Second dimension of catd ({0}) does not match '
                         'number of echoes provided (tes; '
                         '{1})'.format(catd.shape[1], len(tes)))
    elif catd.shape[2] != mmix.shape[0]:
        raise ValueError('Third dimension (number of volumes) of catd ({0}) '
                         'does not match first dimension of '
                         'mmix ({1})'.format(catd.shape[2], mmix.shape[0]))
    elif t2s.ndim == 2:
        if catd.shape[2] != t2s.shape[1]:
            raise ValueError('Third dimension (number of volumes) of catd '
                             '({0}) does not match second dimension of '
                             't2s ({1})'.format(catd.shape[2], t2s.shape[1]))

    mask = t2s != 0  # Override mask because problems

    # demean optimal combination
    tsoc = tsoc[mask, :]
    tsoc_dm = tsoc - tsoc.mean(axis=-1, keepdims=True)

    # compute un-normalized weight dataset (features)
    if mmixN is None:
        mmixN = mmix
    WTS = computefeats2(utils.unmask(tsoc, mask), mmixN, mask, normalize=False)

    # compute PSC dataset - shouldn't have to refit data
    tsoc_B = get_coeffs(tsoc_dm, mmix, mask=None)
    tsoc_Babs = np.abs(tsoc_B)
    PSC = tsoc_B / tsoc.mean(axis=-1, keepdims=True) * 100

    # compute skews to determine signs based on unnormalized weights,
    # correct mmix & WTS signs based on spatial distribution tails
    signs = stats.skew(WTS, axis=0)
    signs /= np.abs(signs)
    mmix = mmix.copy()
    mmix *= signs
    WTS *= signs
    PSC *= signs
    totvar = (tsoc_B**2).sum()
    totvar_norm = (WTS**2).sum()

    # compute Betas and means over TEs for TE-dependence analysis
    betas = get_coeffs(catd, mmix, np.repeat(mask[:, np.newaxis], len(tes),
                                             axis=1))
    n_samp, n_echos, n_components = betas.shape
    n_voxels = mask.sum()
    n_data_voxels = (t2s != 0).sum()
    mu = catd.mean(axis=-1, dtype=float)
    tes = np.reshape(tes, (n_echos, 1))
    fmin, _, _ = utils.getfbounds(n_echos)

    # mask arrays
    mumask = mu[t2s != 0]
    t2smask = t2s[t2s != 0]
    betamask = betas[t2s != 0]

    # set up Xmats
    X1 = mumask.T  # Model 1
    X2 = np.tile(tes, (1, n_data_voxels)) * mumask.T / t2smask.T  # Model 2

    # tables for component selection
    kappas = np.zeros([n_components])
    rhos = np.zeros([n_components])
    varex = np.zeros([n_components])
    varex_norm = np.zeros([n_components])
    Z_maps = np.zeros([n_voxels, n_components])
    F_R2_maps = np.zeros([n_data_voxels, n_components])
    F_S0_maps = np.zeros([n_data_voxels, n_components])
    pred_R2_maps = np.zeros([n_data_voxels, n_echos, n_components])
    pred_S0_maps = np.zeros([n_data_voxels, n_echos, n_components])

    LGR.info('Fitting TE- and S0-dependent models to components')
    for i_comp in range(n_components):
        # size of B is (n_echoes, n_samples)
        B = np.atleast_3d(betamask)[:, :, i_comp].T
        alpha = (np.abs(B)**2).sum(axis=0)
        varex[i_comp] = (tsoc_B[:, i_comp]**2).sum() / totvar * 100.
        varex_norm[i_comp] = (utils.unmask(WTS, mask)[t2s != 0][:, i_comp]**2).sum() /\
            totvar_norm * 100.

        # S0 Model
        # (S,) model coefficient map
        coeffs_S0 = (B * X1).sum(axis=0) / (X1**2).sum(axis=0)
        pred_S0 = X1 * np.tile(coeffs_S0, (n_echos, 1))
        pred_S0_maps[:, :, i_comp] = pred_S0.T
        SSE_S0 = (B - pred_S0)**2
        SSE_S0 = SSE_S0.sum(axis=0)  # (S,) prediction error map
        F_S0 = (alpha - SSE_S0) * (n_echos - 1) / (SSE_S0)
        F_S0_maps[:, i_comp] = F_S0

        # R2 Model
        coeffs_R2 = (B * X2).sum(axis=0) / (X2**2).sum(axis=0)
        pred_R2 = X2 * np.tile(coeffs_R2, (n_echos, 1))
        pred_R2_maps[:, :, i_comp] = pred_R2.T
        SSE_R2 = (B - pred_R2)**2
        SSE_R2 = SSE_R2.sum(axis=0)
        F_R2 = (alpha - SSE_R2) * (n_echos - 1) / (SSE_R2)
        F_R2_maps[:, i_comp] = F_R2

        # compute weights as Z-values
        wtsZ = (WTS[:, i_comp] - WTS[:, i_comp].mean()) / WTS[:, i_comp].std()
        wtsZ[np.abs(wtsZ) > Z_MAX] = (Z_MAX * (np.abs(wtsZ) / wtsZ))[
            np.abs(wtsZ) > Z_MAX]
        Z_maps[:, i_comp] = wtsZ

        # compute Kappa and Rho
        F_S0[F_S0 > F_MAX] = F_MAX
        F_R2[F_R2 > F_MAX] = F_MAX
        norm_weights = np.abs(np.squeeze(
            utils.unmask(wtsZ, mask)[t2s != 0]**2.))
        kappas[i_comp] = np.average(F_R2, weights=norm_weights)
        rhos[i_comp] = np.average(F_S0, weights=norm_weights)

    # tabulate component values
    comptab = np.vstack([kappas, rhos, varex, varex_norm]).T
    if reindex:
        # re-index all components in descending Kappa order
        sort_idx = comptab[:, 0].argsort()[::-1]
        comptab = comptab[sort_idx, :]
        mmix_new = mmix[:, sort_idx]
        betas = betas[..., sort_idx]
        pred_R2_maps = pred_R2_maps[:, :, sort_idx]
        pred_S0_maps = pred_S0_maps[:, :, sort_idx]
        F_S0_maps = F_S0_maps[:, sort_idx]
        F_R2_maps = F_R2_maps[:, sort_idx]
        Z_maps = Z_maps[:, sort_idx]
        WTS = WTS[:, sort_idx]
        PSC = PSC[:, sort_idx]
        tsoc_B = tsoc_B[:, sort_idx]
        tsoc_Babs = tsoc_Babs[:, sort_idx]
    else:
        mmix_new = mmix

    if verbose:
        # Echo-specific weight maps for each of the ICA components.
        io.filewrite(betas, op.join(out_dir, '{0}betas_catd.nii'.format(label)),
                     ref_img)
        # Echo-specific maps of predicted values for R2 and S0 models for each
        # component.
        io.filewrite(utils.unmask(pred_R2_maps, mask),
                     op.join(out_dir, '{0}R2_pred.nii'.format(label)), ref_img)
        io.filewrite(utils.unmask(pred_S0_maps, mask),
                     op.join(out_dir, '{0}S0_pred.nii'.format(label)), ref_img)
        # Weight maps used to average metrics across voxels
        io.filewrite(utils.unmask(Z_maps ** 2., mask),
                     op.join(out_dir, '{0}metric_weights.nii'.format(label)),
                     ref_img)

    comptab = pd.DataFrame(comptab,
                           columns=['kappa', 'rho',
                                    'variance explained',
                                    'normalized variance explained'])
    comptab.index.name = 'component'

    # Generate clustering criteria for component selection
    if method in ['kundu_v2', 'kundu_v3']:
        Z_clmaps = np.zeros([n_voxels, n_components])
        F_R2_clmaps = np.zeros([n_data_voxels, n_components])
        F_S0_clmaps = np.zeros([n_data_voxels, n_components])
        Br_R2_clmaps = np.zeros([n_voxels, n_components])
        Br_S0_clmaps = np.zeros([n_voxels, n_components])

        LGR.info('Performing spatial clustering of components')
        csize = np.max([int(n_voxels * 0.0005) + 5, 20])
        LGR.debug('Using minimum cluster size: {}'.format(csize))
        for i_comp in range(n_components):
            # save out files
            out = np.zeros((n_samp, 4))
            out[:, 0] = np.squeeze(utils.unmask(PSC[:, i_comp], mask))
            out[:, 1] = np.squeeze(utils.unmask(F_R2_maps[:, i_comp],
                                                t2s != 0))
            out[:, 2] = np.squeeze(utils.unmask(F_S0_maps[:, i_comp],
                                                t2s != 0))
            out[:, 3] = np.squeeze(utils.unmask(Z_maps[:, i_comp], mask))

            ccimg = io.new_nii_like(ref_img, out)

            # Do simple clustering on F
            sel = spatclust(ccimg, min_cluster_size=csize, threshold=fmin,
                            index=[1, 2], mask=(t2s != 0))
            F_R2_clmaps[:, i_comp] = sel[:, 0]
            F_S0_clmaps[:, i_comp] = sel[:, 1]
            countsigFR2 = F_R2_clmaps[:, i_comp].sum()
            countsigFS0 = F_S0_clmaps[:, i_comp].sum()

            # Do simple clustering on Z at p<0.05
            sel = spatclust(ccimg, min_cluster_size=csize, threshold=1.95,
                            index=3, mask=mask)
            Z_clmaps[:, i_comp] = sel

            # Do simple clustering on ranked signal-change map
            spclust_input = utils.unmask(stats.rankdata(tsoc_Babs[:, i_comp]),
                                         mask)
            spclust_input = io.new_nii_like(ref_img, spclust_input)
            Br_R2_clmaps[:, i_comp] = spatclust(
                spclust_input, min_cluster_size=csize,
                threshold=(max(tsoc_Babs.shape) - countsigFR2), mask=mask)
            Br_S0_clmaps[:, i_comp] = spatclust(
                spclust_input, min_cluster_size=csize,
                threshold=(max(tsoc_Babs.shape) - countsigFS0), mask=mask)

        if method == 'kundu_v2':
            # WTS, tsoc_B, PSC, and F_S0_maps are not used by Kundu v2.5
            selvars = ['Z_maps', 'F_R2_maps',
                       'Z_clmaps', 'F_R2_clmaps', 'F_S0_clmaps',
                       'Br_R2_clmaps', 'Br_S0_clmaps']
        elif method == 'kundu_v3':
            selvars = ['WTS', 'tsoc_B', 'PSC',
                       'Z_maps', 'F_R2_maps', 'F_S0_maps',
                       'Z_clmaps', 'F_R2_clmaps', 'F_S0_clmaps',
                       'Br_R2_clmaps', 'Br_S0_clmaps']
        elif method is None:
            selvars = []
        else:
            raise ValueError('Method "{0}" not recognized.'.format(method))

        seldict = {}
        for vv in selvars:
            seldict[vv] = eval(vv)
    else:
        seldict = None

    return comptab, seldict, betas, mmix_new


def kundu_metrics(comptable, metric_maps):
    """
    Compute metrics used by Kundu v2.5 and v3.2 decision trees.

    Parameters
    ----------
    comptable : (C x M):obj:`pandas.DataFrame`
        Component metric table, where `C` is components and `M` is metrics
    metric_maps : :obj:`dict`
        A dictionary with component-specific feature maps used for
        classification. The value for each key is a (S x C) array, where `S` is
        voxels and `C` is components. Generated by `dependence_metrics`

    Returns
    -------
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metrics to be used for component selection, with new metrics
        added.
    """
    Z_maps = metric_maps['Z_maps']
    Z_clmaps = metric_maps['Z_clmaps']
    F_R2_maps = metric_maps['F_R2_maps']
    F_S0_clmaps = metric_maps['F_S0_clmaps']
    F_R2_clmaps = metric_maps['F_R2_clmaps']
    Br_S0_clmaps = metric_maps['Br_S0_clmaps']
    Br_R2_clmaps = metric_maps['Br_R2_clmaps']

    """
    Tally number of significant voxels for cluster-extent thresholded R2 and S0
    model F-statistic maps.
    """
    comptable['countsigFR2'] = F_R2_clmaps.sum(axis=0)
    comptable['countsigFS0'] = F_S0_clmaps.sum(axis=0)

    """
    Generate Dice values for R2 and S0 models
    - dice_FR2: Dice value of cluster-extent thresholded maps of R2-model betas
      and F-statistics.
    - dice_FS0: Dice value of cluster-extent thresholded maps of S0-model betas
      and F-statistics.
    """
    comptable['dice_FR2'] = np.zeros(comptable.shape[0])
    comptable['dice_FS0'] = np.zeros(comptable.shape[0])
    for i_comp in comptable.index:
        comptable.loc[i_comp, 'dice_FR2'] = utils.dice(Br_R2_clmaps[:, i_comp],
                                                       F_R2_clmaps[:, i_comp])
        comptable.loc[i_comp, 'dice_FS0'] = utils.dice(Br_S0_clmaps[:, i_comp],
                                                       F_S0_clmaps[:, i_comp])

    comptable.loc[np.isnan(comptable['dice_FR2']), 'dice_FR2'] = 0
    comptable.loc[np.isnan(comptable['dice_FS0']), 'dice_FS0'] = 0

    """
    Generate three metrics of component noise:
    - countnoise: Number of "noise" voxels (voxels highly weighted for
      component, but not from clusters)
    - signal-noise_t: T-statistic for two-sample t-test of F-statistics from
      "signal" voxels (voxels in clusters) against "noise" voxels (voxels not
      in clusters) for R2 model.
    - signal-noise_p: P-value from t-test.
    """
    comptable['countnoise'] = 0
    comptable['signal-noise_t'] = 0
    comptable['signal-noise_p'] = 0
    for i_comp in comptable.index:
        # index voxels significantly loading on component but not from clusters
        comp_noise_sel = ((np.abs(Z_maps[:, i_comp]) > 1.95) &
                          (Z_clmaps[:, i_comp] == 0))
        comptable.loc[i_comp, 'countnoise'] = np.array(
            comp_noise_sel, dtype=np.int).sum()
        # NOTE: Why only compare distributions of *unique* F-statistics?
        noise_FR2_Z = np.log10(np.unique(F_R2_maps[comp_noise_sel, i_comp]))
        signal_FR2_Z = np.log10(np.unique(
            F_R2_maps[Z_clmaps[:, i_comp] == 1, i_comp]))
        (comptable.loc[i_comp, 'signal-noise_t'],
         comptable.loc[i_comp, 'signal-noise_p']) = stats.ttest_ind(
             signal_FR2_Z, noise_FR2_Z, equal_var=False)

    comptable.loc[np.isnan(comptable['signal-noise_t']), 'signal-noise_t'] = 0
    comptable.loc[np.isnan(comptable['signal-noise_p']), 'signal-noise_p'] = 0

    """
    Assemble decision table with five metrics:
    - Kappa values ranked from largest to smallest
    - R2-model F-score map/beta map Dice scores ranked from largest to smallest
    - Signal F > Noise F t-statistics ranked from largest to smallest
    - Number of "noise" voxels (voxels highly weighted for component, but not
      from clusters) ranked from smallest to largest
    - Number of voxels with significant R2-model F-scores within clusters
      ranked from largest to smallest

    Smaller values (i.e., higher ranks) across metrics indicate more BOLD
    dependence and less noise.
    """
    d_table_rank = np.vstack([
        comptable.shape[0] - stats.rankdata(comptable['kappa']),
        comptable.shape[0] - stats.rankdata(comptable['dice_FR2']),
        comptable.shape[0] - stats.rankdata(comptable['signal-noise_t']),
        stats.rankdata(comptable['countnoise']),
        comptable.shape[0] - stats.rankdata(comptable['countsigFR2'])]).T
    comptable['d_table_score'] = d_table_rank.mean(axis=1)

    return comptable


def computefeats2(data, mmix, mask, normalize=True):
    """
    Converts `data` to component space using `mmix`

    Parameters
    ----------
    data : (S x T) array_like
        Input data
    mmix : (T [x C]) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like
        Boolean mask array
    normalize : bool, optional
        Whether to z-score output. Default: True

    Returns
    -------
    data_Z : (S x C) :obj:`numpy.ndarray`
        Data in component space
    """
    if data.ndim != 2:
        raise ValueError('Parameter data should be 2d, not {0}d'.format(data.ndim))
    elif mmix.ndim not in [2]:
        raise ValueError('Parameter mmix should be 2d, not '
                         '{0}d'.format(mmix.ndim))
    elif mask.ndim != 1:
        raise ValueError('Parameter mask should be 1d, not {0}d'.format(mask.ndim))
    elif data.shape[0] != mask.shape[0]:
        raise ValueError('First dimensions (number of samples) of data ({0}) '
                         'and mask ({1}) do not match.'.format(data.shape[0],
                                                               mask.shape[0]))
    elif data.shape[1] != mmix.shape[0]:
        raise ValueError('Second dimensions (number of volumes) of data ({0}) '
                         'and mmix ({1}) do not match.'.format(data.shape[0],
                                                               mmix.shape[0]))

    # demean masked data
    data_vn = stats.zscore(data[mask], axis=-1)

    # get betas of `data`~`mmix` and limit to range [-0.999, 0.999]
    data_R = get_coeffs(data_vn, mmix, mask=None)
    data_R[data_R < -0.999] = -0.999
    data_R[data_R > 0.999] = 0.999

    # R-to-Z transform
    data_Z = np.arctanh(data_R)
    if data_Z.ndim == 1:
        data_Z = np.atleast_2d(data_Z).T

    # normalize data
    if normalize:
        data_Zm = stats.zscore(data_Z, axis=0)
        data_Z = data_Zm + (data_Z.mean(axis=0, keepdims=True) /
                            data_Z.std(axis=0, keepdims=True))
    return data_Z


def get_coeffs(data, X, mask=None, add_const=False):
    """
    Performs least-squares fit of `X` against `data`

    Parameters
    ----------
    data : (S [x E] x T) array_like
        Array where `S` is samples, `E` is echoes, and `T` is time
    X : (T [x C]) array_like
        Array where `T` is time and `C` is predictor variables
    mask : (S [x E]) array_like
        Boolean mask array
    add_const : bool, optional
        Add intercept column to `X` before fitting. Default: False

    Returns
    -------
    betas : (S [x E] x C) :obj:`numpy.ndarray`
        Array of `S` sample betas for `C` predictors
    """
    if data.ndim not in [2, 3]:
        raise ValueError('Parameter data should be 2d or 3d, not {0}d'.format(data.ndim))
    elif X.ndim not in [2]:
        raise ValueError('Parameter X should be 2d, not {0}d'.format(X.ndim))
    elif data.shape[-1] != X.shape[0]:
        raise ValueError('Last dimension (dimension {0}) of data ({1}) does not '
                         'match first dimension of '
                         'X ({2})'.format(data.ndim, data.shape[-1], X.shape[0]))

    # mask data and flip (time x samples)
    if mask is not None:
        if mask.ndim not in [1, 2]:
            raise ValueError('Parameter data should be 1d or 2d, not {0}d'.format(mask.ndim))
        elif data.shape[0] != mask.shape[0]:
            raise ValueError('First dimensions of data ({0}) and mask ({1}) do not '
                             'match'.format(data.shape[0], mask.shape[0]))
        mdata = data[mask, :].T
    else:
        mdata = data.T

    # coerce X to >=2d
    X = np.atleast_2d(X)

    if len(X) == 1:
        X = X.T

    if add_const:  # add intercept, if specified
        X = np.column_stack([X, np.ones((len(X), 1))])

    betas = np.linalg.lstsq(X, mdata, rcond=None)[0].T
    if add_const:  # drop beta for intercept, if specified
        betas = betas[:, :-1]

    if mask is not None:
        betas = utils.unmask(betas, mask)

    return betas


def spatclust(img, min_cluster_size, threshold=None, index=None, mask=None):
    """
    Spatially clusters `img`

    Parameters
    ----------
    img : str or img_like
        Image file or object to be clustered
    min_cluster_size : int
        Minimum cluster size (in voxels)
    threshold : float, optional
        Whether to threshold `img` before clustering
    index : array_like, optional
        Whether to extract volumes from `img` for clustering
    mask : (S,) array_like, optional
        Boolean array for masking resultant data array

    Returns
    -------
    clustered : :obj:`numpy.ndarray`
        Binarized array (values are 0 or 1) of clustered (and thresholded)
        `img` data
    """

    # we need a 4D image for `niimg.iter_img`, below
    img = niimg.copy_img(check_niimg(img, atleast_4d=True))

    # temporarily set voxel sizes to 1mm isotropic so that `min_cluster_size`
    # represents the minimum number of voxels we want to be in a cluster,
    # rather than the minimum size of the desired clusters in mm^3
    if not np.all(np.abs(np.diag(img.affine)) == 1):
        img.set_sform(np.sign(img.affine))

    # grab desired volumes from provided image
    if index is not None:
        if not isinstance(index, list):
            index = [index]
        img = niimg.index_img(img, index)

    # threshold image
    if threshold is not None:
        img = niimg.threshold_img(img, float(threshold))

    clout = []
    for subbrick in niimg.iter_img(img):
        # `min_region_size` is not inclusive (as in AFNI's `3dmerge`)
        # subtract one voxel to ensure we aren't hitting this thresholding issue
        try:
            clsts = connected_regions(subbrick,
                                      min_region_size=int(min_cluster_size) - 1,
                                      smoothing_fwhm=None,
                                      extract_type='connected_components')[0]
        # if no clusters are detected we get a TypeError; create a blank 4D
        # image object as a placeholder instead
        except TypeError:
            clsts = niimg.new_img_like(subbrick,
                                       np.zeros(subbrick.shape + (1,)))
        # if multiple clusters detected, collapse into one volume
        clout += [niimg.math_img('np.sum(a, axis=-1)', a=clsts)]

    # convert back to data array and make boolean
    clustered = utils.load_image(niimg.concat_imgs(clout).get_data()) != 0

    # if mask provided, mask output
    if mask is not None:
        clustered = clustered[mask]

    return clustered
