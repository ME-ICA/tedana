"""
Functions to identify TE-dependent and TE-independent components.
"""
import os
import json
import logging
import pickle
import pkg_resources

from nilearn._utils import check_niimg
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN

from tedana import utils
from tedana.selection._utils import (getelbow_cons, getelbow_mod,
                                     getelbow_aggr, do_svm)

LGR = logging.getLogger(__name__)
RESOURCES = pkg_resources.resource_filename('tedana', 'tests/data')


def selcomps(seldict, mmix, mask, ref_img, manacc, n_echos, t2s, s0, olevel=2,
             oversion=99, filecsdata=True, savecsdiag=True, strict_mode=False):
    """
    Labels ICA components to keep or remove from denoised data

    The selection process uses pre-calculated parameters for each ICA component
    inputted into this function in `seldict` such as
    Kappa (a T2* weighting metric), Rho (an S0 weighting metric), and variance
    explained. Additonal selection metrics are calculated within this function
    and then used to classify each component into one of four groups.

    Parameters
    ----------
    seldict : :obj:`dict`
        As output from `fitmodels_direct`
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the number of volumes in the original data
    mask : (S,) array_like
        Boolean mask array
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    manacc : list
        Comma-separated list of indices of manually accepted components
    n_echos : int
        Number of echos in original data
    t2s : (S,) array_like
    s0 : (S,) array_like
    olevel : int, optional
        Default: 2
    oversion : int, optional
        Default: 99
    filecsdata: bool, optional
        Default: False
    savecsdiag: bool, optional
        Default: True
    strict_mode: bool, optional
        Default: False

    Returns
    -------
    acc : list
        Indices of accepted (BOLD) components in `mmix`
    rej : list
        Indices of rejected (non-BOLD) components in `mmix`
    midk : list
        Indices of mid-K (questionable) components in `mmix`
        These components are typically removed from the data during denoising
    ign : list
        Indices of ignored components in `mmix`
        Ignored components are considered to have too low variance to matter.
        They are not processed through the accept vs reject decision tree and
        are NOT removed during the denoising process

    Notes
    -----
    The selection algorithm used in this function is from work by prantikk
    It is from selcomps function in select_model_fft20e.py in
    version 3.2 of MEICA at:
    https://github.com/ME-ICA/me-ica/blob/b2781dd087ab9de99a2ec3925f04f02ce84f0adc/meica.libs/select_model_fft20e.py
    Many of the early publications using and evaulating the MEICA method used a
    different selection algorithm by prantikk. The final 2.5 version of that
    algorithm in the selcomps function in select_model.py at:
    https://github.com/ME-ICA/me-ica/blob/b2781dd087ab9de99a2ec3925f04f02ce84f0adc/meica.libs/select_model.py

    In both algorithms, the ICA component selection process uses multiple
    metrics that include: kappa, rho, variance explained, compent spatial
    weighting maps, noise and spatial frequency metrics, and measures of
    spatial overlap across metrics. The precise calculations may vary between
    algorithms. The most notable difference is that the v2.5 algorithm is a
    fixed decision tree where all sections were made based on whether
    combinations of metrics crossed various thresholds. In the v3.5 algorithm,
    clustering and support vector machines are also used to classify components
    based on how similar metrics in one component are similar to metrics in
    other components.
    """

    """
    handwerkerd and others are working to "hypercomment" this function to
    help everyone understand it sufficiently with the goal of eventually
    modularizing the algorithm. This is still a work-in-process with later
    sections not fully commented, some points of uncertainty are noted, and the
    summary of the full algorithm is not yet complete.

    There are sections of this code that calculate metrics that are used in
    the decision tree for the selection process and other sections that
    are part of the decision tree. Certain comments are prefaced with METRIC
    and variable names to make clear which are metrics and others are prefaced
    with SELECTION to make clear which are for applying metrics. METRICs tend
    to be summary values that contain a signal number per component.

    Note there are some variables that are calculated in one section of the code
    that are later transformed into another metric that is actually part of a
    selection criterion. This running list is an attempt to summarize
    intermediate metrics vs the metrics that are actually used in decision
    steps. For applied metrics that are made up of intermediate metrics defined
    in earlier sections of the code, the constituent metrics are noted. More
    metrics will be added to the applied metrics section as the commenting of
    this function continues.

    Intermediate Metrics:  seldict['F_S0_clmaps'] seldict['F_R2_clmaps']
        seldict['Br_clmaps_S0'] seldict['Br_clmaps_R2'] seldict['Z_maps']
        dice_tbl countnoise
        counts_FR2_Z tt_table mmix_kurt mmix_std
        spr fproj_arr_val fdist
        Rtz, Dz

    Applied Metrices:
        seldict['Rhos']
        seldict['Kappas']
        seldict['varex']
        countsigFS0
        countsigFR2
        fz (a combination of multiple z-scored metrics: tt_table,
            seldict['varex'], seldict['Kappa'], seldict['Rho'], countnoise,
            mmix_kurt, fdist)
        tt_table[:,0]
        spz (z score of spr)
        KRcut
    """

    """
    If seldict exists, save it into a pickle file called compseldata.pklbz
    that can be loaded directly into python for future analyses
    If seldict=None, load it from the pre-saved pickle file to use for the
    rest of this function
    """
    if filecsdata:
        import bz2
        if seldict is not None:
            LGR.info('Saving component selection data')
            with bz2.BZ2File('compseldata.pklbz', 'wb') as csstate_f:
                pickle.dump(seldict, csstate_f)
        else:
            try:
                with bz2.BZ2File('compseldata.pklbz', 'rb') as csstate_f:
                    seldict = pickle.load(csstate_f)
            except FileNotFoundError:
                LGR.warning('Failed to load component selection data')
                return None

    """
    List of components
    nc and ncl start out as an ordered list of the component numbers
    nc is constant throughout the function.
    ncl changes through his function as components are assigned to other
    categories (i.e. components that are classified as rejected are removed
    from ncl)
    """
    midk = []
    ign = []
    nc = np.arange(len(seldict['Kappas']))
    ncl = np.arange(len(seldict['Kappas']))

    """
    If user has specified components to accept manually, just assign those
    components to the accepted and rejected comp lists and end the function
    """
    if manacc:
        acc = sorted([int(vv) for vv in manacc.split(',')])
        midk = []
        rej = sorted(np.setdiff1d(ncl, acc))
        return acc, rej, midk, []  # Add string for ign

    """
    METRICS: countsigFS0 countsigFR2
    F_S0_clmaps & F_R2_clmaps are the thresholded & binarized clustered maps of
    significant fits for the separate S0 and R2 cross-echo models per component.
    Since the values are 0 or 1, the countsig variables are a count of the
    significant voxels per component.
    The cluster size is a function of the # of voxels in the mask.
    The cluster threshold is based on the # of echos acquired
    """
    countsigFS0 = seldict['F_S0_clmaps'].sum(0)
    countsigFR2 = seldict['F_R2_clmaps'].sum(0)
    countnoise = np.zeros(len(nc))

    """
    Make table of dice values
    METRICS: dice_tbl
    dice_FR2, dice_FS0 are calculated for each component and the concatenated
    values are in dice_tbl
    Br_clmaps_R2 and Br_clmaps_S0 are binarized clustered Z_maps.
    The volume being clustered is the rank order indices of the absolute value
    of the beta values for the fit between the optimally combined time series
    and the mixing matrix (i.e. the lowest beta value is 1 and the highest is
    the # of voxels).
    The cluster size is a function of the # of voxels in the mask.
    The cluster threshold are the voxels with beta ranks greater than
    countsigFS0 or countsigFR2 (i.e. roughly the same number of voxels will be
    in the countsig clusters as the ICA beta map clusters)
    These dice values are the Dice-Sorenson index for the Br_clmap_?? and the
    F_??_clmap.
    If handwerkerd understands this correctly, if the voxels with the above
    threshold F stats are clustered in the same voxels with the highest beta
    values, then the dice coefficient will be 1. If the thresholded F or betas
    aren't spatially clustered (i.e. the component map is less spatially smooth)
    or the clusters are in different locations (i.e. voxels with high betas
    are also noiser so they have lower F values), then the dice coefficients
    will be lower
    """
    dice_tbl = np.zeros([nc.shape[0], 2])
    for ii in ncl:
        dice_FR2 = utils.dice(utils.unmask(seldict['Br_clmaps_R2'][:, ii], mask)[t2s != 0],
                              seldict['F_R2_clmaps'][:, ii])
        dice_FS0 = utils.dice(utils.unmask(seldict['Br_clmaps_S0'][:, ii], mask)[t2s != 0],
                              seldict['F_S0_clmaps'][:, ii])
        dice_tbl[ii, :] = [dice_FR2, dice_FS0]  # step 3a here and above
    dice_tbl[np.isnan(dice_tbl)] = 0

    """
    Make table of noise gain
    METRICS: countnoise, counts_FR2_Z, tt_table
    (This is a bit confusing & is handwerkerd's attempt at making sense of this)
    seldict['Z_maps'] is the Fisher Z normalized beta fits for the optimally
    combined time series and the mixing matrix. Z_clmaps is a binarized cluster
    of Z_maps with the cluster size based on the # of voxels and the cluster
    threshold of 1.95. utils.andb is a sum of the True values in arrays so
    comp_noise_sel is true for voxels where the Z values are greater than 1.95
    but not part of a cluster of Z values that are greater than 1.95.
    Spatially unclustered voxels with high Z values could be considerd noisy.
    countnoise is the # of voxels per component where comp_noise_sel is true.

    counts_FR2_Z is the number of voxels with Z values above the threshold
    that are in clusters (signal) and the number outside of clusters (noise)

    tt_table is a bit confusing. For each component, the first index is
    some type of normalized, log10, signal/noise t statistic and the second is
    the p value for the signal/noise t statistic (for the R2 model).
    In general, these should be bigger t or have lower p values when most of
    the Z values above threshold are inside clusters.
    Because of the log10, values below 1 are negative, which is later used as
    a threshold. It doesn't seem like the p values are ever used.
    """
    tt_table = np.zeros([len(nc), 4])
    counts_FR2_Z = np.zeros([len(nc), 2])
    for ii in nc:
        comp_noise_sel = utils.andb([np.abs(seldict['Z_maps'][:, ii]) > 1.95,
                                     seldict['Z_clmaps'][:, ii] == 0]) == 2
        countnoise[ii] = np.array(comp_noise_sel, dtype=np.int).sum()
        noise_FR2_Z_mask = utils.unmask(comp_noise_sel, mask)[t2s != 0]
        noise_FR2_Z = np.log10(np.unique(seldict['F_R2_maps'][noise_FR2_Z_mask, ii]))
        signal_FR2_Z_mask = utils.unmask(seldict['Z_clmaps'][:, ii], mask)[t2s != 0] == 1
        signal_FR2_Z = np.log10(np.unique(seldict['F_R2_maps'][signal_FR2_Z_mask, ii]))
        counts_FR2_Z[ii, :] = [len(signal_FR2_Z), len(noise_FR2_Z)]
        ttest = stats.ttest_ind(signal_FR2_Z, noise_FR2_Z, equal_var=True)
        # avoid DivideByZero RuntimeWarning
        if signal_FR2_Z.size > 0 and noise_FR2_Z.size > 0:
            mwu = stats.norm.ppf(stats.mannwhitneyu(signal_FR2_Z, noise_FR2_Z)[1])
        else:
            mwu = -np.inf
        tt_table[ii, 0] = np.abs(mwu) * ttest[0] / np.abs(ttest[0])
        tt_table[ii, 1] = ttest[1]
    tt_table[np.isnan(tt_table)] = 0
    tt_table[np.isinf(tt_table[:, 0]), 0] = np.percentile(tt_table[~np.isinf(tt_table[:, 0]), 0],
                                                          98)

    """
    Time series derivative kurtosis
    METRICS: mmix_kurt and mmix_std
    Take the derivative of the time series for each component in the ICA
    mixing matrix and calculate the kurtosis & standard deviation.
    handwerkerd thinks these metrics are later used to calculate measures
    of time series spikiness and drift in the component time series.
    """
    mmix_dt = (mmix[:-1] - mmix[1:])
    mmix_kurt = stats.kurtosis(mmix_dt)
    mmix_std = np.std(mmix_dt, axis=0)

    """
    SELECTION #1 (prantikk labeled "Step 1")
    Reject anything that is obviously an artifact
    Obvious artifacts are components with Rho>Kappa or with more clustered,
    significant voxels for the S0 model than the R2 model
    """
    LGR.debug('Rejecting gross artifacts based on Rho/Kappa values and S0/R2 counts')
    rej = ncl[utils.andb([seldict['Rhos'] > seldict['Kappas'], countsigFS0 > countsigFR2]) > 0]
    ncl = np.setdiff1d(ncl, rej)

    """
    prantikk labeled "Step 2"
    Compute 3-D spatial FFT of Beta maps to detect high-spatial
    frequency artifacts

    METRIC spr, fproj_arr_val, fdist
    PSC is the mean centered beta map for each ICA component
    The FFT is sequentially calculated across each dimension of PSC & the max
    value is removed (probably the 0Hz bin). The maximum remaining frequency
    magnitude along the z dimenions is calculated leaving a 2D matrix.
    spr contains a count of the number of frequency bins in the 2D matrix where
    the frequency magnitude is greater than 4* the maximum freq in the matrix.
    spr is later z-normed across components into spz and this is actually used
    as a selection metric.
    handwerkerd interpretation: spr is bigger the more values of the fft are
    >1/4 the max. Thus, if you assume the highest mag bins are low frequency, &
    all components have roughly the same low freq power (i.e. a brain-shaped
    blob), then spr will be bigger the more high frequency bins have magnitudes
    that are more than 1/4 of the low frequency bins.

    fproj_arr_val is a flattened 1D vector of the 2D max projection fft
    of each component. This seems to be later used in an SVM to train on
    this value for rejected components to classify some remaining n_components
    as midk
    Note: fproj_arr is created here and is a ranked list of FFT values, but is
    not used anywhere in the code. Was fproj_arr_val supposed to contain this
    ranking?

    fdist isn't completely clear to handwerkerd yet but it looks like the fit of
    the fft of the spatial map to a Gaussian distribution. If so, then the
    worse the fit, the more high frequency power would be in the component
    """
    LGR.debug('Computing 3D spatial FFT of beta maps to detect high-spatial frequency artifacts')
    # spatial information is important so for NIFTI we convert back to 3D space
    if utils.get_dtype(ref_img) == 'NIFTI':
        dim1 = np.prod(check_niimg(ref_img).shape[:2])
    else:
        dim1 = mask.shape[0]
    fproj_arr = np.zeros([dim1, len(nc)])
    fproj_arr_val = np.zeros([dim1, len(nc)])
    spr = []
    fdist = []
    for ii in nc:
        # convert data back to 3D array
        if utils.get_dtype(ref_img) == 'NIFTI':
            tproj = utils.new_nii_like(ref_img, utils.unmask(seldict['PSC'],
                                                             mask)[:, ii]).get_data()
        else:
            tproj = utils.unmask(seldict['PSC'], mask)[:, ii]
        fproj = np.fft.fftshift(np.abs(np.fft.rfftn(tproj)))
        fproj_z = fproj.max(axis=-1)
        fproj[fproj == fproj.max()] = 0
        spr.append(np.array(fproj_z > fproj_z.max() / 4, dtype=np.int).sum())
        fproj_arr[:, ii] = stats.rankdata(fproj_z.flatten())
        fproj_arr_val[:, ii] = fproj_z.flatten()
        if utils.get_dtype(ref_img) == 'NIFTI':
            fprojr = np.array([fproj, fproj[:, :, ::-1]]).max(0)
            fdist.append(np.max([utils.fitgaussian(fproj.max(jj))[3:].max() for
                         jj in range(fprojr.ndim)]))
        else:
            fdist = np.load(os.path.join(RESOURCES, 'fdist.npy'))
    if type(fdist) is not np.ndarray:
        fdist = np.array(fdist)
    spr = np.array(spr)
    # import ipdb; ipdb.set_trace()

    """
    prantikk labelled Step 3
    Create feature space of component properties
    METRIC fz, spz, Rtz, Dz

    fz is matrix of multiple other metrics described above and calculated
    in this section. Most are all of these have one number per component and
    they are z-scored across components
    Attempted explanations in order:
    Tz: The z-scored t statistics of the spatial noisiness metric in tt_table
    Vz: The z-scored the natural log of the non-normalized variance explained
        of each component
    Ktz: The z-scored natural log of the Kappa values
    (the '/ 2' does not seem necessary beacuse it will be removed by z-scoring)
    KRr: The z-scored ratio of the natural log of Kappa / nat log of Rho
    (unclear why sometimes using stats.zcore and other times writing the eq out)
    cnz: The z-scored measure of a noisy voxel count where the noisy voxels are
         the voxels with large beta estimates, but aren't part of clusters
    Rz: z-scored rho values (why aren't this log scaled, like kappa in Ktz?)
    mmix_kurt: Probably a rough measure of the spikiness of each component's
        time series in the ICA mixing matrix
    fdist_z: z-score of fdist, which is probably a measure of high freq info
        in the spatial FFT of the components (with lower being more high freq?)

    NOT in fz:
    spz: Z-scored measure probably of how much high freq is in the data. Larger
        values mean more bins of the FFT have over 1/4 the power of the maximum
        bin (read about spr above for more info)
    Rtz: Z-scored natural log of the Rho values
    Dz: Z-scored Fisher Z transformed dice values of the overlap between
        clusters for the F stats and clusters of the ICA spatial beta maps with
        roughly the same number of voxels as in the clustered F maps.
        Dz saves this for the R2 model, there are also Dice coefs for the S0
        model in dice_tbl
    """
    LGR.debug('Creating feature space of component properties')
    fdist_pre = fdist.copy()
    fdist_pre[fdist > np.median(fdist) * 3] = np.median(fdist) * 3
    fdist_z = (fdist_pre - np.median(fdist_pre)) / fdist_pre.std()
    spz = (spr-spr.mean())/spr.std()
    Tz = (tt_table[:, 0] - tt_table[:, 0].mean()) / tt_table[:, 0].std()
    varex_ = np.log(seldict['varex'])
    Vz = (varex_-varex_.mean()) / varex_.std()
    Rz = (seldict['Rhos'] - seldict['Rhos'].mean()) / seldict['Rhos'].std()
    Ktz = np.log(seldict['Kappas']) / 2
    Ktz = (Ktz-Ktz.mean()) / Ktz.std()
    Rtz = np.log(seldict['Rhos']) / 2
    Rtz = (Rtz-Rtz.mean())/Rtz.std()
    KRr = stats.zscore(np.log(seldict['Kappas']) / np.log(seldict['Rhos']))
    cnz = (countnoise-countnoise.mean()) / countnoise.std()
    Dz = stats.zscore(np.arctanh(dice_tbl[:, 0] + 0.001))
    fz = np.array([Tz, Vz, Ktz, KRr, cnz, Rz, mmix_kurt, fdist_z])

    """
    METRICS Kcut, Rcut, KRcut, KRcutguesses, Khighelbowval
    Step 3: Make initial guess of where BOLD components are and use DBSCAN
    to exclude noise components and find a sample set of 'good' components
    """
    LGR.debug('Making initial guess of BOLD components')
    # The F threshold for the echo fit (based on the # of echos) for p<0.05
    #    p<0.025, and p<0.001 (Confirm this is accurate since the function
    #    contains a lookup table rather than a calculation)
    F05, F025, F01 = utils.getfbounds(n_echos)
    # epsmap is [index,level of overlap with dicemask,
    # number of high Rho components]
    epsmap = []
    Rhos_sorted = np.array(sorted(seldict['Rhos']))[::-1]
    """
    Make an initial guess as to number of good components based on
     consensus of control points across Rhos and Kappas
    For terminology later, typically getelbow _aggr > _mod > _cons
      though this might not be universally true. A more "inclusive" threshold
      has a lower kappa since that means more components are above that thresh
      and are likely to be accepted. For Rho, a more "inclusive" threshold is
      higher since that means fewer components will be rejected based on rho.
    KRcut seems weird to handwerkerd. I see that the thresholds are slightly
     shifted for kappa & rho later in the code, but why would we ever want to
     set a common threhsold reference point for both? These are two different
     elbows on two different data sets.
    """
    KRcutguesses = [getelbow_mod(seldict['Rhos']), getelbow_cons(seldict['Rhos']),
                    getelbow_aggr(seldict['Rhos']), getelbow_mod(seldict['Kappas']),
                    getelbow_cons(seldict['Kappas']), getelbow_aggr(seldict['Kappas'])]
    KRcut = np.median(KRcutguesses)
    """
    Also a bit weird to handwerkerd. This is the 75th percentile of Kappa F
    stats of the components with the 3 elbow selection criteria and the
    F states for 3 significance thresholds based on the # of echos.
    This is some type of way to get a significance criterion for a component
    fit, but it's include why this specific criterion is useful.
    """
    Khighelbowval = stats.scoreatpercentile([getelbow_mod(seldict['Kappas'], val=True),
                                             getelbow_cons(seldict['Kappas'], val=True),
                                             getelbow_aggr(seldict['Kappas'], val=True)] +
                                            list(utils.getfbounds(n_echos)),
                                            75, interpolation_method='lower')
    """
    Default to the most inclusive kappa threshold (_cons) unless:
    1. That threshold is more than twice the median of Kappa & Rho thresholds
    2. and the moderate elbow is more inclusive than a p=0.01
    handwerkerd: This actually seems like a way to avoid using the theoretically
       most liberal threshold only when there was a bad estimate and _mod is
       is more inclusive. My one concern is that it's an odd way to test that
       the _mod elbow is any better. Why not at least see if _mod < _cons?
    prantikk's orig comment for this section is:
      "only use exclusive when inclusive is extremely inclusive - double KRcut"
    """
    cond1 = getelbow_cons(seldict['Kappas']) > KRcut * 2
    cond2 = getelbow_mod(seldict['Kappas'], val=True) < F01
    if cond1 and cond2:
        Kcut = getelbow_mod(seldict['Kappas'], val=True)
    else:
        Kcut = getelbow_cons(seldict['Kappas'], val=True)
    """
    handwerkerd: The goal seems to be to maximize the rejected components
       based on the rho cut by defaulting to a lower Rcut value. Again, if
       that is the goal, why not just test if _mod < _cons?
    prantikk's orig comment for this section is:
        only use inclusive when exclusive is extremely exclusive - half KRcut
        (remember for Rho inclusive is higher, so want both Kappa and Rho
        to defaut to lower)
    """
    if getelbow_cons(seldict['Rhos']) > KRcut * 2:
        Rcut = getelbow_mod(seldict['Rhos'], val=True)
    # for above, consider something like:
    # min([getelbow_mod(Rhos,True),sorted(Rhos)[::-1][KRguess] ])
    else:
        Rcut = getelbow_cons(seldict['Rhos'], val=True)

    # Rcut should never be higher than Kcut (handwerkerd: not sure why)
    if Rcut > Kcut:
        Kcut = Rcut

    # KRelbow has a 2 for componts that are above the Kappa accept threshold
    # and below the rho reject threshold
    KRelbow = utils.andb([seldict['Kappas'] > Kcut, seldict['Rhos'] < Rcut])
    """
    Make guess of Kundu et al 2011 plus remove high frequencies,
    generally high variance, and high variance given low Kappa
    the first index of tt_table is a t static of a what handwerkerd thinks
      is a spatial noise metric. Since log10 of these values are taken the >0
      threshold means the metric is >1. tt_lim seems to be a fairly aggressive
      percentile that is then divided by 3.
    """
    tt_lim = stats.scoreatpercentile(tt_table[tt_table[:, 0] > 0, 0],
                                     75, interpolation_method='lower') / 3
    """
    KRguess is a list of components to potentially accept. it starts with a
      list of components that cross the Kcut and Rcut threshold and weren't
      previously rejected for other reasons. From that list, it removes more
      components based on several additional criteria:
      1. tt_table less than the tt_lim threshold (spatial noisiness metric)
      2. spz (a z-scored probably high spatial freq metric) >1
      3. Vz (a z-scored variance explained metric) >2
      4. If both (seems to be if a component has a relatively high variance
          the acceptance threshold for Kappa values is doubled):
         A. The variance explained is greater than half the KRcut highest
             variance component
        B. Kappa is less than twice Kcut
    """
    KRguess = np.setdiff1d(np.setdiff1d(nc[KRelbow == 2], rej),
                           np.union1d(nc[tt_table[:, 0] < tt_lim],
                           np.union1d(np.union1d(nc[spz > 1],
                                                 nc[Vz > 2]),
                                      nc[utils.andb([seldict['varex'] > 0.5 *
                                         sorted(seldict['varex'])[::-1][int(KRcut)],
                                                seldict['Kappas'] < 2*Kcut]) == 2])))
    guessmask = np.zeros(len(nc))
    guessmask[KRguess] = 1
    """
    Throw lower-risk bad components out based on 3 criteria all being true:
      1. tt_table (a spatial noisiness metric) <0
      2. A components variance explains is greater than the median variance
         explained
      3. The component index is greater than the KRcut index. Since the
          components are sorted by kappa, this is another kappa thresholding)
    """
    rejB = ncl[utils.andb([tt_table[ncl, 0] < 0,
                           seldict['varex'][ncl] > np.median(seldict['varex']), ncl > KRcut]) == 3]
    rej = np.union1d(rej, rejB)
    # adjust ncl again to only contain the remaining non-rejected components
    ncl = np.setdiff1d(ncl, rej)

    """
    This is where handwerkerd has paused in hypercommenting the function.
    """
    LGR.debug('Using DBSCAN to find optimal set of "good" BOLD components')
    for ii in range(20000):
        eps = .005 + ii * .005
        db = DBSCAN(eps=eps, min_samples=3).fit(fz.T)

        # it would be great to have descriptive names, here
        # DBSCAN found at least three non-noisy clusters
        cond1 = db.labels_.max() > 1
        # DBSCAN didn't detect more classes than the total # of components / 6
        cond2 = db.labels_.max() < len(nc) / 6
        # TODO: confirm if 0 is a special label for DBSCAN
        # my intuition here is that we're confirming DBSCAN labelled previously
        # rejected components as noise (i.e., no overlap between `rej` and
        # labelled DBSCAN components)
        cond3 = np.intersect1d(rej, nc[db.labels_ == 0]).shape[0] == 0
        # DBSCAN labelled less than half of the total components as noisy
        cond4 = np.array(db.labels_ == -1, dtype=int).sum() / float(len(nc)) < .5

        if cond1 and cond2 and cond3 and cond4:
            epsmap.append([ii, utils.dice(guessmask, db.labels_ == 0),
                           np.intersect1d(nc[db.labels_ == 0],
                           nc[seldict['Rhos'] > getelbow_mod(Rhos_sorted,
                                                             val=True)]).shape[0]])
        db = None

    epsmap = np.array(epsmap)
    LGR.debug('Found DBSCAN solutions for {}/20000 eps resolutions'.format(len(epsmap)))
    group0 = []
    dbscanfailed = False
    if len(epsmap) != 0:
        # Select index that maximizes Dice with guessmask but first
        # minimizes number of higher Rho components
        ii = int(epsmap[np.argmax(epsmap[epsmap[:, 2] == np.min(epsmap[:, 2]), 1], 0), 0])
        LGR.debug('Component selection tuning: {:.05f}'.format(epsmap[:, 1].max()))
        db = DBSCAN(eps=.005+ii*.005, min_samples=3).fit(fz.T)
        ncl = nc[db.labels_ == 0]
        ncl = np.setdiff1d(ncl, rej)
        ncl = np.setdiff1d(ncl, ncl[ncl > len(nc) - len(rej)])
        group0 = ncl.copy()
        group_n1 = nc[db.labels_ == -1]
        to_clf = np.setdiff1d(nc, np.union1d(ncl, rej))
    if len(group0) == 0 or len(group0) < len(KRguess) * .5:
        dbscanfailed = True
        LGR.debug('DBSCAN guess failed; using elbow guess method instead')
        ncl = np.setdiff1d(np.setdiff1d(nc[KRelbow == 2], rej),
                           np.union1d(nc[tt_table[:, 0] < tt_lim],
                           np.union1d(np.union1d(nc[spz > 1],
                                      nc[Vz > 2]),
                                      nc[utils.andb([seldict['varex'] > 0.5 *
                                                     sorted(seldict['varex'])[::-1][int(KRcut)],
                                                     seldict['Kappas'] < 2 * Kcut]) == 2])))
        group0 = ncl.copy()
        group_n1 = []
        to_clf = np.setdiff1d(nc, np.union1d(group0, rej))
    if len(group0) < 2 or (len(group0) < 4 and float(len(rej))/len(group0) > 3):
        LGR.warning('Extremely limited reliable BOLD signal space! '
                    'Not filtering components beyond BOLD/non-BOLD guesses.')
        midkfailed = True
        min_acc = np.array([])
        if len(group0) != 0:
            # For extremes, building in a 20% tolerance
            toacc_hi = np.setdiff1d(nc[utils.andb([fdist <= np.max(fdist[group0]),
                                                   seldict['Rhos'] < F025, Vz > -2]) == 3],
                                    np.union1d(group0, rej))
            min_acc = np.union1d(group0, toacc_hi)
            to_clf = np.setdiff1d(nc, np.union1d(min_acc, rej))
        else:
            toacc_hi = []
            min_acc = []
        diagstep_keys = ['Rejected components', 'Kappa-Rho cut point',
                         'Kappa cut point', 'Rho cut point', 'DBSCAN failed to converge',
                         'Mid-Kappa failed (limited BOLD signal)', 'Kappa-Rho guess',
                         'min_acc', 'toacc_hi']
        diagstep_vals = [list(rej), KRcut, Kcut, Rcut, dbscanfailed,
                         midkfailed, list(KRguess), list(min_acc), list(toacc_hi)]
        with open('csstepdata.json', 'w') as ofh:
            json.dump(dict(zip(diagstep_keys, diagstep_vals)), ofh,
                      indent=4, sort_keys=True, default=str)
        return list(sorted(min_acc)), list(sorted(rej)), [], list(sorted(to_clf))

    # Find additional components to reject based on Dice - doing this here
    # since Dice is a little unstable, need to reference group0
    rej_supp = []
    dice_rej = False
    if not dbscanfailed and len(rej) + len(group0) < 0.75 * len(nc):
        dice_rej = True
        rej_supp = np.setdiff1d(np.setdiff1d(np.union1d(rej,
                                                        nc[dice_tbl[nc, 0] <= dice_tbl[nc, 1]]),
                                             group0), group_n1)
        rej = np.union1d(rej, rej_supp)

    # Temporal features
    # larger is worse - spike
    mmix_kurt_z = (mmix_kurt-mmix_kurt[group0].mean()) / mmix_kurt[group0].std()
    # smaller is worse - drift
    mmix_std_z = -1 * ((mmix_std-mmix_std[group0].mean()) / mmix_std[group0].std())
    mmix_kurt_z_max = np.max([mmix_kurt_z, mmix_std_z], 0)

    """
    Step 2: Classifiy midk and ignore using separte SVMs for
    different variance regimes
    # To render hyperplane:
    min_x = np.min(spz2);max_x=np.max(spz2)
    # plotting separating hyperplane
        ww = clf_.coef_[0]
        aa = -ww[0] / ww[1]
        # make sure the next line is long enough
        xx = np.linspace(min_x - 2, max_x + 2)
        yy = aa * xx - (clf_.intercept_[0]) / ww[1]
        plt.plot(xx, yy, '-')
    """
    LGR.debug('Attempting to classify midk components')
    # Tried getting rid of accepting based on SVM altogether,
    # now using only rejecting
    toacc_hi = np.setdiff1d(nc[utils.andb([fdist <= np.max(fdist[group0]),
                               seldict['Rhos'] < F025, Vz > -2]) == 3],
                            np.union1d(group0, rej))
    toacc_lo = np.intersect1d(to_clf,
                              nc[utils.andb([spz < 1, Rz < 0, mmix_kurt_z_max < 5,
                                             Dz > -1, Tz > -1, Vz < 0, seldict['Kappas'] >= F025,
                                             fdist < 3 * np.percentile(fdist[group0], 98)]) == 8])
    midk_clf, clf_ = do_svm(fproj_arr_val[:, np.union1d(group0, rej)].T,
                            [0] * len(group0) + [1] * len(rej),
                            fproj_arr_val[:, to_clf].T,
                            svmtype=2)
    midk = np.setdiff1d(to_clf[utils.andb([midk_clf == 1, seldict['varex'][to_clf] >
                                           np.median(seldict['varex'][group0])]) == 2],
                        np.union1d(toacc_hi, toacc_lo))
    # only use SVM to augment toacc_hi only if toacc_hi isn't already
    # conflicting with SVM choice
    if len(np.intersect1d(to_clf[utils.andb([midk_clf == 1,
                                             Vz[to_clf] > 0]) == 2], toacc_hi)) == 0:
        svm_acc_fail = True
        toacc_hi = np.union1d(toacc_hi, to_clf[midk_clf == 0])
    else:
        svm_acc_fail = False

    """
    Step 3: Compute variance associated with low T2* areas
    (e.g. draining veins and low T2* areas)
    # To write out veinmask
    veinout = np.zeros(t2s.shape)
    veinout[t2s!=0] = veinmaskf
    utils.filewrite(veinout, 'veinmaskf', ref_img)
    veinBout = utils.unmask(veinmaskB, mask)
    utils.filewrite(veinBout, 'veins50', ref_img)
    """
    LGR.debug('Computing variance associated with low T2* areas (e.g., draining veins)')
    tsoc_B_Zcl = np.zeros(seldict['tsoc_B'].shape)
    tsoc_B_Zcl[seldict['Z_clmaps'] != 0] = np.abs(seldict['tsoc_B'])[seldict['Z_clmaps'] != 0]
    sig_B = [stats.scoreatpercentile(tsoc_B_Zcl[tsoc_B_Zcl[:, ii] != 0, ii], 25)
             if len(tsoc_B_Zcl[tsoc_B_Zcl[:, ii] != 0, ii]) != 0
             else 0 for ii in nc]
    sig_B = np.abs(seldict['tsoc_B']) > np.tile(sig_B, [seldict['tsoc_B'].shape[0], 1])

    veinmask = utils.andb([t2s < stats.scoreatpercentile(t2s[t2s != 0], 15,
                                                         interpolation_method='lower'),
                           t2s != 0]) == 2
    veinmaskf = veinmask[mask]
    veinR = np.array(sig_B[veinmaskf].sum(0),
                     dtype=float) / sig_B[~veinmaskf].sum(0)
    veinR[np.isnan(veinR)] = 0

    veinc = np.union1d(rej, midk)
    rej_veinRZ = ((veinR-veinR[veinc].mean())/veinR[veinc].std())[veinc]
    rej_veinRZ[rej_veinRZ < 0] = 0
    rej_veinRZ[countsigFR2[veinc] > np.array(veinmaskf, dtype=int).sum()] = 0
    t2s_lim = [stats.scoreatpercentile(t2s[t2s != 0], 50,
                                       interpolation_method='lower'),
               stats.scoreatpercentile(t2s[t2s != 0], 80,
                                       interpolation_method='lower') / 2]
    phys_var_zs = []
    for t2sl_i in range(len(t2s_lim)):
        t2sl = t2s_lim[t2sl_i]
        veinW = sig_B[:, veinc]*np.tile(rej_veinRZ, [sig_B.shape[0], 1])
        veincand = utils.unmask(utils.andb([s0[t2s != 0] < np.median(s0[t2s != 0]),
                                t2s[t2s != 0] < t2sl]) >= 1,
                                t2s != 0)[mask]
        veinW[~veincand] = 0
        invein = veinW.sum(axis=1)[(utils.unmask(veinmaskf, mask) *
                                    utils.unmask(veinW.sum(axis=1) > 1, mask))[mask]]
        minW = 10 * (np.log10(invein).mean()) - 1 * 10**(np.log10(invein).std())
        veinmaskB = veinW.sum(axis=1) > minW
        tsoc_Bp = seldict['tsoc_B'].copy()
        tsoc_Bp[tsoc_Bp < 0] = 0
        vvex = np.array([(tsoc_Bp[veinmaskB, ii]**2.).sum() /
                         (tsoc_Bp[:, ii]**2.).sum() for ii in nc])
        group0_res = np.intersect1d(KRguess, group0)
        phys_var_zs.append((vvex - vvex[group0_res].mean()) / vvex[group0_res].std())
        veinBout = utils.unmask(veinmaskB, mask)
        utils.filewrite(veinBout.astype(float), 'veins_l%i' % t2sl_i, ref_img)

    # Mask to sample veins
    phys_var_z = np.array(phys_var_zs).max(0)
    Vz2 = (varex_ - varex_[group0].mean())/varex_[group0].std()

    """
    Step 4: Learn joint TE-dependence spatial and temporal models to move
    remaining artifacts to ignore class
    """
    LGR.debug('Learning joint TE-dependence spatial/temporal models to ignore remaining artifacts')

    to_ign = []

    minK_ign = np.max([F05, getelbow_cons(seldict['Kappas'], val=True)])
    newcest = len(group0) + len(toacc_hi[seldict['Kappas'][toacc_hi] > minK_ign])
    phys_art = np.setdiff1d(nc[utils.andb([phys_var_z > 3.5,
                                           seldict['Kappas'] < minK_ign]) == 2], group0)
    rank_diff = stats.rankdata(phys_var_z) - stats.rankdata(seldict['Kappas'])
    phys_art = np.union1d(np.setdiff1d(nc[utils.andb([phys_var_z > 2, rank_diff > newcest / 2,
                                                      Vz2 > -1]) == 3],
                                       group0), phys_art)
    # Want to replace field_art with an acf/SVM based approach
    # instead of a kurtosis/filter one
    field_art = np.setdiff1d(nc[utils.andb([mmix_kurt_z_max > 5,
                                            seldict['Kappas'] < minK_ign]) == 2], group0)
    field_art = np.union1d(np.setdiff1d(nc[utils.andb([mmix_kurt_z_max > 2,
                                           (stats.rankdata(mmix_kurt_z_max) -
                                            stats.rankdata(seldict['Kappas'])) > newcest / 2,
                                           Vz2 > 1, seldict['Kappas'] < F01]) == 4],
                                        group0), field_art)
    field_art = np.union1d(np.setdiff1d(nc[utils.andb([mmix_kurt_z_max > 3,
                                                       Vz2 > 3, seldict['Rhos'] >
                                                       np.percentile(seldict['Rhos'][group0],
                                                                     75)]) == 3],
                                        group0), field_art)
    field_art = np.union1d(np.setdiff1d(nc[utils.andb([mmix_kurt_z_max > 5, Vz2 > 5]) == 2],
                                        group0), field_art)
    misc_art = np.setdiff1d(nc[utils.andb([(stats.rankdata(Vz) -
                                            stats.rankdata(Ktz)) > newcest / 2,
                            seldict['Kappas'] < Khighelbowval]) == 2], group0)
    ign_cand = np.unique(list(field_art)+list(phys_art)+list(misc_art))
    midkrej = np.union1d(midk, rej)
    to_ign = np.setdiff1d(list(ign_cand), midkrej)
    toacc = np.union1d(toacc_hi, toacc_lo)
    ncl = np.setdiff1d(np.union1d(ncl, toacc), np.union1d(to_ign, midkrej))
    ign = np.setdiff1d(nc, list(ncl) + list(midk) + list(rej))
    orphan = np.setdiff1d(nc, list(ncl) + list(to_ign) + list(midk) + list(rej))

    # Last ditch effort to save some transient components
    if not strict_mode:
        Vz3 = (varex_ - varex_[ncl].mean())/varex_[ncl].std()
        ncl = np.union1d(ncl, np.intersect1d(orphan,
                                             nc[utils.andb([seldict['Kappas'] > F05,
                                                            seldict['Rhos'] < F025,
                                                            seldict['Kappas'] > seldict['Rhos'],
                                                            Vz3 <= -1,
                                                            Vz3 > -3,
                                                            mmix_kurt_z_max < 2.5]) == 6]))
        ign = np.setdiff1d(nc, list(ncl)+list(midk)+list(rej))
        orphan = np.setdiff1d(nc, list(ncl) + list(to_ign) + list(midk) + list(rej))

    if savecsdiag:
        diagstep_keys = ['Rejected components', 'Kappa-Rho cut point', 'Kappa cut',
                         'Rho cut', 'DBSCAN failed to converge', 'Kappa-Rho guess',
                         'Dice rejected', 'rej_supp', 'to_clf',
                         'Mid-kappa components', 'svm_acc_fail', 'toacc_hi', 'toacc_lo',
                         'Field artifacts', 'Physiological artifacts',
                         'Miscellaneous artifacts', 'ncl', 'Ignored components']
        diagstep_vals = [list(rej), KRcut.item(), Kcut.item(), Rcut.item(),
                         dbscanfailed, list(KRguess), dice_rej,
                         list(rej_supp), list(to_clf), list(midk),
                         svm_acc_fail, list(toacc_hi), list(toacc_lo),
                         list(field_art), list(phys_art),
                         list(misc_art), list(ncl), list(ign)]

        with open('csstepdata.json', 'w') as ofh:
            json.dump(dict(zip(diagstep_keys, diagstep_vals)), ofh,
                      indent=4, sort_keys=True, default=str)
        allfz = np.array([Tz, Vz, Ktz, KRr, cnz, Rz, mmix_kurt, fdist_z])
        np.savetxt('csdata.txt', allfz)

    return list(sorted(ncl)), list(sorted(rej)), list(sorted(midk)), list(sorted(ign))
