"""The io module handles most file input and output in the `tedana` workflow.

Other functions in the module help simplify writing out
data from multiple echoes or write very complex outputs.
"""
import logging
import os
import os.path as op
import json
from string import Formatter

import numpy as np
import nibabel as nib
import pandas as pd
from nilearn._utils import check_niimg
from nilearn.image import new_img_like

from tedana import utils
from tedana.stats import computefeats2, get_coeffs


LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


class OutputGenerator():
    """A class for managing tedana outputs.
    """
    def __init__(
        self,
        reference_img,
        convention="bidsv1.5.0",
        out_dir=".",
        prefix=None,
        config="auto",
    ):

        if config == "auto":
            config = op.join(utils.get_resource_path(), "outputs.json")

        config = load_json(config)
        cfg = {}
        for k, v in config:
            if convention not in v.keys():
                raise ValueError(
                    f"Convention {convention} is not one of the supported conventions "
                    f"({', '.join(v.keys())})"
                )
            cfg[k] = v[convention]
        self.config = cfg
        self.reference_img = check_niimg(reference_img)
        self.convention = convention
        self.out_dir = op.abspath(out_dir)
        self.figures_dir = op.join(out_dir, "figures")
        self.prefix = prefix + "_" if prefix is not None else ""

        if not op.isdir(self.out_dir):
            LGR.info(f"Generating output directory: {self.out_dir}")
            os.mkdir(self.out_dir)

        if not op.isdir(self.figures_dir):
            LGR.info(f"Generating figures directory: {self.figures_dir}")
            os.mkdir(self.figures_dir)

    def _determine_suffix(self, description, name):
        if description.endswith("img"):
            allowed_suffixes = [".nii", ".nii.gz"]
            preferred_suffix = ".nii.gz"
        elif description.endswith("json"):
            allowed_suffixes = [".json"]
            preferred_suffix = ".json"
        elif description.endswith("tsv"):
            allowed_suffixes = [".tsv"]
            preferred_suffix = ".tsv"

        if not any(name.endswith(suff) for suff in allowed_suffixes):
            suffix = preferred_suffix

        return suffix

    def get_name(self, description, **kwargs):
        """Generates an image file full path to simplify file output

        Parameters
        ----------
        description : str
            The description of the file. Must be a key in self.config.img_table
        kwargs : keyword arguments
            The most common is ``echo``.

        Returns
        -------
        The full path for the image name
        """
        name = self.config[self.convention][description]
        suffix = self._determine_suffix(description, name)

        name_variables = get_fields(name)
        for key, value in kwargs.items():
            assert key in name_variables

        name = name.format(**kwargs)
        name = op.join(self.out_dir, self.prefix + name + suffix)
        return name

    def save_file(self, data, description, **kwargs):
        name = self.get_name(description, **kwargs)
        if description.endswith("img"):
            self.save_img(data, name)
        elif description.endswith("json"):
            self.save_json(data, name)
        elif description.endswith("tsv"):
            self.save_tsv(data, name)

        return name

    def save_img(self, data, name):
        assert isinstance(data, np.ndarray)
        assert data.ndim in (1, 2)
        img = new_nii_like(self.reference_img, data)
        img.to_filename(name)

    def save_json(self, data, name):
        assert isinstance(data, dict)
        with open(name, "w") as fo:
            json.dump(fo, data, indent=4, sort_keys=True)

    def save_tsv(self, data, name):
        assert isinstance(data, pd.DataFrame)
        data.to_csv(name, sep="\t", line_terminator="\n", na_rep="n/a", index=False)


def get_fields(name):
    """Identify all fields in an unformatted string.

    Examples
    --------
    >>> string = "{field1}{field2}{field3}"
    >>> fields = get_fields(string)
    >>> fields
    ["field1", "field2", "field3"]
    """
    formatter = Formatter()
    fields = [temp[1] for temp in formatter.parse(name) if temp[1] is not None]
    return fields


def load_json(path: str) -> dict:
    """Loads a json file from path

    Parameters
    ----------
    path: str
        The path to the json file to load

    Returns
    -------
    A dict representation of the JSON data

    Raises
    ------
    FileNotFoundError if the file does not exist
    IsADirectoryError if the path is a directory instead of a file
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def add_decomp_prefix(comp_num, prefix, max_value):
    """
    Create component name with leading zeros matching number of components

    Parameters
    ----------
    comp_num : :obj:`int`
        Component number
    prefix : :obj:`str`
        A prefix to prepend to the component name. An underscore is
        automatically added between the prefix and the component number.
    max_value : :obj:`int`
        The maximum component number in the whole decomposition. Used to
        determine the appropriate number of leading zeros in the component
        name.

    Returns
    -------
    comp_name : :obj:`str`
        Component name in the form <prefix>_<zeros><comp_num>
    """
    n_digits = int(np.log10(max_value)) + 1
    comp_name = '{0:08d}'.format(int(comp_num))
    comp_name = '{0}_{1}'.format(prefix, comp_name[8 - n_digits:])
    return comp_name


# File Writing Functions
def write_split_ts(data, mmix, mask, comptable, generator, echo=0):
    """
    Splits `data` into denoised / noise / ignored time series and saves to disk

    Parameters
    ----------
    data : (S x T) array_like
        Input time series
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like
        Boolean mask array
    generator : :obj:`tedana.io.OutputGenerator`
        Reference image to dictate how outputs are saved to disk
    out_dir : :obj:`str`, optional
        Output directory.
    echo: :obj: `int`, optional
        Echo number to generate filenames, used by some verbose
        functions. Default 0.

    Returns
    -------
    varexpl : :obj:`float`
        Percent variance of data explained by extracted + retained components

    Notes
    -----
    This function writes out several files:

    ============================    ============================================
    Filename                        Content
    ============================    ============================================
    [prefix]Accepted_bold.nii.gz    High-Kappa time series.
    [prefix]Rejected_bold.nii.gz    Low-Kappa time series.
    [prefix]Denoised_bold.nii.gz    Denoised time series.
    ============================    ============================================
    """
    acc = comptable[comptable.classification == 'accepted'].index.values
    rej = comptable[comptable.classification == 'rejected'].index.values

    # mask and de-mean data
    mdata = data[mask]
    dmdata = mdata.T - mdata.T.mean(axis=0)

    # get variance explained by retained components
    betas = get_coeffs(dmdata.T, mmix, mask=None)
    varexpl = (1 - ((dmdata.T - betas.dot(mmix.T))**2.).sum() /
               (dmdata**2.).sum()) * 100
    LGR.info('Variance explained by ICA decomposition: {:.02f}%'.format(varexpl))

    # create component and de-noised time series and save to files
    hikts = betas[:, acc].dot(mmix.T[acc, :])
    lowkts = betas[:, rej].dot(mmix.T[rej, :])
    dnts = data[mask] - lowkts

    if len(acc) != 0:
        fout = generator.save_file(utils.unmask(hikts, mask), 'high kappa ts', echo=echo)
        LGR.info('Writing high-Kappa time series: {}'.format(fout))

    if len(rej) != 0:
        fout = generator.save_file(utils.unmask(lowkts, mask), 'low kappa ts', echo=echo)
        LGR.info('Writing low-Kappa time series: {}'.format(fout))

    fout = generator.save_file(utils.unmask(dnts, mask), 'denoised ts', echo=echo)
    LGR.info('Writing denoised time series: {}'.format(fout))
    return varexpl


def writeresults(ts, mask, comptable, mmix, n_vols, generator):
    """
    Denoises `ts` and saves all resulting files to disk

    Parameters
    ----------
    ts : (S x T) array_like
        Time series to denoise and save to disk
    mask : (S,) array_like
        Boolean mask array
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. Requires at least two columns: "component" and
        "classification".
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    n_vols : :obj:`int`
        Number of volumes in original time series
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk

    Notes
    -----
    This function writes out several files:

    =========================================    =====================================
    Filename                                     Content
    =========================================    =====================================
    desc-optcomAccepted_bold.nii.gz              High-Kappa time series.
    desc-optcomRejected_bold.nii.gz              Low-Kappa time series.
    desc-optcomDenoised_bold.nii.gz              Denoised time series.
    desc-ICA_components.nii.gz                   Spatial component maps for all
                                                 components.
    desc-ICAAccepted_components.nii.gz           Spatial component maps for accepted
                                                 components.
    desc-ICAAccepted_stat-z_components.nii.gz    Z-normalized spatial component maps
                                                 for accepted components.
    =========================================    =====================================

    See Also
    --------
    tedana.io.write_split_ts: Writes out time series files
    """
    acc = comptable[comptable.classification == 'accepted'].index.values
    write_split_ts(ts, mmix, mask, comptable, generator)

    ts_B = get_coeffs(ts, mmix, mask)
    fout = generator.save_file(ts_B, 'ICA components')
    LGR.info('Writing full ICA coefficient feature set: {}'.format(fout))

    if len(acc) != 0:
        fout = generator.save_file(ts_B[:, acc], 'ICA accepted components')
        LGR.info('Writing denoised ICA coefficient feature set: {}'.format(fout))

        # write feature versions of components
        feats = computefeats2(split_ts(ts, mmix, mask, comptable)[0], mmix[:, acc], mask)
        feats = utils.unmask(feats, mask)
        fname = generator.save_file(feats, 'z-scored ICA accepted components')
        LGR.info('Writing Z-normalized spatial component maps: {}'.format(fname))


def writeresults_echoes(catd, mmix, mask, comptable, generator):
    """
    Saves individually denoised echos to disk

    Parameters
    ----------
    catd : (S x E x T) array_like
        Input data time series
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like
        Boolean mask array
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk

    Notes
    -----
    This function writes out several files:

    =====================================    ===================================
    Filename                                 Content
    =====================================    ===================================
    echo-[echo]_desc-Accepted_bold.nii.gz    High-Kappa timeseries for echo
                                             number ``echo``.
    echo-[echo]_desc-Rejected_bold.nii.gz    Low-Kappa timeseries for echo
                                             number ``echo``.
    echo-[echo]_desc-Denoised_bold.nii.gz    Denoised timeseries for echo
                                             number ``echo``.
    =====================================    ===================================

    See Also
    --------
    tedana.io.write_split_ts: Writes out the files.
    """

    for i_echo in range(catd.shape[1]):
        LGR.info('Writing Kappa-filtered echo #{:01d} timeseries'.format(i_echo + 1))
        write_split_ts(catd[:, i_echo, :], mmix, mask, comptable, generator, echo=(i_echo + 1))


# File Loading Functions
def load_data(data, n_echos=None):
    """
    Coerces input `data` files to required 3D array output

    Parameters
    ----------
    data : (X x Y x M x T) array_like or :obj:`list` of img_like
        Input multi-echo data array, where `X` and `Y` are spatial dimensions,
        `M` is the Z-spatial dimensions with all the input echos concatenated,
        and `T` is time. A list of image-like objects (e.g., .nii) are
        accepted, as well
    n_echos : :obj:`int`, optional
        Number of echos in provided data array. Only necessary if `data` is
        array_like. Default: None

    Returns
    -------
    fdata : (S x E x T) :obj:`numpy.ndarray`
        Output data where `S` is samples, `E` is echos, and `T` is time
    ref_img : :obj:`str` or :obj:`numpy.ndarray`
        Filepath to reference image for saving output files or NIFTI-like array
    """
    if n_echos is None:
        raise ValueError('Number of echos must be specified. '
                         'Confirm that TE times are provided with the `-e` argument.')

    if isinstance(data, list):
        if len(data) == 1:  # a z-concatenated file was provided
            data = data[0]
        elif len(data) == 2:  # inviable -- need more than 2 echos
            raise ValueError('Cannot run `tedana` with only two echos: '
                             '{}'.format(data))
        else:  # individual echo files were provided (surface or volumetric)
            fdata = np.stack([utils.load_image(f) for f in data], axis=1)
            ref_img = check_niimg(data[0])
            ref_img.header.extensions = []
            return np.atleast_3d(fdata), ref_img

    img = check_niimg(data)
    (nx, ny), nz = img.shape[:2], img.shape[2] // n_echos
    fdata = utils.load_image(img.get_fdata().reshape(nx, ny, nz, n_echos, -1, order='F'))
    # create reference image
    ref_img = img.__class__(np.zeros((nx, ny, nz, 1)), affine=img.affine,
                            header=img.header, extra=img.extra)
    ref_img.header.extensions = []
    ref_img.header.set_sform(ref_img.header.get_sform(), code=1)

    return fdata, ref_img


# Helper Functions
def new_nii_like(ref_img, data, affine=None, copy_header=True):
    """
    Coerces `data` into NiftiImage format like `ref_img`

    Parameters
    ----------
    ref_img : :obj:`str` or img_like
        Reference image
    data : (S [x T]) array_like
        Data to be saved
    affine : (4 x 4) array_like, optional
        Transformation matrix to be used. Default: `ref_img.affine`
    copy_header : :obj:`bool`, optional
        Whether to copy header from `ref_img` to new image. Default: True

    Returns
    -------
    nii : :obj:`nibabel.nifti1.Nifti1Image`
        NiftiImage
    """

    ref_img = check_niimg(ref_img)
    newdata = data.reshape(ref_img.shape[:3] + data.shape[1:])
    if '.nii' not in ref_img.valid_exts:
        # this is rather ugly and may lose some information...
        nii = nib.Nifti1Image(newdata, affine=ref_img.affine,
                              header=ref_img.header)
    else:
        # nilearn's `new_img_like` is a very nice function
        nii = new_img_like(ref_img, newdata, affine=affine,
                           copy_header=copy_header)
    nii.set_data_dtype(data.dtype)

    return nii


def split_ts(data, mmix, mask, comptable):
    """
    Splits `data` time series into accepted component time series and remainder

    Parameters
    ----------
    data : (S x T) array_like
        Input data, where `S` is samples and `T` is time
    mmix : (T x C) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like
        Boolean mask array
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. Requires at least two columns: "component" and
        "classification".

    Returns
    -------
    hikts : (S x T) :obj:`numpy.ndarray`
        Time series reconstructed using only components in `acc`
    resid : (S x T) :obj:`numpy.ndarray`
        Original data with `hikts` removed
    """
    acc = comptable[comptable.classification == 'accepted'].index.values

    cbetas = get_coeffs(data - data.mean(axis=-1, keepdims=True),
                        mmix, mask)
    betas = cbetas[mask]
    if len(acc) != 0:
        hikts = utils.unmask(betas[:, acc].dot(mmix.T[acc, :]), mask)
    else:
        hikts = None

    resid = data - hikts

    return hikts, resid
