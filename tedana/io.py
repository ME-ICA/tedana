"""Handle most file input and output in the `tedana` workflow.

Other functions in the module help write outputs which require multiple
data sources, assist in writing per-echo verbose outputs, or act as helper
functions for any of the above.
"""

import json
import logging
import os
import os.path as op
from copy import deepcopy
from string import Formatter
from typing import List

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn._utils import check_niimg
from nilearn.image import new_img_like

from tedana import utils
from tedana.stats import computefeats2, get_coeffs

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")

ALLOWED_COMPONENT_DELIMITERS = (
    "\t",
    "\n",
    " ",
    ",",
)


class CustomEncoder(json.JSONEncoder):
    """Class for converting some types because of JSON serialization and numpy incompatibilities.

    See here: https://stackoverflow.com/q/50916422/2589328
    """

    def default(self, obj):
        """Return the default encoder for CustomEncoder."""
        # int64 non-serializable but is a numpy output
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)

        # containers that are not serializable
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)

        return super().default(obj)


class OutputGenerator:
    """A class for managing tedana outputs.

    Parameters
    ----------
    reference_img : img_like
        The reference image which defines affine, shape, etc. of output images.
    convention : {"bidsv1.5.0", "orig", or other str}, optional
        Default is "bidsv1.5.0". Must correspond to a key in ``config``.
    out_dir : str, optional
        Output directory. Default is current working directory (".").
    prefix : None or str, optional
        Prefix to prepend to output filenames. Default is None, which means no prefix will be used.
    config : str, optional
        Path to configuration json file, which determines appropriate filenames based on file
        descriptions. Default is "auto", which uses tedana's default configuration file.
    make_figures : bool, optional
        Whether or not to actually make a figures directory
    overwrite : bool, optional
        Whether to force overwrites of data. Default False.

    Attributes
    ----------
    config : dict
        File naming configuration information.
    reference_img : img_like
        The reference image which defines affine, shape, etc. of output images.
    convention : str
        The naming convention for output files.
    out_dir : str
        Directory in which outputs will be saved.
    figures_dir : str
        Directory in which figures will be saved.
        This will correspond to a "figures" subfolder of ``out_dir``.
    prefix : str
        Prefix to prepend to output filenames.
    overwrite : bool
        Whether to force file overwrites.
    verbose : bool
        Whether or not to generate verbose output.
    registry : dict
        A registry of all files saved
    """

    def __init__(
        self,
        reference_img,
        convention="bidsv1.5.0",
        out_dir=".",
        prefix="",
        config="auto",
        make_figures=True,
        overwrite=False,
        verbose=False,
        old_registry=None,
    ):
        if config == "auto":
            config = op.join(utils.get_resource_path(), "config", "outputs.json")

        if convention == "bids":
            # modify to update default bids convention number
            convention = "bidsv1.5.0"

        config = load_json(config)

        cfg = {}
        for k, v in config.items():
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
        self.prefix = _infer_prefix(prefix)
        self.overwrite = overwrite
        self.verbose = verbose
        self.registry = {}
        if old_registry:
            root = old_registry["root"]
            rel_root = op.relpath(root, start=self.out_dir)
            del old_registry["root"]
            for k, v in old_registry.items():
                if isinstance(v, list):
                    self.registry[k] = [op.join(rel_root, vv) for vv in v]
                else:
                    self.registry[k] = op.join(rel_root, v)

        if not op.isdir(self.out_dir):
            LGR.info(f"Generating output directory: {self.out_dir}")
            os.mkdir(self.out_dir)

        if not op.isdir(self.figures_dir) and make_figures:
            LGR.info(f"Generating figures directory: {self.figures_dir}")
            os.mkdir(self.figures_dir)

    def _determine_extension(self, description, name):
        """Infer the extension for a file based on its description.

        Parameters
        ----------
        description : str
            The description of the file. Corresponds to a key in ``self.config``.
        name : str
            Filename corresponding to the description within ``self.config``.

        Returns
        -------
        extension : str
            File extension for the filename.
        """
        if description.endswith("img"):
            allowed_extensions = [".nii", ".nii.gz"]
            preferred_extension = ".nii.gz"
        elif description.endswith("json"):
            allowed_extensions = [".json"]
            preferred_extension = ".json"
        elif description.endswith("tsv"):
            allowed_extensions = [".tsv"]
            preferred_extension = ".tsv"

        if not any(name.endswith(ext) for ext in allowed_extensions):
            extension = preferred_extension
        else:
            extension = ""

        return extension

    def register_input(self, names):
        """Register input filenames.

        Parameters
        ----------
        names : list[str]
            The list of filenames being input as multi-echo volumes.
        """
        self.registry["input img"] = [op.relpath(name, start=self.out_dir) for name in names]

    def get_name(self, description, **kwargs):
        """Generate a file full path to simplify file output.

        Parameters
        ----------
        description : str
            The description of the file. Must be a key in ``self.config``.
        kwargs : keyword arguments
            Additional arguments used to format the base filename string.
            The most common is ``echo``.

        Returns
        -------
        name : str
            The full path for the filename.

        Notes
        -----
        This function uses kwargs to allow us to match named format
        specifiers in a configuration with a variable passed to this
        function. get_fields simplifies this process by creating a set of
        name variables based on the configuration which we expect to match
        a passed variable name, and then we fill in the value.
        """
        name = self.config[description]
        extension = self._determine_extension(description, name)

        name_variables = get_fields(name)
        for key, value in kwargs.items():
            if key not in name_variables:
                raise ValueError(
                    f"Argument {key} passed but has no match in format "
                    f"string. Available format variables: "
                    f"{name_variables} from {kwargs} and {name}."
                )

        name = name.format(**kwargs)
        name = op.join(self.out_dir, self.prefix + name + extension)
        return name

    def save_file(self, data, description, **kwargs):
        """Save data to a filename determined by the file's description and config info.

        Parameters
        ----------
        data : dict or img_like or pandas.DataFrame
            Data to save to file.
        description : str
            Description of the data, used to determine the appropriate filename from
            ``self.config``.

        Returns
        -------
        name : str
            The full file path of the saved file.
        """
        name = self.get_name(description, **kwargs)
        if op.exists(name) and not self.overwrite:
            raise RuntimeError(
                f"File {name} already exists. In order to allow overwrite "
                "please use the --overwrite option in the command line or the "
                "overwrite parameter in the Python API."
            )

        if description.endswith("img"):
            self.save_img(data, name)
        elif description.endswith("json"):
            prepped = prep_data_for_json(data)
            self.save_json(prepped, name)
        elif description.endswith("tsv"):
            self.save_tsv(data, name)
        else:
            raise ValueError(f"Unsupported file {description}")

        self.registry[description] = op.basename(name)

        return name

    def save_img(self, data, name):
        """Save image data to a nifti file.

        Parameters
        ----------
        data : img_like
            Data to save to a file.
        name : str
            Full file path for output file.

        Notes
        -----
        Will coerce 64-bit float and int arrays into 32-bit arrays.
        """
        data_type = type(data)
        if isinstance(data, nib.nifti1.Nifti1Image):
            data.to_filename(name)
            return
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Data supplied must of type np.ndarray, not {data_type}.")

        if data.ndim not in (1, 2):
            raise TypeError(f"Data must have number of dimensions in (1, 2), not {data.ndim}")

        # Coerce data to be 32-bit max in the cases of float64, int64
        # Note that int64 niftis cannot be read by mricroGL, AFNI
        vox_type = data.dtype
        if vox_type == np.int64:
            data = np.int32(data)
        elif vox_type == np.float64:
            data = np.float32(data)

        # Make new img and save
        img = new_nii_like(self.reference_img, data)
        img.to_filename(name)

    def save_json(self, data, name):
        """Save dictionary data to a json file.

        Parameters
        ----------
        data : dict
            Data to save to a file.
        name : str
            Full file path for output file.
        """
        data_type = type(data)
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, not type {data_type}.")

        with open(name, "w") as fo:
            json.dump(data, fo, indent=4, sort_keys=True, cls=CustomEncoder)

    def save_tsv(self, data, name):
        """Save DataFrame to a tsv file.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to save to a file.
        name : str
            Full file path for output file.
        """
        data_type = type(data)
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be pd.Data, not type {data_type}.")

        # Replace blanks with numpy NaN
        deblanked = data.replace("", np.nan)
        deblanked.to_csv(name, sep="\t", lineterminator="\n", na_rep="n/a", index=False)

    def save_self(self):
        """Save the registry to a json file.

        Returns
        -------
        fname
            Full file path for output file.
        """
        fname = self.save_file(self.registry, "registry json")
        return fname


class InputHarvester:
    """Class for turning a registry file into a lookup table to get previous data."""

    loaders = {
        "json": lambda f: load_json(f),
        "tsv": lambda f: pd.read_csv(f, delimiter="\t"),
        "img": lambda f: nib.load(f),
    }

    def __init__(self, path):
        self._full_path = op.abspath(path)
        self._base_dir = op.dirname(self._full_path)
        self._registry = load_json(self._full_path)

    def get_file_path(self, description):
        """Get file path.

        Parameters
        ----------
        description : str
            Description of the file to get the path for.
        """
        if description in self._registry.keys():
            return op.join(self._base_dir, self._registry[description])
        else:
            return None

    def get_file_contents(self, description):
        """Get file contents.

        Notes
        -----
        Since we restrict to just these three types, this function should always return.
        If more types are added, the loaders dict will need to be updated with an appropriate
        loader.
        """
        for ftype, loader in InputHarvester.loaders.items():
            if ftype in description:
                return loader(self.get_file_path(description))

    @property
    def registry(self):
        """The underlying file registry, including the root directory."""
        d = self._registry
        d["root"] = self._base_dir
        return d


def versiontuple(v):
    """Convert a version string into a tuple of ints.

    Parameters
    ----------
    v : str
        Version string to convert.

    Returns
    -------
    tuple
        Tuple of ints corresponding to the version string.
    """
    return tuple(map(int, (v.split("."))))


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
    """Load a json file from path.

    Parameters
    ----------
    path : str
        The path to the json file to load

    Returns
    -------
    data : dict
        A dictionary representation of the JSON data.

    Raises
    ------
    FileNotFoundError if the file does not exist
    IsADirectoryError if the path is a directory instead of a file
    """
    with open(path) as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError:
            raise ValueError(f"File {path} is not a valid JSON.")
    return data


def add_decomp_prefix(comp_num, prefix, max_value):
    """Create component name with leading zeros matching number of components.

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
    comp_name = f"{int(comp_num):08d}"
    comp_name = f"{prefix}_{comp_name[8 - n_digits :]}"
    return comp_name


def denoise_ts(data, mmix, mask, comptable):
    """Apply component classifications to data for denoising.

    Parameters
    ----------
    data : (S x T) array_like
        Input time series
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like
        Boolean mask array
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. Requires at least one column: "classification".

    Returns
    -------
    dnts : (S x T) array_like
        Denoised data (i.e., data with rejected components removed).
    hikts : (S x T) array_like
        High-Kappa data (i.e., data composed only of accepted components).
    lowkts : (S x T) array_like
        Low-Kappa data (i.e., data composed only of rejected components).
    """
    acc = comptable[comptable.classification == "accepted"].index.values
    rej = comptable[comptable.classification == "rejected"].index.values

    # mask and de-mean data
    mdata = data[mask]
    dmdata = mdata.T - mdata.T.mean(axis=0)

    # get variance explained by retained components
    betas = get_coeffs(dmdata.T, mmix, mask=None)
    varexpl = (1 - ((dmdata.T - betas.dot(mmix.T)) ** 2.0).sum() / (dmdata**2.0).sum()) * 100
    LGR.info(f"Variance explained by decomposition: {varexpl:.02f}%")

    # create component-based data
    hikts = utils.unmask(betas[:, acc].dot(mmix.T[acc, :]), mask)
    lowkts = utils.unmask(betas[:, rej].dot(mmix.T[rej, :]), mask)
    dnts = utils.unmask(data[mask] - lowkts[mask], mask)
    return dnts, hikts, lowkts


# File Writing Functions
def write_split_ts(data, mmix, mask, comptable, io_generator, echo=0):
    """Split `data` into denoised / noise / ignored time series and save to disk.

    Parameters
    ----------
    data : (S x T) array_like
        Input time series
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like
        Boolean mask array
    io_generator : :obj:`tedana.io.OutputGenerator`
        Reference image to dictate how outputs are saved to disk
    out_dir : :obj:`str`, optional
        Output directory.
    echo : :obj: `int`, optional
        Echo number to generate filenames, used by some verbose
        functions. Currently this is only not 0 when
        io_generator.verbose==True. Default 0.

    Returns
    -------
    varexpl : :obj:`float`
        Percent variance of data explained by extracted + retained components

    Generated Files
    ---------------

    =====================================    ============================================
    Filename                                 Content
    =====================================    ============================================
    desc-denoised_bold.nii.gz                Denoised time series.

    if io_generator.verbose==True
    desc-optcomAccepted_bold.nii.gz          High-Kappa time series.
    desc-optcomRejected_bold.nii.gz          Low-Kappa time series.

    if echo>0
    echo-[echo]_desc-Accepted_bold.nii.gz    High-Kappa timeseries for echo
                                             number ``echo``.
    echo-[echo]_desc-Rejected_bold.nii.gz    Low-Kappa timeseries for echo
                                             number ``echo``.
    echo-[echo]_desc-Denoised_bold.nii.gz    Denoised timeseries for echo
                                             number ``echo``.
    =====================================    ============================================
    """
    acc = comptable[comptable.classification == "accepted"].index.values
    rej = comptable[comptable.classification == "rejected"].index.values

    dnts, hikts, lowkts = denoise_ts(data, mmix, mask, comptable)

    if len(acc) != 0:
        if echo != 0:
            # This outputs time series for a single echo
            # In practice this only happens when verbose is true
            # in io.writeresults_echoes. Neither this or the elif below
            # are written out if verbose is not true
            fout = io_generator.save_file(hikts, "high kappa ts split img", echo=echo)
            LGR.info(f"Writing high-Kappa time series: {fout}")
        elif io_generator.verbose:
            fout = io_generator.save_file(hikts, "high kappa ts img")
            LGR.info(f"Writing high-Kappa time series: {fout}")

    if len(rej) != 0:
        if echo != 0:
            fout = io_generator.save_file(lowkts, "low kappa ts split img", echo=echo)
            LGR.info(f"Writing low-Kappa time series: {fout}")
        elif io_generator.verbose:
            fout = io_generator.save_file(lowkts, "low kappa ts img")
            LGR.info(f"Writing low-Kappa time series: {fout}")

    if echo != 0:
        fout = io_generator.save_file(dnts, "denoised ts split img", echo=echo)
    else:
        fout = io_generator.save_file(dnts, "denoised ts img")

    LGR.info(f"Writing denoised time series: {fout}")


def writeresults(ts, mask, comptable, mmix, io_generator):
    """Denoise `ts` and save all resulting files to disk.

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
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk

    See Also
    --------
    tedana.io.write_split_ts : Writes out time series files

    Generated Files
    ---------------

    =========================================    ===========================================
    Filename                                     Content
    =========================================    ===========================================
    desc-denoised_bold.nii.gz                    Denoised time series.

    desc-optcomAccepted_bold.nii.gz              High-Kappa time series. (only with verbose)
    desc-optcomRejected_bold.nii.gz              Low-Kappa time series. (only with verbose)
    desc-ICA_components.nii.gz                   Spatial component maps for all
                                                 components.
    desc-ICAAccepted_components.nii.gz           Spatial component maps for accepted
                                                 components.
    desc-ICAAccepted_stat-z_components.nii.gz    Z-normalized spatial component maps
                                                 for accepted components.
    =========================================    ===========================================
    """
    acc = comptable[comptable.classification == "accepted"].index.values
    write_split_ts(ts, mmix, mask, comptable, io_generator)

    ts_pes = get_coeffs(ts, mmix, mask)
    fout = io_generator.save_file(ts_pes, "ICA components img")
    LGR.info(f"Writing full ICA coefficient feature set: {fout}")

    if len(acc) != 0:
        fout = io_generator.save_file(ts_pes[:, acc], "ICA accepted components img")
        LGR.info(f"Writing denoised ICA coefficient feature set: {fout}")

        # write feature versions of components
        feats = computefeats2(split_ts(ts, mmix, mask, comptable)[0], mmix[:, acc], mask)
        feats = utils.unmask(feats, mask)
        fname = io_generator.save_file(feats, "z-scored ICA accepted components img")
        LGR.info(f"Writing Z-normalized spatial component maps: {fname}")


def writeresults_echoes(catd, mmix, mask, comptable, io_generator):
    """Save individually denoised echos to disk.

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

    See Also
    --------
    tedana.io.write_split_ts : Writes out the files.

    Generated Files
    ---------------

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
    """
    for i_echo in range(catd.shape[1]):
        LGR.info(f"Writing Kappa-filtered echo #{i_echo + 1:01d} timeseries")
        write_split_ts(catd[:, i_echo, :], mmix, mask, comptable, io_generator, echo=(i_echo + 1))


# File Loading Functions
def load_data(data, n_echos=None):
    """Coerce input `data` files to required 3D array output.

    Parameters
    ----------
    data : :obj:`list` of img_like, :obj:`list` of :obj:`str`, :obj:`str`, or img_like
        A list of echo-wise img objects or paths to files.
        Single img objects or filenames are allowed as well, to support z-concatenated data.
    n_echos : :obj:`int`, optional
        Number of echos in provided data array. Only necessary if `data` is a single,
        z-concatenated file. Default: None

    Returns
    -------
    fdata : (S x E x T) :obj:`numpy.ndarray`
        Output data where `S` is samples, `E` is echos, and `T` is time.
    ref_img : img_like
        Reference image object for saving output files.
    """
    if n_echos is None and (isinstance(data, str) or len(data) == 1):
        raise ValueError(
            "Number of echos must be specified when a single z-concatenated file is supplied."
        )

    if not isinstance(data, (list, str, nib.spatialimages.SpatialImage)):
        raise TypeError(f"Unsupported type: {type(data)}")
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, (str, nib.spatialimages.SpatialImage)):
                raise TypeError(f"Unsupported type: {type(item)}")

        if len(data) == 1:  # a z-concatenated file was provided
            data = data[0]
        elif len(data) == 2:  # inviable -- need more than 2 echos
            raise ValueError(f"Cannot run `tedana` with only two echos: {data}")
        else:  # individual echo files were provided (surface or volumetric)
            fdata = np.stack([utils.reshape_niimg(f) for f in data], axis=1)
            ref_img = check_niimg(data[0])
            ref_img.header.extensions = []
            return np.atleast_3d(fdata), ref_img

    # Z-concatenated file/img
    img = check_niimg(data)
    (nx, ny), nz = img.shape[:2], img.shape[2] // n_echos
    fdata = utils.reshape_niimg(img.get_fdata().reshape(nx, ny, nz, n_echos, -1, order="F"))
    # create reference image
    ref_img = img.__class__(
        np.zeros((nx, ny, nz, 1)), affine=img.affine, header=img.header, extra=img.extra
    )
    ref_img.header.extensions = []
    ref_img.header.set_sform(ref_img.header.get_sform(), code=1)

    return fdata, ref_img


# Helper Functions
def new_nii_like(ref_img, data, affine=None, copy_header=True):
    """Coerce `data` into NiftiImage format like `ref_img`.

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
    if ".nii" not in ref_img.valid_exts:
        # this is rather ugly and may lose some information...
        nii = nib.Nifti1Image(newdata, affine=ref_img.affine, header=ref_img.header)
    else:
        # nilearn's `new_img_like` is a very nice function
        nii = new_img_like(ref_img, newdata, affine=affine, copy_header=copy_header)
    nii.set_data_dtype(data.dtype)

    return nii


def split_ts(data, mmix, mask, comptable):
    """Split `data` time series into accepted component time series and remainder.

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
    acc = comptable[comptable.classification == "accepted"].index.values

    cbetas = get_coeffs(data - data.mean(axis=-1, keepdims=True), mmix, mask)
    betas = cbetas[mask]
    if len(acc) != 0:
        hikts = utils.unmask(betas[:, acc].dot(mmix.T[acc, :]), mask)
    else:
        hikts = None

    resid = data - hikts

    return hikts, resid


def prep_data_for_json(d) -> dict:
    """Attempt to create a JSON serializable dictionary from a data dictionary.

    Parameters
    ----------
    d : dict
        A dictionary that will be converted into something JSON serializable

    Returns
    -------
    An attempt at JSON serializable data

    Raises
    ------
    ValueError if it cannot force the dictionary to be serializable
    TypeError if you do not supply a dict

    Notes
    -----
    Use this to make something serializable when writing JSON to disk.
    To speed things up since there are a small number of conversions, this
    function does not check for serializability, but does use conversion
    rules for cases encountered where things were not serializable.
    Add more conversion rules to this function in cases where a
    tedana-generated object does not serialize to JSON.
    """
    if not isinstance(d, dict):
        raise TypeError(f"Dictionary required to force serialization; got type {type(d)} instead.")
    # The input dictionary may want to retain types, so we copy
    d = deepcopy(d)
    for k, v in d.items():
        if isinstance(v, dict):
            # One of the values in the dict is the problem, need to recurse
            v = prep_data_for_json(v)
        elif isinstance(v, np.ndarray):
            if v.dtype == np.int64 or v.dtype == np.uint64:
                v = int(v)
            v = v.tolist()
        elif isinstance(v, np.int64) or isinstance(v, np.uint64):
            v = int(v)

        # NOTE: add more special cases for type conversions above this
        # comment line as an elif block
        d[k] = v
    return d


def str_to_component_list(s: str) -> List[int]:
    """Convert a string to a list of component indices.

    Parameters
    ----------
    s : str
        The string to convert into a list of component indices.

    Returns
    -------
    List[int] of component indices.

    Raises
    ------
    ValueError, if the string cannot be split by an allowed delimeter
    """
    if not s:
        LGR.warning("Component string is empty ")
        return []

    # Strip off newline at end in case we've been given a one-line file
    if s[-1] == "\n":
        s = s[:-1]

    # Search across all allowed delimiters for a match
    for d in ALLOWED_COMPONENT_DELIMITERS:
        possible_list = s.split(d)
        if len(possible_list) > 1:
            # We have a likely hit
            # Check to see if extra delimeter at end and get rid of it
            if possible_list[-1] == "":
                possible_list = possible_list[:-1]
            break
        elif len(possible_list) == 1 and possible_list[0].isnumeric():
            # We have a likely hit and there is just one component
            break

    # Make sure we can actually convert this split list into an integer
    # Crash with a sensible error if not
    for x in possible_list:
        try:
            int(x)
        except ValueError:
            raise ValueError(
                "While parsing component list, failed to convert to int."
                f' Offending element is "{x}", offending string is "{s}".'
            )

    return [int(x) for x in possible_list]


def fname_to_component_list(fname: str) -> List[int]:
    """Read a file of component indices.

    Parameters
    ----------
    fname : str
        The name of the file to read the list of component indices from.

    Returns
    -------
    List[int] of component indices.

    Raises
    ------
    ValueError, if the string cannot be split by an allowed delimeter or the
    csv file cannot be interpreted.
    """
    if fname[-3:] == "csv":
        try:
            contents = pd.read_csv(fname)
        except Exception:
            LGR.warning(f"{fname} is empty ")
            return []

        columns = contents.columns
        if len(columns) == 2 and "0" in columns:
            return contents["0"].tolist()
        elif len(columns) >= 2 and "Components" in columns:
            return contents["Components"].tolist()
        else:
            raise ValueError(f"Cannot determine a components column in file {fname}")

    with open(fname) as fp:
        contents = fp.read()
        if len(contents) == 0:
            LGR.warning(f"{fname} is empty ")
        return str_to_component_list(contents)


def _infer_prefix(prefix):
    """Determine the appropriate prefix for output files."""
    prefix = prefix + "_" if (prefix != "" and not prefix.endswith("_")) else prefix
    return prefix
