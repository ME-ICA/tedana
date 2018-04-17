import os
import nibabel as nib
import numpy as np
from tedana.utils.utils import make_opt_com


class MultiEchoData():
    """
    Object for holding multi-echo data and related information (e.g., echo
    times, header, mask)

    Parameters
    ----------
    fname : str
        Filepath to Z-concatenated multi-echo data image. Must be in a format
        compatible with nibabel (see http://nipy.org/nibabel/ for more info on
        currently supported filetypes).
    tes : array_like
        Echo times for `zcat`

    Attributes
    ----------
    img, header : nibabel-object
        Nibabel object containing loaded input data (i.e., `nib.load(fname)`)
    tes : np.ndarray
        Array containing echo times (in milliseconds) for input data
    zcat : (X x Y x ZE x T) np.ndarray
        Multi-echo data array, where individual echo time series are
        concatenated along the third dimension (i.e., ZE)
    catdata : (X x Y x Z x E x T) np.ndarray
        Multi-echo data array, where X, Y, Z are spatial dimensions, E
        corresponds to individual echo data, and T is time
    mask : (X x Y x Z) np.ndarray, bool
        Boolean array that encompasses points in `catdata` for which the
        provided data has good signal across echoes
    masksum : (X x Y x Z) np.ndarray, int
        Array containing information about number of echoes that
        contributed to generation of `mask`
    """

    def __init__(self, fname, tes):
        self.fname = os.path.abspath(fname)
        if not os.path.exists(fname):
            raise IOError('Cannot find file: {0}. '.format(self.fname) +
                          'Please ensure file exists on current system.')
        try:
            self.img = nib.load(fname)
        except Exception as e:
            print('Error loading file: {0}'.format(self.fname) +
                  'Please ensure file is in a format compatible with nibabel.')
            raise e

        self.header = self.img.header
        self.zcat = self.img.get_fdata()
        self.tes = np.asarray(tes)

        self._cat2echos()
        self._gen_ad_mask()

    def _cat2echos(self):
        """
        Uncatenates individual echo time series from Z-concatenated array
        """

        nx, ny, Ne = self.zcat.shape[:2], self.tes.size
        nz = self.zcat.shape[2] // Ne

        self.catdata = self.zcat.reshape(nx, ny, nz, Ne, -1, order='F')

    def _gen_ad_mask(self):
        """
        Generates mask from `catdata`

        By default, generates a 3D mask (boolean) for voxels with good signal
        in at least one echo and an 3D array (integer) detailing how many
        echoes have good signal in that voxel

        Returns
        -------
        mask : (X x Y x Z) np.ndarray, bool
            Boolean mask array
        masksum : (X x Y x Z) np.ndarray, int
            Value mask array
        """

        emeans = self.catdata.mean(axis=-1)
        first_echo = emeans[:, :, :, 0]
        perc33 = np.percentile(first_echo[first_echo.nonzero()], 33,
                               interpolation='higher')
        medv = (first_echo == perc33)
        lthrs = np.vstack([emeans[:, :, :, echo][medv] / 3 for echo in
                           range(self.tes.size)])
        lthrs = lthrs[:, lthrs.sum(0).argmax()]
        mthr = np.ones(self.catdata.shape[:-1])
        for echo in range(self.tes.size):
            mthr[:, :, :, echo] *= lthrs[echo]

        self.masksum = (np.abs(emeans) > mthr).astype('int').sum(axis=-1)
        self.mask = (self.masksum != 0)


class T2Star():
    """
    Generates T2* map and optimal combination from multi-echo data

    Parameters
    ----------
    medata : MultiEchoData
        Instance of MultiEchoData() class

    Attributes
    ----------
    optcom : (X x Y x Z) np.ndarray
    t2s : (X x Y x Z) np.ndarray
    s0 : (X x Y x Z) np.ndarray
    t2ss : (X x Y x Z) np.ndarray
    s0s : (X x Y x Z) np.ndarray
    t2sG : (X x Y x Z) np.ndarray
    s0G : (X x Y x Z) np.ndarray
    """

    def __init__(self, medata):
        if not isinstance(medata, MultiEchoData):
            raise TypeError("Input must be an instance of MultiEchoData.")
        self.medata = medata

        self._gen_t2sadmap()
        self.optcom = make_opt_com()

    def _gen_t2sadmap(self):
        pass
