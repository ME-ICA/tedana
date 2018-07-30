"""
"""
import numpy as np
import pandas as pd
import nibabel as nib


def row_idx(arr1, arr2):
    """
    Get a 1D index of rows in arr1 that exist in arr2.

    Parameters
    ----------
    arr1 : (X x 3) :obj:`numpy.ndarray`
    arr2 : (Z x 3) :obj:`numpy.ndarray`

    Returns
    -------
    idx : 1D :obj:`numpy.ndarray`
        Index of rows in arr1 that exist in arr2.

    Notes
    -----
    This works amazingly well, but is quite slow.
    """
    df1 = pd.DataFrame(arr1, columns=['x', 'y', 'z'], dtype=str)
    df2 = pd.DataFrame(arr2, columns=['x', 'y', 'z'], dtype=str)

    df1['unique_value'] = df1[['x', 'y', 'z']].apply(lambda x: '_'.join(x),
                                                     axis=1)
    df1 = df1[['unique_value']]
    df1['idx1'] = df1.index
    df1 = df1.set_index('unique_value')

    df2['unique_value'] = df2[['x', 'y', 'z']].apply(lambda x: '_'.join(x),
                                                     axis=1)
    df2 = df2[['unique_value']]
    df2['idx2'] = df2.index
    df2 = df2.set_index('unique_value')

    catted = pd.concat((df1, df2), axis=1, ignore_index=False)
    if any(pd.isnull(catted['idx1'])):
        raise Exception('We have a weird error where there is >=1 '
                        'voxel in echo-specific mask outside of union mask.')

    catted = catted.dropna()
    catted = catted.sort_values(by=['idx2'])
    rel_echo_idx = catted['idx1'].values
    return rel_echo_idx


def apply_mask_me(img, mask_img):
    """
    Apply multi-echo mask to set of nifti images. Mask may vary by echo.

    Parameters
    ----------
    img : (X x Y x Z x E [x T]) :obj:`nibabel.nifti.Nifti1Image`
        Data img.
    mask_img : (X x Y x Z [x E]) :obj:`nibabel.nifti.Nifti1Image`
        Mask img.

    Returns
    -------
    masked_arr : (M x E [x T]) :obj:`numpy.ndarray`
        Masked data. M refers to the number of voxels in the mask img.
    """
    if not isinstance(mask_img, nib.Nifti1Image):
        raise TypeError('Provided mask is not a Nifti1Image.')
    elif len(mask_img.shape) not in (3, 4):
        raise ValueError('Mask must be 3D (X x Y x Z) or 4D (X x Y x Z x E).')

    if not np.array_equal(img.affine, mask_img.affine):
        raise ValueError('Input img affine must match mask_img affine.')

    data = img.get_data()
    if len(img.shape) == 4:
        _, _, _, n_echos = img.shape
        n_vols = 1
        data = data[:, :, :, :, None]
    elif len(img.shape) == 5:
        _, _, _, n_echos, n_vols = img.shape
    else:
        raise ValueError('Input img must be 4D (X x Y x Z x E) or '
                         '5D (X x Y x Z x E x T).')

    n_x, n_y, n_z = mask_img.shape[:3]
    mask_arr = np.array(mask_img.get_data()).astype(bool)
    if mask_arr.ndim == 3:
        # We can probably simplify/speed things up when the mask is not
        # echo-dependent
        mask_arr = mask_arr[:, :, :, None]
        mask_arr = np.tile(mask_arr, (1, 1, 1, n_echos))

    union_mask = np.any(mask_arr, axis=3)
    union_idx = np.vstack(np.where(union_mask)).T
    masked_arr = np.empty((np.sum(union_mask), n_echos, n_vols))
    masked_arr[:] = np.nan

    for i_echo in range(n_echos):
        echo_mask = mask_arr[:, :, :, i_echo]
        abs_echo_idx = np.vstack(np.where(echo_mask)).T
        rel_echo_idx = row_idx(union_idx, abs_echo_idx)
        masked_arr[rel_echo_idx, i_echo, :] = data[abs_echo_idx[:, 0],
                                                   abs_echo_idx[:, 1],
                                                   abs_echo_idx[:, 2],
                                                   i_echo, :]
    return masked_arr


def unmask_me(X, mask_img):
    """
    Unmask multi-echo data to nifti image. Mask may vary by echo.

    Parameters
    ----------
    X : (M x E [x T]) :obj:`numpy.ndarray`
        Masked data. M refers to the number of voxels in the mask img.
    mask_img : (X x Y x Z [x E]) :obj:`nibabel.nifti.Nifti1Image`
        Mask img.

    Returns
    -------
    out_img : (X x Y x Z x E [x T]) :obj:`nibabel.nifti.Nifti1Image`
        Data img.
    """
    if not isinstance(mask_img, nib.Nifti1Image):
        raise TypeError('Provided mask is not a Nifti1Image.')
    elif len(mask_img.shape) not in (3, 4):
        raise ValueError('Mask must be 3D (X x Y x Z) or 4D (X x Y x Z x E).')

    if X.ndim == 2:
        X = X[:, :, None]
    elif X.ndim != 3:
        raise ValueError('X must be 2D (M x E) or 3D (M x E x T).')

    _, n_echos, n_vols = X.shape

    mask_arr = np.array(mask_img.get_data()).astype(bool)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, :, None]
        mask_arr = np.tile(mask_arr, (1, 1, 1, n_echos))
    n_x, n_y, n_z = mask_arr.shape[:3]

    out = np.zeros((n_x, n_y, n_z, n_echos, n_vols))
    for i_vol in range(n_vols):
        unmasked_arr = np.zeros(mask_arr.shape)
        for j_echo in range(n_echos):
            echo_x = X[:, j_echo, i_vol]
            # NaNs should occur where data have been masked out by echo-
            # specific mask but where voxels exist in overall mask
            echo_x = echo_x[~np.isnan(echo_x)]
            echo_mask_idx = np.vstack(np.where(mask_arr[:, :, :, j_echo])).T
            if echo_x.shape[0] != echo_mask_idx.shape[0]:
                raise ValueError('Masked data do not match dimensions of '
                                 'echo-specific mask.')
            unmasked_arr[echo_mask_idx[:, 0], echo_mask_idx[:, 1],
                         echo_mask_idx[:, 2], j_echo] = echo_x
        out[:, :, :, :, i_vol] = unmasked_arr
    out_img = nib.Nifti1Image(out, header=mask_img.header,
                              affine=mask_img.affine)
    return out_img
