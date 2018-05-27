"""
"""
import numpy as np
import nibabel as nib


def apply_mask_nii_me(imgs, mask_img):
    """
    Apply multi-echo mask to set of nifti images. Mask may vary by echo.
    """
    pass


def apply_mask_gii_me(img, mask_img):
    """
    Apply multi-echo mask to set of gifti images. Mask may vary by echo.

    Parameters
    ----------
    img : :obj:`nibabel.gifti.gifti.GiftiImage`
        3D (Vertex, Echo, Time) gifti image containing data.
    mask_img : :obj:`nibabel.gifti.gifti.GiftiImage`
        1D (Vertex) or 2D (Vertex, Echo) gifti image with boolean or bool-like
        values.

    Returns
    -------
    masked_arr : :obj:`numpy.ndarray`
        M x E x T data.
    """
    if not isinstance(mask_img, nib.gifti.GiftiImage):
        raise TypeError('Provided mask is not a GiftiImage.')
    elif len(mask_img.darrays) > 1:
        raise ValueError('Mask may only contain one GiftiDataArray.')

    if not isinstance(img, nib.gifti.GiftiImage):
        raise TypeError('Provided data img is not a GiftiImage.')

    img_arrs = [np.array(darr.data) for darr in img.darrays]
    all_imgs_same = np.all(arr.shape == img_arrs[0].shape for arr in img_arrs)
    if not all_imgs_same:
        raise ValueError('All volumes in data img must have same number of '
                         'vertices.')
    n_echos = img_arrs[0].shape[1]
    n_vols = len(img_arrs)

    mask_arr = np.array(mask_img.darrays[0].data).astype(bool)
    if img_arrs[0].shape[0] != mask_arr.shape[0]:
        raise ValueError('Multi-echo mask img and data img must have '
                         'same number of vertices.')

    if mask_arr.ndim == 1:
        mask_arr = np.tile(mask_arr, (n_echos, 1))
    elif mask_arr.shape[1] != img_arrs[0].shape[1]:
        raise ValueError('Multi-echo mask img and data img must have '
                         'same number of echos.')

    union_mask = np.any(mask_arr, axis=0)
    union_idx = np.where(union_mask)[0]
    masked_arrs = [np.empty((np.sum(union_mask), n_echos)) for _ in
                   range(len(img_arrs))]
    for i_vol in range(n_vols):
        masked_arrs[i_vol][:] = np.nan

    for i_echo in range(n_echos):
        echo_mask = mask_arr[i_echo, :]
        abs_echo_idx = np.where(echo_mask)[0]
        rel_echo_idx = np.array([np.where(union_idx == idx_val)[0][0] for
                                 idx_val in abs_echo_idx])
        for j_vol in range(n_vols):
            masked_arrs[j_vol][i_echo, rel_echo_idx] = \
                img_arrs[j_vol][i_echo, abs_echo_idx]

    masked_arr = np.dstack(masked_arrs)
    return masked_arr


def unmask_nii_me(X, mask_img):
    """
    Unmask multi-echo data to nifti image. Mask may vary by echo.
    """
    pass


def unmask_gii_me(X, mask_img):
    """
    Unmask multi-echo data to gifti image. Mask may vary by echo.
    """
    if not isinstance(mask_img, nib.gifti.GiftiImage):
        raise TypeError('Provided mask is not a GiftiImage.')
    elif len(mask_img.darrays) > 1:
        raise ValueError('Mask may only contain one GiftiDataArray.')

    if X.ndim == 2:
        X = X[:, :, None]
    elif X.ndim != 3:
        raise ValueError('Input X it not 2D or 3D.')

    _, n_echos, n_vols = X.shape

    mask_arr = np.array(mask_img.darrays[0].data).astype(bool)
    if mask_arr.ndim == 1:
        mask_arr = np.tile(mask_arr, (n_echos, 1))

    out_darrs = []
    for _ in range(n_vols):
        unmasked_arr = np.zeros(mask_arr.shape)
        for j_echo in range(n_echos):
            echo_x = X[:, j_echo, :]
            echo_mask = mask_arr[j_echo, :]
            echo_mask_idx = np.where(echo_mask)[0]
            echo_x = echo_x[~np.isnan(echo_x)]
            if echo_x.shape[0] != echo_mask_idx.shape[0]:
                raise ValueError('Masked data do not match dimensions of '
                                 'echo-specific mask.')
            unmasked_arr[j_echo, echo_mask_idx] = echo_x
        unmasked_darr = nib.gifti.GiftiDataArray(unmasked_arr)
        out_darrs.append(unmasked_darr)
    out_img = nib.gifti.GiftiImage(header=mask_img.header,
                                   extra=mask_img.extra,
                                   meta=mask_img.meta,
                                   labeltable=mask_img.labeltable,
                                   darrays=out_darrs)
    return out_img
