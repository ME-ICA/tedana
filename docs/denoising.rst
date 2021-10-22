##############################
Denoising Data with Components
##############################

Decomposition-based denoising methods like ``tedana`` will produce two important outputs: component time series and component classifications.
The component classifications will indicate whether each componet is "good" (accepted) or "bad" (rejected).
To remove noise from your data, you can regress the "bad" components out of it, though there are multiple methods to accomplish this.

``tedana`` will produce **non-aggressively denoised** data automatically.
However, users may wish to incorporate other regressors in their denoising step,
in which case further denoising the ``tedana``-denoised data is not recommended.
Alternatively, users may wish to employ a different denoising approach (e.g., **aggressive denoising**).

Let's start by loading the necessary data.

.. tab:: Python

  .. code-block:: python

    import numpy as np
    import pandas as pd
    from nilearn import masking

    # For this, you need the mixing matrix, the data you're denoising,
    # a brain mask, and an index of "bad" components
    data_file = "preprocessed_data.nii.gz"
    mixing_file = "mixing.tsv"
    mask_file = "mask.nii.gz"
    den_idx = np.array([0, 1, 2, 3, 4, 5])

    # Load the mixing matrix
    mixing_df = pd.read_table(mixing_file, index_col="component")
    mixing = mixing_df.data

    # Apply the mask to the data image to get a 2d array
    data = masking.apply_mask(data_file, mask_file)

    # Transpose to voxels-by-time
    data = data.T

    # The first dimension should be time
    assert data.shape[1] == mixing.shape[0]

.. tab:: FSL

  .. code-block:: bash

    data_file=preprocessed_data.nii.gz
    mixing_file=mixing.tsv
    mask_file=mask.nii.gz
    den_idx=(0, 1, 2, 3, 4, 5)

.. tab:: AFNI

  .. code-block:: bash

    data_file=preprocessed_data.nii.gz
    mixing_file=mixing.tsv
    mask_file=mask.nii.gz
    den_idx=(0, 1, 2, 3, 4, 5)

********************
Aggressive Denoising
********************

If you regress just nuisance regressors (i.e., rejected components) out of your data,
then retain the residuals for further analysis, you are doing aggressive denoising.

.. tab:: Python

  .. code-block:: python

    # Fit GLM to bad components only
    betas = np.linalg.lstsq(motion_components, data, rcond=None)[0]

    # Denoise the data with the bad components
    pred_data = np.dot(motion_components, betas)
    data_denoised = data - pred_data

    # Save to file
    img_denoised = masking.unmask(data_denoised.T, mask_file)
    img_denoised.to_filename("denoised.nii.gz")

.. tab:: FSL

  .. code-block:: bash

    3dcalc --input stuff

.. tab:: AFNI

  .. code-block:: bash

    3dcalc --input stuff

************************
Non-Aggressive Denoising
************************

If you include both nuisance regressors and regressors of interest in your regression,
you are doing nonaggressive denoising.

.. tab:: Python

  .. code-block:: python

    # Fit GLM to all components
    betas = np.linalg.lstsq(mixing, data, rcond=None)[0]

    # Denoise the data using the betas from just the bad components
    pred_data = np.dot(motion_components, betas[den_idx, :])
    data_denoised = data - pred_data

    # Save to file
    img_denoised = masking.unmask(data_denoised.T, mask_file)
    img_denoised.to_filename("denoised.nii.gz")

.. tab:: FSL

  .. code-block:: bash

    3dcalc --input stuff

.. tab:: AFNI

  .. code-block:: bash

    3dcalc --input stuff


***************************
Component orthogonalization
***************************

Independent component analysis decomposes the data into _independent_ components, obviously.
Unlike principal components analysis, the components from ICA are not orthogonal, so they may explain shared variance.
If you want to ensure that variance shared between the accepted and rejected components does not contaminate the denoised data,
you may wish to orthogonalize the rejected components with respect to the accepted components.
This way, you can regress the rejected components out of the data in the form of, what we call, "pure evil" components.

.. tab:: Python

  .. code-block:: python

    good_idx = np.setdiff1d(np.arange(mixing.shape[1]), den_idx)

    # Separate the mixing matrix into "good" and "bad" components
    bad_mixing = mixing[:, den_idx]
    good_mixing = mixing[:, good_idx]

    # Regress the good components out of the bad ones
    betas = np.linalg.lstsq(good_mixing, bad_mixing, rcond=None)[0]
    pred_bad_mixing = np.dot(good_mixing, betas)
    orth_motion_components = bad_mixing - pred_bad_mixing

    # Replace the old component time series in the mixing matrix with the new ones
    mixing[:, den_idx] = orth_motion_components

.. tab:: FSL

  .. code-block:: bash

    3dcalc --input stuff

.. tab:: AFNI

  .. code-block:: bash

    3dcalc --input stuff

Once you have these "pure evil" components, you can perform aggressive denoising on the data.

.. tab:: Python

  .. code-block:: python

    # Fit GLM to bad components only
    betas = np.linalg.lstsq(orth_motion_components, data, rcond=None)[0]

    # Denoise the data with the bad components
    pred_data = np.dot(orth_motion_components, betas)
    data_denoised = data - pred_data

    # Save to file
    img_denoised = masking.unmask(data_denoised.T, mask_file)
    img_denoised.to_filename("denoised.nii.gz")

.. tab:: FSL

  .. code-block:: bash

    3dcalc --input stuff

.. tab:: AFNI

  .. code-block:: bash

    3dcalc --input stuff
