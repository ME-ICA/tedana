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
    from nilearn import image, masking

    # For this, you need the mixing matrix, the data you're denoising,
    # a brain mask, and an index of "bad" components
    data_file = "desc-optcom_bold.nii.gz"
    mixing_file = "desc-ICA_mixing.tsv"
    metrics_file = "desc-tedana_metrics.tsv"
    mask_file = "desc-adaptiveGoodSignal_mask.nii.gz"

    # Load the mixing matrix
    mixing_df = pd.read_table(mixing_file, index_col="component")
    mixing = mixing_df.data  # Shape is time-by-components

    # Load the component table
    metrics_df = pd.read_table(metrics_file)
    rejected_components_idx = metrics_df.loc[
        metrics_df["classification"] == "rejected"
    ].index.values
    kept_components_idx = metrics_df.loc[
        metrics_df["classification"] != "rejected"
    ].index.values

    # Select "bad" components from the mixing matrix
    rejected_components = mixing[:, rejected_components_idx]

    # Binarize the adaptive mask
    mask_img = image.math_img("img >= 1", img=mask_file)

    # Apply the mask to the data image to get a 2d array
    data = masking.apply_mask(data_file, mask_img)
    data = data.T  # Transpose to voxels-by-time

.. tab:: Python with fMRIPrep confounds

  .. code-block:: python

    import numpy as np
    import pandas as pd
    from nilearn import masking

    # For this, you need the mixing matrix, the data you're denoising,
    # a brain mask, and an index of "bad" components
    data_file = "sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    mixing_file = "sub-01_task-rest_desc-ICA_mixing.tsv"
    metrics_file = "sub-01_task-rest_desc-tedana_metrics.tsv"
    mask_file = "sub-01_task-rest_desc-brain_mask.nii.gz"
    confounds_file = "sub-01_task-rest_desc-confounds_timeseries.tsv"

    # Load the mixing matrix
    mixing_df = pd.read_table(mixing_file, index_col="component")
    mixing = mixing_df.data  # Shape is time-by-components

    # Load the component table
    metrics_df = pd.read_table(metrics_file)
    rejected_components_idx = metrics_df.loc[
        metrics_df["classification"] == "rejected"
    ].index.values
    kept_components_idx = metrics_df.loc[
        metrics_df["classification"] != "rejected"
    ].index.values

    # Load the confounds file
    confounds_df = pd.read_table(confounds_file)
    confounds = confounds_df[
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf", "white_matter"
    ].values

    # Select "bad" components from the mixing matrix
    rejected_components = mixing[:, rejected_components_idx]

    # Apply the mask to the data image to get a 2d array
    data = masking.apply_mask(data_file, mask_file)
    data = data.T  # Transpose to voxels-by-time

.. tab:: AFNI

  .. code-block:: bash

    data_file=desc-optcom_bold.nii.gz
    mixing_file=desc-ICA_mixing.tsv
    metrics_file=desc-tedana_metrics.tsv
    mask_file=desc-adaptiveGoodSignal_mask.nii.gz

********************************************************************************
Remove all noise-correlated fluctuations ("aggressive" denoising)
********************************************************************************

If you regress just nuisance regressors (i.e., rejected components) out of your data,
then retain the residuals for further analysis, you are doing aggressive denoising.

.. tab:: Python

  .. code-block:: python

    # Fit GLM to bad components only (after adding a constant term)
    regressors = np.hstack((rejected_components, np.ones(rejected_components.shape[0], 1)))
    betas = np.linalg.lstsq(regressors, data, rcond=None)[0][:-1]

    # Denoise the data with the bad components
    pred_data = np.dot(rejected_components, betas)
    data_denoised = data - pred_data

    # Save to file
    img_denoised = masking.unmask(data_denoised.T, mask_file)
    img_denoised.to_filename("desc-aggrDenoised_bold.nii.gz")

.. tab:: Python with fMRIPrep confounds

  .. code-block:: python

    # Fit GLM to bad components and nuisance regressors only (after adding a constant term)
    regressors = np.hstack((rejected_components, confounds, np.ones(rejected_components.shape[0], 1)))
    betas = np.linalg.lstsq(regressors, data, rcond=None)[0][:-1]

    # Denoise the data with the bad components and other nuisance regressors
    pred_data = np.dot(rejected_components, betas)
    data_denoised = data - pred_data

    # Save to file
    img_denoised = masking.unmask(data_denoised.T, mask_file)
    img_denoised.to_filename("desc-aggrDenoised_bold.nii.gz")

.. tab:: AFNI

  .. code-block:: bash

    3dcalc --input stuff

*******************************************************************************************************************************
Remove noise-correlated fluctuations that aren't correlated with fluctuations in accepted components (Non-aggressive denoising)
*******************************************************************************************************************************

If you include both nuisance regressors and regressors of interest in your regression,
you are doing nonaggressive denoising.

.. tab:: Python

  .. code-block:: python

    # Fit GLM to all components (after adding a constant term)
    regressors = np.hstack((mixing, np.ones(mixing.shape[0], 1)))
    betas = np.linalg.lstsq(regressors, data, rcond=None)[0][:-1]

    # Denoise the data using the betas from just the bad components
    pred_data = np.dot(rejected_components, betas[rejected_components_idx, :])
    data_denoised = data - pred_data

    # Save to file
    img_denoised = masking.unmask(data_denoised.T, mask_file)
    img_denoised.to_filename("desc-nonaggrDenoised_bold.nii.gz")

.. tab:: Python with fMRIPrep confounds

  .. code-block:: python

    # Fit GLM to all components and nuisance regressors (after adding a constant term)
    regressors = np.hstack((confounds, mixing, np.ones(mixing.shape[0], 1)))
    betas = np.linalg.lstsq(regressors, data, rcond=None)[0][:-1]

    # Denoise the data using the betas from just the bad components
    confounds_idx = np.concat(
        np.arange(confounds.shape[1]),
        rejected_components_idx + confounds.shape[1],
      )
    pred_data = np.dot(np.hstack(confounds, rejected_components), betas[confounds_idx, :])
    data_denoised = data - pred_data

    # Save to file
    img_denoised = masking.unmask(data_denoised.T, mask_file)
    img_denoised.to_filename("desc-nonaggrDenoised_bold.nii.gz")

.. tab:: AFNI

  .. code-block:: bash

    3dcalc --input stuff


********************************************************************************
Component orthogonalization
********************************************************************************

Independent component analysis decomposes the data into _independent_ components, obviously.
Unlike principal components analysis, the components from ICA are not orthogonal, so they may explain shared variance.
If you want to ensure that variance shared between the accepted and rejected components does not contaminate the denoised data,
you may wish to orthogonalize the rejected components with respect to the accepted components.
This way, you can regress the rejected components out of the data in the form of, what we call, "pure evil" components.

.. note::
  The ``tedana`` workflow's ``--tedort`` option performs this orthogonalization automatically and
  writes out a separate mixing matrix file.

.. tab:: Python

  .. code-block:: python

    # Separate the mixing matrix into "good" and "bad" components
    rejected_components = mixing[:, rejected_components_idx]
    kept_components = mixing[:, kept_components_idx]

    # Regress the good components out of the bad ones
    betas = np.linalg.lstsq(kept_components, rejected_components, rcond=None)[0]
    pred_rejected_components = np.dot(kept_components, betas)
    orth_rejected_components = rejected_components - pred_rejected_components

    # Replace the old component time series in the mixing matrix with the new ones
    mixing[:, rejected_components_idx] = orth_rejected_components

.. tab:: Python with fMRIPrep confounds

  .. code-block:: python

    # Separate the mixing matrix and confounds into "good" and "bad" time series
    rejected_components = mixing[:, rejected_components_idx]
    kept_components = mixing[:, kept_components_idx]
    bad_timeseries = np.hstack((rejected_components, confounds))

    # Regress the good components out of the bad time series
    betas = np.linalg.lstsq(kept_components, bad_timeseries, rcond=None)[0]
    pred_bad_timeseries = np.dot(kept_components, betas)
    orth_bad_timeseries = bad_timeseries - pred_bad_timeseries

    # Replace the old component time series in the mixing matrix with the new ones
    mixing[:, rejected_components_idx] = orth_bad_timeseries[:rejected_components.shape[1]]
    orth_confounds = orth_bad_timeseries[rejected_components.shape[1]:]

.. tab:: AFNI

  .. code-block:: bash

    3dcalc --input stuff

Once you have these "pure evil" components, you can perform aggressive denoising on the data.

.. tab:: Python

  .. code-block:: python

    # Fit GLM to bad components only (after adding a constant term)
    regressors = np.hstack(
        (orth_rejected_components, np.ones(orth_rejected_components.shape[0], 1))
    )
    betas = np.linalg.lstsq(orth_rejected_components, data, rcond=None)[0][:-1]

    # Denoise the data with the bad components
    pred_data = np.dot(orth_rejected_components, betas)
    data_denoised = data - pred_data

    # Save to file
    img_denoised = masking.unmask(data_denoised.T, mask_file)
    img_denoised.to_filename("desc-orthAggrDenoised_bold.nii.gz")

.. tab:: Python with fMRIPrep confounds

  .. code-block:: python

    # Fit GLM to bad components only (after adding a constant term)
    regressors = np.hstack(
        (orth_bad_timeseries, np.ones(orth_bad_timeseries.shape[0], 1))
    )
    betas = np.linalg.lstsq(regressors, data, rcond=None)[0][:-1]

    # Denoise the data with the bad components
    pred_data = np.dot(regressors, betas)
    data_denoised = data - pred_data

    # Save to file
    img_denoised = masking.unmask(data_denoised.T, mask_file)
    img_denoised.to_filename("desc-orthAggrDenoised_bold.nii.gz")

.. tab:: AFNI

  .. code-block:: bash

    3dcalc --input stuff
