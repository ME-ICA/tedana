##############################
Denoising Data with Components
##############################

Decomposition-based denoising methods like ``tedana`` will produce two important outputs:
component time series and component classifications.
The component classifications will indicate whether each component is "good" (accepted) or "bad" (rejected).
To remove noise from your data, you can regress the "bad" components out of it,
though there are multiple ways to accomplish this.

.. important::

  This page assumes that you have reviewed the component classifications produced by ``tedana`` already,
  and that you agree with those classifications.
  If you think some components were misclassified,
  you can find instructions for reviewing and changing classifications in
  :ref:`manual classification`.

By default, ``tedana`` will perform a regression including both "good" and "bad" components,
and then will selectively remove the "bad" components from the data.
This is colloquially known as "non-aggressive denoising".

However, users may wish to apply a different type of denoising,
or to incorporate other regressors into their denoising step,
and we will discuss these alternatives here.

``tedana`` uses independent component analysis (ICA) to decompose the data into components
which each have a spatial map of weights and time series that are assumed to reflect meaningful underlying signals.
It then classifies each component as "accepted" (BOLD-like) or "rejected" (non-BOLD-like).
The data are denoised by taking the time series of the rejected components and regressing them from the data.

``tedana`` uses a spatial ICA, which means the components are `statistically independent` across space, not time.
That means the time series from accepted and rejected components can share variance.
If a rejected component shares meaningful variance with an accepted component,
then regressing the rejected components' time series from your data may also remove meaningful signal
associated with the accepted component as well.
Depending on the application,
people may take different approaches on how to handle variance that is shared between accepted and rejected components.
These different approaches towards decomposition-based denoising methods are described here.

This page has three purposes:

1.  Describe different approaches to denoising using ICA components.
2.  Provide sample code that performs each type of denoising.
3.  Describe how to incorporate external regressors (e.g., motion parameters) into the denoising step.

.. admonition:: Which should you use?

  The decision about which denoising approach you should use depends on a number of factors.

  The first thing to know is that aggressive denoising, while appropriate for orthogonal time series,
  may remove signal associated with "good" time series that are independent to, but not orthogonal with,
  the "bad" time series being regressed out.

  As an alternative, users may want to try non-aggressive denoising or aggressive denoising combined with component orthogonalization.

  The main difference between the orthogonalization+aggressive and non-aggressive approaches is that,
  with orthogonalization,
  all the variance common across the accepted and rejected components is put in the accepted basket,
  and therefore it is not removed in the nuisance regression.
  In contrast, with the non-aggressive denoising,
  the shared variance is smartly split during the estimation process and therefore some amount is removed.

Now we can get started!
This walkthrough will show you how to perform different kinds of denoising on example data.

For this walkthrough, we will use Python and the Nilearn package to denoise data that were preprocessed in fMRIPrep.
However, you can definitely accomplish all of these steps in other languages, like MATLAB,
or using other neuroimaging toolboxes, like AFNI.

Let's start by loading the necessary data.
No matter which type of denoising you want to use, you will need to include this step.

For this, you need the mixing matrix, the data you're denoising, a brain mask,
and an index of "bad" components.

.. code-block:: python

  import pandas as pd  # A library for working with tabular data

  # Files from fMRIPrep
  data_file = "sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
  mask_file = "sub-01_task-rest_desc-brain_mask.nii.gz"
  confounds_file = "sub-01_task-rest_desc-confounds_timeseries.tsv"

  # Files from tedana (after running on fMRIPrepped data)
  mixing_file = "sub-01_task-rest_desc-ICA_mixing.tsv"
  metrics_file = "sub-01_task-rest_desc-tedana_metrics.tsv"

  # Load the mixing matrix
  mixing_df = pd.read_table(mixing_file)  # Shape is time-by-components

  # Load the component table
  metrics_df = pd.read_table(metrics_file)
  rejected_columns = metrics_df.loc[metrics_df["classification"] == "rejected", "Component"]
  accepted_columns = metrics_df.loc[metrics_df["classification"] == "accepted", "Component"]

  # Load the fMRIPrep confounds file
  confounds_df = pd.read_table(confounds_file)

  # Select external nuisance regressors we want to use for denoising
  confounds = confounds_df[
      [
          "trans_x",
          "trans_y",
          "trans_z",
          "rot_x",
          "rot_y",
          "rot_z",
          "csf",
          "white_matter",
      ]
  ].to_numpy()

  # Select "bad" components from the mixing matrix
  rejected_components = mixing_df[rejected_columns].to_numpy()
  accepted_components = mixing_df[accepted_columns].to_numpy()


*****************************************************************
Remove all noise-correlated fluctuations ("aggressive" denoising)
*****************************************************************

If you regress just nuisance regressors (i.e., rejected components) out of your data,
then retain the residuals for further analysis, you are doing "aggressive" denoising.

.. code-block:: python

  import numpy as np  # A library for working with numerical data
  from nilearn.maskers import NiftiMasker  # A class for masking and denoising fMRI data

  # Combine the rejected components and the fMRIPrep confounds into a single array
  regressors = np.hstack((rejected_components, confounds))

  masker = NiftiMasker(
      mask_img=mask_file,
      standardize_confounds=True,
      standardize=False,
      smoothing_fwhm=None,
      detrend=False,
      low_pass=None,
      high_pass=None,
      t_r=None,  # This shouldn't be necessary since we aren't bandpass filtering
      reports=False,
  )

  # Denoise the data by fitting and transforming the data file using the masker
  denoised_img_2d = masker.fit_transform(data_file, confounds=regressors)

  # Transform denoised data back into 4D space
  denoised_img_4d = masker.inverse_transform(denoised_img_2d)

  # Save to file
  denoised_img_4d.to_filename(
      "sub-01_task-rest_space-MNI152NLin2009cAsym_desc-aggrDenoised_bold.nii.gz"
  )


*********************************************************************************************************************************
Remove noise-correlated fluctuations that aren't correlated with fluctuations in accepted components ("non-aggressive" denoising)
*********************************************************************************************************************************

If you include both nuisance regressors and regressors of interest in your regression,
you are doing "non-aggressive" denoising.

Unfortunately, non-aggressive denoising is difficult to do with :mod:`nilearn`'s Masker
objects, so we will end up using :mod:`numpy` directly for this approach.

.. code-block:: python

  import numpy as np  # A library for working with numerical data
  from nilearn.masking import apply_mask, unmask  # Functions for (un)masking fMRI data

  # Apply the mask to the data image to get a 2d array
  data = apply_mask(data_file, mask_file)

  # Fit GLM to accepted components, rejected components and nuisance regressors
  # (after adding a constant term)
  regressors = np.hstack(
      (
          confounds,
          rejected_components,
          accepted_components,
          np.ones((mixing_df.shape[0], 1)),
      ),
  )
  betas = np.linalg.lstsq(regressors, data, rcond=None)[0][:-1]

  # Denoise the data using the betas from just the bad components
  confounds_idx = np.arange(confounds.shape[1] + rejected_components.shape[1])
  pred_data = np.dot(np.hstack((confounds, rejected_components)), betas[confounds_idx, :])
  data_denoised = data - pred_data

  # Save to file
  denoised_img = unmask(data_denoised, mask_file)
  denoised_img.to_filename(
      "sub-01_task-rest_space-MNI152NLin2009cAsym_desc-nonaggrDenoised_bold.nii.gz"
  )


************************************************************************************
Orthogonalize the noise components w.r.t. the accepted components prior to denoising
************************************************************************************

If you want to ensure that variance shared between the accepted and rejected components does not contaminate the denoised data,
you may wish to orthogonalize the rejected components with respect to the accepted components.
This way, you can regress the rejected components out of the data in the form of what we call "pure evil" components.

.. note::

  The ``tedana`` workflow's ``--tedort`` option performs this orthogonalization automatically and
  writes out a separate mixing matrix file.
  However, this orthogonalization only takes the components into account,
  so you will need to separately perform the orthogonalization yourself if you have other regressors you want to account for.

.. code-block:: python

  import numpy as np  # A library for working with numerical data
  from nilearn.maskers import NiftiMasker  # A class for masking and denoising fMRI data

  # Combine the confounds and rejected components in a single array
  bad_timeseries = np.hstack((rejected_components, confounds))

  # Regress the good components out of the bad time series to get "pure evil" regressors
  betas = np.linalg.lstsq(accepted_components, bad_timeseries, rcond=None)[0]
  pred_bad_timeseries = np.dot(accepted_components, betas)
  orth_bad_timeseries = bad_timeseries - pred_bad_timeseries

  # Once you have these "pure evil" components, you can denoise the data
  masker = NiftiMasker(
      mask_img=mask_file,
      standardize_confounds=True,
      standardize=False,
      smoothing_fwhm=None,
      detrend=False,
      low_pass=None,
      high_pass=None,
      t_r=None,  # This shouldn't be necessary since we aren't bandpass filtering
      reports=False,
  )

  # Denoise the data by fitting and transforming the data file using the masker
  denoised_img_2d = masker.fit_transform(data_file, confounds=orth_bad_timeseries)

  # Transform denoised data back into 4D space
  denoised_img_4d = masker.inverse_transform(denoised_img_2d)

  # Save to file
  denoised_img_4d.to_filename(
      "sub-01_task-rest_space-MNI152NLin2009cAsym_desc-orthAggrDenoised_bold.nii.gz"
  )
