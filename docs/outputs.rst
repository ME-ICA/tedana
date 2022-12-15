.. _outputs:

#################
Outputs of tedana
#################


******************************
Outputs of the tedana workflow
******************************

================================================    =====================================================
Filename                                            Content
================================================    =====================================================
dataset_description.json                            Top-level metadata for the workflow.
T2starmap.nii.gz                                    Full estimated T2* 3D map.
                                                    Values are in seconds.
                                                    The difference between the limited and full maps
                                                    is that, for voxels affected by dropout where
                                                    only one echo contains good data, the full map uses
                                                    the T2* estimate from the first two echoes, while the
                                                    limited map has a NaN.
S0map.nii.gz                                        Full S0 3D map.
                                                    The difference between the limited and full maps
                                                    is that, for voxels affected by dropout where
                                                    only one echo contains good data, the full map uses
                                                    the S0 estimate from the first two echoes, while the
                                                    limited map has a NaN.
desc-optcom_bold.nii.gz                             Optimally combined time series.
desc-optcomDenoised_bold.nii.gz                     Denoised optimally combined time series. Recommended
                                                    dataset for analysis.
desc-optcomRejected_bold.nii.gz                     Combined time series from rejected components.
desc-optcomAccepted_bold.nii.gz                     High-kappa time series. This dataset does not
                                                    include thermal noise or low variance components.
                                                    Not the recommended dataset for analysis.
desc-adaptiveGoodSignal_mask.nii.gz                 Integer-valued mask used in the workflow, where
                                                    each voxel's value corresponds to the number of good
                                                    echoes to be used for T2\*/S0 estimation.
desc-PCA_mixing.tsv                                 Mixing matrix (component time series) from PCA
                                                    decomposition in a tab-delimited file. Each column is
                                                    a different component, and the column name is the
                                                    component number.
desc-PCA_decomposition.json                         Metadata for the PCA decomposition.
desc-PCA_stat-z_components.nii.gz                   Component weight maps from PCA decomposition.
                                                    Each map corresponds to the same component index in
                                                    the mixing matrix and component table.
                                                    Maps are in z-statistics.
desc-PCA_metrics.tsv                                TEDPCA component table. A BIDS Derivatives-compatible
                                                    TSV file with summary metrics and inclusion/exclusion
                                                    information for each component from the PCA
                                                    decomposition.
desc-PCA_metrics.json                               Metadata about the metrics in ``desc-PCA_metrics.tsv``.
desc-ICA_mixing.tsv                                 Mixing matrix (component time series) from ICA
                                                    decomposition in a tab-delimited file. Each column is
                                                    a different component, and the column name is the
                                                    component number.
desc-ICA_components.nii.gz                          Full ICA coefficient feature set.
desc-ICA_stat-z_components.nii.gz                   Z-statistic component weight maps from ICA
                                                    decomposition.
                                                    Values are z-transformed standardized regression
                                                    coefficients. Each map corresponds to the same
                                                    component index in the mixing matrix and component table.
desc-ICA_decomposition.json                         Metadata for the ICA decomposition.
desc-tedana_metrics.tsv                             TEDICA component table. A BIDS Derivatives-compatible
                                                    TSV file with summary metrics and inclusion/exclusion
                                                    information for each component from the ICA
                                                    decomposition.
desc-tedana_metrics.json                            Metadata about the metrics in
                                                    ``desc-tedana_metrics.tsv``.
desc-ICAAccepted_components.nii.gz                  High-kappa ICA coefficient feature set
desc-ICAAcceptedZ_components.nii.gz                 Z-normalized spatial component maps
report.txt                                          A summary report for the workflow with relevant
                                                    citations.
references.bib                                      The BibTeX entries for references cited in
                                                    report.txt.
tedana_report.html                                  The interactive HTML report.
================================================    =====================================================

If ``verbose`` is set to True:

==============================================================  =====================================================
Filename                                                        Content
==============================================================  =====================================================
desc-limited_T2starmap.nii.gz                                   Limited T2* map/time series.
                                                                Values are in seconds.
                                                                The difference between the limited and full maps
                                                                is that, for voxels affected by dropout where
                                                                only one echo contains good data, the full map uses
                                                                the S0 estimate from the first two echoes, while the
                                                                limited map has a NaN.
desc-limited_S0map.nii.gz                                       Limited S0 map/time series.
                                                                The difference between the limited and full maps
                                                                is that, for voxels affected by dropout where
                                                                only one echo contains good data, the full map uses
                                                                the S0 estimate from the first two echoes, while the
                                                                limited map has a NaN.
echo-[echo]_desc-[PCA|ICA]_components.nii.gz                    Echo-wise PCA/ICA component weight maps.
echo-[echo]_desc-[PCA|ICA]R2ModelPredictions_components.nii.gz  Component- and voxel-wise R2-model predictions,
                                                                separated by echo.
echo-[echo]_desc-[PCA|ICA]S0ModelPredictions_components.nii.gz  Component- and voxel-wise S0-model predictions,
                                                                separated by echo.
desc-[PCA|ICA]AveragingWeights_components.nii.gz                Component-wise averaging weights for metric
                                                                calculation.
desc-[PCA|ICA]S0_stat-F_statmap.nii.gz                          F-statistic map for each component, for the S0 model.
desc-[PCA|ICA]T2_stat-F_statmap.nii.gz                          F-statistic map for each component, for the T2 model.
desc-optcomPCAReduced_bold.nii.gz                               Optimally combined data after dimensionality
                                                                reduction with PCA. This is the input to the ICA.
echo-[echo]_desc-Accepted_bold.nii.gz                           High-Kappa time series for echo number ``echo``
echo-[echo]_desc-Rejected_bold.nii.gz                           Low-Kappa time series for echo number ``echo``
echo-[echo]_desc-Denoised_bold.nii.gz                           Denoised time series for echo number ``echo``
==============================================================  =====================================================

If ``gscontrol`` includes 'gsr':

================================================    =====================================================
Filename                                            Content
================================================    =====================================================
desc-globalSignal_map.nii.gz                        Spatial global signal
desc-globalSignal_timeseries.tsv                    Time series of global signal from optimally combined
                                                    data.
desc-optcomWithGlobalSignal_bold.nii.gz             Optimally combined time series with global signal
                                                    retained.
desc-optcomNoGlobalSignal_bold.nii.gz               Optimally combined time series with global signal
                                                    removed.
================================================    =====================================================

If ``gscontrol`` includes 't1c':

================================================    =====================================================
Filename                                            Content
================================================    =====================================================
desc-T1likeEffect_min.nii.gz                        T1-like effect
desc-optcomAcceptedT1cDenoised_bold.nii.gz          T1-corrected high-kappa time series by regression
desc-optcomT1cDenoised_bold.nii.gz                  T1-corrected denoised time series
desc-TEDICAAcceptedT1cDenoised_components.nii.gz    T1-GS corrected high-kappa components
desc-TEDICAT1cDenoised_mixing.tsv                   T1-GS corrected mixing matrix
================================================    =====================================================


****************
Component tables
****************

TEDPCA and TEDICA use component tables to track relevant metrics, component
classifications, and rationales behind classifications.
The component tables are stored as tsv files for BIDS-compatibility.

In order to make sense of the rationale codes in the component tables,
consult the tables below.
TEDPCA rationale codes start with a "P", while TEDICA codes start with an "I".

===============    =============================================================
Classification     Description
===============    =============================================================
accepted           BOLD-like components included in denoised and high-Kappa data
rejected           Non-BOLD components excluded from denoised and high-Kappa data
ignored            Low-variance components included in denoised, but excluded
                   from high-Kappa data
===============    =============================================================


TEDPCA codes
============

=====  ===============  ========================================================
Code   Classification   Description
=====  ===============  ========================================================
P001   rejected         Low Rho, Kappa, and variance explained
P002   rejected         Low variance explained
P003   rejected         Kappa equals fmax
P004   rejected         Rho equals fmax
P005   rejected         Cumulative variance explained above 95% (only in
                        stabilized PCA decision tree)
P006   rejected         Kappa below fmin (only in stabilized PCA decision tree)
P007   rejected         Rho below fmin (only in stabilized PCA decision tree)
=====  ===============  ========================================================


TEDICA codes
============

=====  =================  ========================================================
Code   Classification     Description
=====  =================  ========================================================
I001   rejected|accepted  Manual classification
I002   rejected           Rho greater than Kappa
I003   rejected           More significant voxels in S0 model than R2 model
I004   rejected           S0 Dice is higher than R2 Dice and high variance
                          explained
I005   rejected           Noise F-value is higher than signal F-value and high
                          variance explained
I006   ignored            No good components found
I007   rejected           Mid-Kappa component
I008   ignored            Low variance explained
I009   rejected           Mid-Kappa artifact type A
I010   rejected           Mid-Kappa artifact type B
I011   ignored            ign_add0
I012   ignored            ign_add1
=====  =================  ========================================================

.. _interactive reports:

*********************
ICA Components Report
*********************

The reporting page for the tedana decomposition presents a series
of interactive plots designed to help you evaluate the quality of your
analyses. This page describes the plots forming the reports and well as
information on how to take advantage of the interactive functionalities.
You can also play around with `our demo`_.

.. _our demo: https://me-ica.github.io/tedana-ohbm-2020/


Report Structure
================

The image below shows a representative report, which has two sections: a) the summary view,
and b) the individual component view.

.. image:: /_static/rep01_overallview.png
  :align: center

.. note::
  When a report is initially loaded, as no component is selected on the
  summary view, the individual component view appears empty.


Summary View
------------

This view provides an overview of the decomposition and component
selection results. It includes four different plots.

* **Kappa/Rho Scatter:** This is a scatter plot of `Kappa` vs. `Rho` features for all components.
  In the plot, each dot represents a different component. The x-axis represents the kappa feature, and the
  y-axis represents the rho feature. These are two of the most
  informative features describing the likelihood of the component
  being BOLD or non-BOLD. Additional information is provided via color
  and size. In particular, color informs about its classification
  status (e.g., accepted, rejected); while size relates to
  the amount of variance explained by the component (larger dot,
  larger variance).

.. image:: /_static/rep01_kapparhoScatter.png
  :align: center
  :height: 400px

* **Kappa Scree Plot:** This scree plot provides a view of the components ranked by `Kappa`.
  As in the previous plot, each dot represents a component. The color of the dot informs us
  about classification status. In this plot, size is not related to variance explained.

.. image:: /_static/rep01_kappaScree.png
  :align: center
  :height: 400px

* **Rho Scree Plot:** This scree plot provides a view of the components ranked by `Rho`.
  As in the previous plot, each dot represents a component. The color of the dot informs us
  about classification status. In this plot, size is not related to variance explained.

.. image:: /_static/rep01_rhoScree.png
  :align: center
  :height: 400px

* **Variance Explained Plot:** This pie plot provides a summary of how much variance is explained
  by each individual component, as well as the total variance explained by each of the three
  classification categories (i.e., accepted, rejected, ignored). In this plot, each component is
  represented as a wedge, whose size is directly related to the amount of variance explained. The
  color of the wedge inform us about the classification status of the component. For this view,
  components are sorted by classification first, and inside each classification group by variance
  explained.

.. image:: /_static/rep01_varexpPie.png
  :align: center
  :height: 400px


Individual Component View
-------------------------

This view provides detailed information about an individual
component (selected in the summary view, see below). It includes three different plots.

* **Time series:** This plot shows the time series associated with a given component
  (selected in the summary view). The x-axis represents time (in units of TR), and the
  y-axis represents signal levels (in arbitrary units). Finally, the color of the trace
  informs us about the component classification status.

.. image:: /_static/rep01_tsPlot.png
  :align: center
  :height: 150px

* **Component beta map:** This plot shows the map of the beta coefficients associated with
  a given component (selected in the summary view). The colorbar represents the amplitude
  of the beta coefficients.

.. image:: /_static/rep01_betaMap.png
  :align: center
  :height: 400px

* **Spectrum:** This plot shows the spectrogram associated with a given component
  (selected in the summary view). The x-axis represents frequency (in Hz), and the
  y-axis represents spectral amplitude.

.. image:: /_static/rep01_fftPlot.png
  :align: center
  :height: 150px


Reports User Interactions
=========================

As previously mentioned, all summary plots in the report allow user interactions. While the
Kappa/Rho Scatter plot allows full user interaction (see the toolbar that accompanies the plot
and the example below), the other three plots allow the user to select components and update the
figures.

.. image:: /_static/rep01_tools.png
  :align: center
  :height: 25px

The table below includes information about all available interactions

.. |Reset| image:: /_static/rep01_tool_reset.png
  :height: 25px

.. |WZoom| image:: /_static/rep01_tool_wheelzoom.png
  :height: 25px

.. |BZoom| image:: /_static/rep01_tool_areazoom.png
  :height: 25px

.. |CHair| image:: /_static/rep01_tool_crosshair.png
  :height: 25px

.. |Pan| image:: /_static/rep01_tool_pan.png
  :height: 25px

.. |Hover| image:: /_static/rep01_tool_hover.png
  :height: 25px

.. |Sel| image:: /_static/rep01_tool_select.png
  :height: 25px

.. |Save| image:: /_static/rep01_tool_save.png
  :height: 25px

============  =======  =======================================================
Interaction   Icon     Description
============  =======  =======================================================
Reset         |Reset|  Resets the data bounds of the plot to their values when
                       the plot was initially created.

Wheel Zoom    |WZoom|  Zoom the plot in and out, centered on the current
                       mouse location.

Box Zoom      |BZoom|  Define a rectangular region of a plot to zoom to by
                       dragging the mouse over the plot region.

Crosshair     |CHair|  Draws a crosshair annotation over the plot, centered on
                       the current mouse position

Pan           |Pan|    Allows the user to pan a plot by left-dragging a mouse
                       across the plot region.

Hover         |Hover|  If active, the plot displays informational tooltips
                       whenever the cursor is directly over a plot element.

Selection     |Sel|    Allows user to select components by tapping on the dot
                       or wedge that represents them. Once a component is
                       selected, the plots forming the individual component
                       view update to show component specific information.

Save          |Save|   Saves an image reproduction of the plot in PNG format.
============  =======  =======================================================

.. note::
  Specific user interactions can be switched on/off by clicking on their associated icon within
  the toolbar of a given plot. Active interactions show an horizontal blue line underneath their
  icon, while inactive ones lack the line.


************
Carpet plots
************

In additional to the elements described above, ``tedana``'s interactive reports include carpet plots for the main outputs of the workflow:
the optimally combined data, the denoised data, the high-Kappa (accepted) data, and the low-Kappa (rejected) data.

These plots may be useful for visual quality control of the overall denoising run.

.. image:: /_static/carpet_overview.png
  :align: center
  :height: 400px


**************************
Citable workflow summaries
**************************

``tedana`` generates a report for the workflow, customized based on the parameters used and including relevant citations.
The report is saved in a plain-text file, report.txt, in the output directory.

An example report

  .. note::

    The boilerplate text includes citations in LaTeX format.
    \\citep refers to parenthetical citations, while \\cite refers to textual ones.

  TE-dependence analysis was performed on input data using the tedana workflow \\citep{dupre2021te}.
  An adaptive mask was then generated, in which each voxel's value reflects the number of echoes with 'good' data.
  A two-stage masking procedure was applied, in which a liberal mask (including voxels with good data in at least the first echo) was used for optimal combination, T2*/S0 estimation, and denoising, while a more conservative mask (restricted to voxels with good data in at least the first three echoes) was used for the component classification procedure.
  Multi-echo data were then optimally combined using the T2* combination method \\citep{posse1999enhancement}.
  Next, components were manually classified as BOLD (TE-dependent), non-BOLD (TE-independent), or uncertain (low-variance).
  This workflow used numpy \\citep{van2011numpy}, scipy \\citep{virtanen2020scipy}, pandas \\citep{mckinney2010data,reback2020pandas}, scikit-learn \\citep{pedregosa2011scikit}, nilearn, bokeh \\citep{bokehmanual}, matplotlib \\citep{Hunter:2007}, and nibabel \\citep{brett_matthew_2019_3233118}.
  This workflow also used the Dice similarity index \\citep{dice1945measures,sorensen1948method}.

  References

  .. note::

    The references are also provided in the ``references.bib`` output file.

  .. code-block:: bibtex

    @Manual{bokehmanual,
        title = {Bokeh: Python library for interactive visualization},
        author = {{Bokeh Development Team}},
        year = {2018},
        url = {https://bokeh.pydata.org/en/latest/},
    }
    @article{dice1945measures,
        title={Measures of the amount of ecologic association between species},
        author={Dice, Lee R},
        journal={Ecology},
        volume={26},
        number={3},
        pages={297--302},
        year={1945},
        publisher={JSTOR},
        url={https://doi.org/10.2307/1932409},
        doi={10.2307/1932409}
    }
    @article{dupre2021te,
        title={TE-dependent analysis of multi-echo fMRI with* tedana},
        author={DuPre, Elizabeth and Salo, Taylor and Ahmed, Zaki and Bandettini, Peter A and Bottenhorn, Katherine L and Caballero-Gaudes, C{\'e}sar and Dowdle, Logan T and Gonzalez-Castillo, Javier and Heunis, Stephan and Kundu, Prantik and others},
        journal={Journal of Open Source Software},
        volume={6},
        number={66},
        pages={3669},
        year={2021},
        url={https://doi.org/10.21105/joss.03669},
        doi={10.21105/joss.03669}
    }
    @inproceedings{mckinney2010data,
        title={Data structures for statistical computing in python},
        author={McKinney, Wes and others},
        booktitle={Proceedings of the 9th Python in Science Conference},
        volume={445},
        number={1},
        pages={51--56},
        year={2010},
        organization={Austin, TX},
        url={https://doi.org/10.25080/Majora-92bf1922-00a},
        doi={10.25080/Majora-92bf1922-00a}
    }
    @article{pedregosa2011scikit,
        title={Scikit-learn: Machine learning in Python},
        author={Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
        journal={the Journal of machine Learning research},
        volume={12},
        pages={2825--2830},
        year={2011},
        publisher={JMLR. org},
        url={http://jmlr.org/papers/v12/pedregosa11a.html}
    }
    @article{posse1999enhancement,
        title={Enhancement of BOLD-contrast sensitivity by single-shot multi-echo functional MR imaging},
        author={Posse, Stefan and Wiese, Stefan and Gembris, Daniel and Mathiak, Klaus and Kessler, Christoph and Grosse-Ruyken, Maria-Liisa and Elghahwagi, Barbara and Richards, Todd and Dager, Stephen R and Kiselev, Valerij G},
        journal={Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine},
        volume={42},
        number={1},
        pages={87--97},
        year={1999},
        publisher={Wiley Online Library},
        url={https://doi.org/10.1002/(SICI)1522-2594(199907)42:1<87::AID-MRM13>3.0.CO;2-O},
        doi={10.1002/(SICI)1522-2594(199907)42:1<87::AID-MRM13>3.0.CO;2-O}
    }
    @software{reback2020pandas,
        author = {The pandas development team},
        title = {pandas-dev/pandas: Pandas},
        month = feb,
        year = 2020,
        publisher = {Zenodo},
        version = {latest},
        doi = {10.5281/zenodo.3509134},
        url = {https://doi.org/10.5281/zenodo.3509134}
    }
    @article{sorensen1948method,
        title={A method of establishing groups of equal amplitude in plant sociology based on similarity of species content and its application to analyses of the vegetation on Danish commons},
        author={Sorensen, Th A},
        journal={Biol. Skar.},
        volume={5},
        pages={1--34},
        year={1948}
    }
    @article{van2011numpy,
        title={The NumPy array: a structure for efficient numerical computation},
        author={Van Der Walt, Stefan and Colbert, S Chris and Varoquaux, Gael},
        journal={Computing in science \& engineering},
        volume={13},
        number={2},
        pages={22--30},
        year={2011},
        publisher={IEEE},
        url={https://doi.org/10.1109/MCSE.2011.37},
        doi={10.1109/MCSE.2011.37}
    }
    @article{virtanen2020scipy,
        title={SciPy 1.0: fundamental algorithms for scientific computing in Python},
        author={Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E and Haberland, Matt and Reddy, Tyler and Cournapeau, David and Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and Bright, Jonathan and others},
        journal={Nature methods},
        volume={17},
        number={3},
        pages={261--272},
        year={2020},
        publisher={Nature Publishing Group},
        url={https://doi.org/10.1038/s41592-019-0686-2},
        doi={10.1038/s41592-019-0686-2}
    }
