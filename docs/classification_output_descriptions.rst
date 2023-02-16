##################################
Classification output descriptions
##################################

Tedana outputs multiple files that can be used to subsequent analyses and to
better understand one's denoising results.
In addition to the `descriptions of file names`_ this page explains the
contents of several of those files in more detail.
`Building decision trees`_ covers the full process, and not just the
descriptions of outputted files, in more detail.

.. _Building decision trees: building\ decision\ trees.html

TEDPCA codes
============

In ``tedana`` PCA is used to reduce the number of dimensions (components) in the
dataset. Without this step, the number of components would be one less than
the number of volumes, many of those components would effectively be
Gaussian noise and ICA would not reliably converge. Standard methods for data
reduction use cost functions, like MDL, KIC, and AIC to estimate the variance
that is just noise and remove the lowest variance components under that threshold.
By default, ``tedana`` uses AIC. Of those three, AIC is the least agressive and
will retain the most components.

``Tedana`` includes additional `kundu` and `kundu-stabilize` approaches that
identify and remove components that don't contain T2* or S0 signal and are more
likely to be noise. If the `--tedpca kundu` option is used, the PCA_metrics tsv
file will include an accepted vs rejected classification column and also a
rationale column of codes documenting why a PCA component removed. If MDL, KIC,
or AIC are used then the classification column will exist, but will include
include the accepted components and the rationale column will contain n/a"
When kundu is used, these are brief explanations of the the rationale codes

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


ICA Classification Outputs
==========================

The component table is stored in ``desc-tedana_metrics.tsv`` or
``tedana_metrics.tsv``. Each row is a component number. Each column is a metric
that is calculated for each component. Short descriptions of each column metric
are in the output log, ``tedana_[date_time].tsv``, and the actual metric
calculations are in `collect.py`_ The final two columns are `classification`
and `classification_tags`. `classification` should include `accepted` or
`rejected` for every component and `rejected` components are be removed
through denoising. `classification_tags` provide more information on why
components received a specific classification. Each component can receive
more than one tag. The following tags are included depending if ``--tree``
is minimal, kundu, or if ``ica_reclassify`` is run.

===================== ================  ========================================
Tag                   Included in Tree  Explanation
===================== ================  ========================================
Likely BOLD           minimal,kundu     Accepted because likely to include some
                                        BOLD signal
Unlikely BOLD         minimal,kundu     Rejected because likely to include a
                                        lot of non-BOLD signal
Low variance          minimal,kundu     Accepted because too low variance to
                                        lose a degree-of-freedom by rejecting
Less likely BOLD      kundu             Rejected based on some edge criteria
                                        based on relative rankings of components
Accept borderline     kundu             Accepted based on some edge criteria
                                        based on relative rankings of components
No provisional accept kundu             Accepted because because kundu tree did
                                        not find any components to consider
                                        accepting so the conservative "failure"
                                        case is accept everything rather than
                                        rejecting everything
manual reclassify     manual_classify   Classification based on user input. If
                                        done after automatic selection then
                                        the preceding tag from automatic
                                        selection is retained and this tag
                                        notes the classification was manually
                                        changed
===================== ================  ========================================

The decision tree is a list of nodes where the classification of each component
could change. The information on which nodes and how classifications changed is
in several places:

- The information in the output log includes the name of each
  node and the count of components that changed classification during execution.
- The same information is stored in the `ICA decision tree` json file (see
  `descriptions of file names`_) in the "output" field for each node. That information
  is organized so that it can be used to generate a visual or text-based summary of
  what happened when the decision tree was run on a dataset.
- The `ICA status table` lists the classification status of each component after
  each node was run. This is particularly useful to trying to understand how a
  specific component ended receiving its classification.

.. _collect.py: https://github.com/ME-ICA/tedana/blob/main/tedana/metrics/collect.py
.. _descriptions of file names: output_file_descriptions.html