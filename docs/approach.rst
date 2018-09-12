tedana's approach
=================

``tedana`` works by decomposing multi-echo BOLD data via PCA and ICA.
These components are then analyzed to determine whether they are TE-dependent
or -independent. TE-dependent components are classified as BOLD, while
TE-independent components are classified as non-BOLD, and are discarded as part
of data cleaning.

In ``tedana``, we take the time series from all the collected TEs, combine them,
and decompose the resulting data into components that can be classified as BOLD
or non-BOLD. This is performed in a series of steps including:

* Principal components analysis
* Independent components analysis
* Component classification

The tedana workflow
-------------------

.. image:: /_static/tedana-workflow.png
