#####################
ICA Components Report
#####################

The reporting page for the tedana decomposition presents a series
of interactive plots designed to help you evaluate the quality of your
analyses. Here you can find both a description of the plots and infromation
on how to take advantage of the interactive functionalities.

Report Structure
================

The image below shows a representative report, which has two sections:

.. image:: /_static/rep01_overallview.png
  :align: center
Summary View
------------
This view provides an overview of the decomposition and component
selection results. It includes four different plots.

* **Kappa/Rho Scatter:** This is a scatter plot of `Kappa` vs. `Rho` features for all components.
  In the plot, each dot represents a different component. The x-axis represents the kappa feature, and the
  y-axis represents the rho feature. These are two of the most
  informative features describing the likelihood of the component
  being BOLD or non-BOLD. Additional information is provided via color
  and size. In particuale, color informs about its classification
  status (e.g., accepted, rejected); while size relates to
  the amount of variance explained by the component (larger dot,
  larger variance).
.. image:: /_static/rep01_kapparhoScatter.png
  :align: center
  :height: 400px

* **Kappa Scree Plot:** This scree plots provides a view of the components ranked by `Kappa`
  As in the previous plot, each dot represents a component. The color of the dot inform us
  about classification status. In this plot size is not related to variance explained.
.. image:: /_static/rep01_kappaScree.png
  :align: center
  :height: 400px

* **Rho Scree Plot:** This scree plots provides a view of the components ranked by `Rho`
  As in the previous plot, each dot represents a component. The color of the dot inform us
  about classification status. In this plot size is not related to variance explained.
.. image:: /_static/rep01_rhoScree.png
  :align: center
  :height: 400px

* **Variance Explained Plot:** This pie plot provides a summary of how much variance is explained
  by each individual component, as well as, the total variance explained by each of the three
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
component (selected in the summary view, see below). It includes three different
interactive plots.

* **Time series:** This plot shows the time series associated with a given component
  (selected in the summary view). The x-axis represents time (in units of TR), and the
  y-axis represents signal levels (in arbitrary units). Finally, the color of the trace
  informs us about the component classification status.

.. image:: /_static/rep01_tsPlot.png
  :align: center
  :height: 150px

* **Spectrum:** This plot shows the spectrogram associated with a given component
  (selected in the summary view). The x-axis represents frequency (in Hz), and the
  y-axis represents spectral amplitude? Finally, the color of the trace
  informs us about the component classification status.

.. image:: /_static/rep01_fftPlot.png
  :align: center
  :height: 150px

.. note::
  When a report is initially loaded, as no component is selected on the
  summary view, the individual component view appears empty.

Reports User Interactions
=========================

As previously mentioned, all plots in the report allow user interactions. The list of permitted
interactions vary by plot, but can be easily infered by the toolbar that accompanies each plot
(see example below).

.. image:: /_static/rep01_tools.png
  :align: center
  :height: 25px

The table below includes information about all available interactions

.. |Reset Logo| image:: /_static/rep01_tool_reset.png
  :align: center
  :height: 25px
  :alt: Reset

============  ==================  ============= =============
Interaction   Icon                Description   Available at
============  ==================  ============= =============
Reset         

Wheel Zoom    |Reset Logo|
============  ==================  ============= =============



Specific user interactions can be swithed on/off by clicking on their associated icon within
the toolbar of a given plot. Active interactions show an horizontal blue line underneath their
icon, while inactive ones lack the line.






