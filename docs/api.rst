.. _api_ref:

.. currentmodule:: tedana

###
API
###


.. _api_workflows_ref:

*****************************************
:mod:`tedana.workflows`: Common workflows
*****************************************

.. automodule:: tedana.workflows
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.workflows

.. autosummary::
   :template: function.rst
   :toctree: generated/

   tedana_workflow
   ica_reclassify_workflow
   t2smap_workflow
   parser_utils.check_tedpca_value
   parser_utils.check_n_robust_runs_value
   parser_utils.is_valid_file
   parser_utils.parse_manual_list_int
   parser_utils.parse_manual_list_str


.. _api_decay_ref:

********************************************************
:mod:`tedana.decay`: Modeling signal decay across echoes
********************************************************

.. automodule:: tedana.decay
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.decay

.. autosummary::
   :template: function.rst
   :toctree: generated/

   fit_decay
   fit_decay_ts
   monoexponential
   fit_monoexponential
   fit_loglinear
   modify_t2s_s0_maps
   rmse_of_fit_decay_ts


.. _api_combine_ref:

**********************************************************
:mod:`tedana.combine`: Combining time series across echoes
**********************************************************

.. automodule:: tedana.combine
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.combine

.. autosummary::
   :toctree: generated/
   :template: function.rst

   make_optcom


.. _api_decomposition_ref:

***********************************************
:mod:`tedana.decomposition`: Data decomposition
***********************************************

.. automodule:: tedana.decomposition
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.decomposition

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tedpca
   tedica
   ica.r_ica
   ica.f_ica


.. _api_metrics_ref:

******************************************************
:mod:`tedana.metrics`: Computing TE-dependence metrics
******************************************************

.. automodule:: tedana.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.metrics

.. autosummary::
   :toctree: generated/
   :template: module.rst

   collect
   dependence
   external


.. _api_selection_ref:

********************************************
:mod:`tedana.selection`: Component selection
********************************************

.. automodule:: tedana.selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.selection

.. autosummary::
   :toctree: generated/
   :template: class.rst

   component_selector.ComponentSelector
   component_selector.TreeError

   :template: function.rst

   component_selector.load_config
   component_selector.validate_tree

.. autosummary::
   :toctree: generated/
   :template: module.rst

   selection_nodes
   selection_utils
   tedica
   tedpca


.. _api_gscontrol_ref:

**********************************************
:mod:`tedana.gscontrol`: Global signal control
**********************************************

.. automodule:: tedana.gscontrol
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.gscontrol

.. autosummary::
   :toctree: generated/
   :template: function.rst

   gscontrol_raw
   minimum_image_regression


.. _api_io_ref:

******************************************
:mod:`tedana.io`: Reading and writing data
******************************************

.. automodule:: tedana.io
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.io

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OutputGenerator
   InputHarvester
   CustomEncoder

   :template: function.rst

   load_data_nilearn
   load_json
   get_fields
   prep_data_for_json
   add_decomp_prefix
   denoise_ts
   split_ts
   write_split_ts
   writeresults
   writeresults_echoes
   download_json
   load_ref_img
   versiontuple
   str_to_component_list
   fname_to_component_list


.. _api_reporting_ref:

********************************************
:mod:`tedana.reporting`: Reporting functions
********************************************

.. automodule:: tedana.reporting
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.reporting

.. autosummary::
   :toctree: generated/
   :template: function.rst

   html_report.generate_report
   quality_metrics.calculate_rejected_components_impact
   static_figures.comp_figures
   static_figures.pca_results
   static_figures.plot_t2star_and_s0
   static_figures.plot_rmse
   static_figures.plot_adaptive_mask
   static_figures.carpet_plot
   static_figures.plot_component
   static_figures.plot_gscontrol
   static_figures.plot_heatmap
   static_figures.plot_decay_variance


.. _api_stats_ref:

******************************************
:mod:`tedana.stats`: Statistical functions
******************************************

.. automodule:: tedana.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.stats

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_coeffs
   voxelwise_univariate_zstats
   getfbounds
   fit_model
   t_to_z


.. _api_bibtex_ref:

*********************************************************
:mod:`tedana.bibtex`: Tools for working with BibTeX files
*********************************************************

.. automodule:: tedana.bibtex
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.bibtex

.. autosummary::
   :toctree: generated/
   :template: function.rst

   find_braces
   reduce_idx
   index_bibtex_identifiers
   find_citations
   reduce_references
   get_description_references


.. _api_utils_ref:

**************************************
:mod:`tedana.utils`: Utility functions
**************************************

.. automodule:: tedana.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: tedana.utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   andb
   dice
   get_spectrum
   make_adaptive_mask
   threshold_map
   unmask
   sec2millisec
   millisec2sec
   load_mask
   create_legendre_polynomial_basis_set
   parse_volume_indices
   check_t2s_values
   check_te_values
   setup_loggers
   teardown_loggers
   get_resource_path
   get_system_version_info
   log_newsletter_info
