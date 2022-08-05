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

   tedana.workflows.tedana_workflow
   tedana.workflows.t2smap_workflow


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

   tedana.decay.fit_decay
   tedana.decay.fit_decay_ts


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

   tedana.combine.make_optcom


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

   tedana.decomposition.tedpca
   tedana.decomposition.tedica


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

   tedana.metrics.collect
   tedana.metrics.dependence


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
   :template: function.rst

   tedana.selection.manual_selection
   tedana.selection.kundu_selection_v2
   tedana.selection.kundu_tedpca


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

   tedana.gscontrol.gscontrol_raw
   tedana.gscontrol.minimum_image_regression


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

   tedana.io.OutputGenerator

   :template: function.rst

   tedana.io.load_data
   tedana.io.new_nii_like
   tedana.io.add_decomp_prefix
   tedana.io.denoise_ts
   tedana.io.split_ts
   tedana.io.write_split_ts
   tedana.io.writeresults
   tedana.io.writeresults_echoes


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

   tedana.stats.get_coeffs
   tedana.stats.computefeats2
   tedana.stats.getfbounds


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

   tedana.bibtex.


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

   tedana.utils.andb
   tedana.utils.dice
   tedana.utils.get_spectrum
   tedana.utils.reshape_niimg
   tedana.utils.make_adaptive_mask
   tedana.utils.threshold_map
   tedana.utils.unmask
   tedana.utils.sec2millisec
   tedana.utils.millisec2sec
