API
===

:mod:`tedana.workflows`: Common workflows
--------------------------------------------------

.. automodule:: tedana.workflows
   :no-members:
   :no-inherited-members:

.. autosummary:: tedana.workflows
   :template: function.rst
   :toctree: generated/

   tedana.workflows.tedana_workflow
   tedana.workflows.t2smap_workflow

.. currentmodule:: tedana

.. _calibration_ref:


:mod:`tedana.decay`: Modeling signal decay across echoes
--------------------------------------------------------

.. automodule:: tedana.decay
   :no-members:
   :no-inherited-members:

.. autosummary:: tedana.decay
   :template: function.rst
   :toctree: generated/

   tedana.decay.fit_decay
   tedana.decay.fit_decay_ts

.. currentmodule:: tedana

.. _calibration_ref:


:mod:`tedana.combine`: Combining time series across echoes
----------------------------------------------------------

.. automodule:: tedana.combine
   :no-members:
   :no-inherited-members:

.. autosummary:: tedana.combine
   :toctree: generated/
   :template: function.rst

   tedana.combine.make_optcom

.. currentmodule:: tedana

.. _calibration_ref:


:mod:`tedana.decomposition`: Data decomposition
--------------------------------------------------

.. automodule:: tedana.decomposition
   :no-members:
   :no-inherited-members:

.. autosummary:: tedana.decomposition
   :toctree: generated/
   :template: function.rst

   tedana.decomposition.tedpca
   tedana.decomposition.tedica

   :template: module.rst

   tedana.decomposition._utils

.. currentmodule:: tedana

.. _calibration_ref:


:mod:`tedana.model`: Computing TE-dependence metrics
----------------------------------------------------

.. automodule:: tedana.model
   :no-members:
   :no-inherited-members:

.. autosummary:: tedana.model
   :toctree: generated/
   :template: function.rst

   tedana.model.dependence_metrics
   tedana.model.kundu_metrics

   :template: module.rst

   tedana.model.fit

.. currentmodule:: tedana

.. _calibration_ref:


:mod:`tedana.selection`: Component selection
--------------------------------------------------

.. automodule:: tedana.selection
   :no-members:
   :no-inherited-members:

.. autosummary:: tedana.selection
   :toctree: generated/
   :template: function.rst

   tedana.selection.manual_selection
   tedana.selection.kundu_selection_v2

   :template: module.rst

   tedana.selection._utils

.. currentmodule:: tedana

.. _calibration_ref:


:mod:`tedana.gscontrol`: Global signal control
--------------------------------------------------

.. automodule:: tedana.gscontrol
   :no-members:
   :no-inherited-members:

.. autosummary:: tedana.gscontrol
   :toctree: generated/
   :template: function.rst

   tedana.gscontrol.gscontrol_raw
   tedana.gscontrol.gscontrol_mmix

.. currentmodule:: tedana

.. _calibration_ref:


:mod:`tedana.io`: Reading and writing data
--------------------------------------------------

.. automodule:: tedana.io
   :no-members:
   :no-inherited-members:

.. autosummary:: tedana.io
   :toctree: generated/

   :template: function.rst

   tedana.io.split_ts
   tedana.io.filewrite
   tedana.io.load_data
   tedana.io.new_nii_like
   tedana.io.write_split_ts
   tedana.io.writefeats
   tedana.io.writeresults
   tedana.io.writeresults_echoes

.. currentmodule:: tedana

.. _calibration_ref:


:mod:`tedana.utils`: Utility functions
--------------------------------------------------

.. automodule:: tedana.utils
   :no-members:
   :no-inherited-members:

.. autosummary:: tedana.utils
   :toctree: generated/

   :template: function.rst

   tedana.utils.andb
   tedana.utils.dice
   tedana.utils.getfbounds
   tedana.utils.load_image
   tedana.utils.make_adaptive_mask
   tedana.utils.unmask

.. currentmodule:: tedana

.. _calibration_ref:
