.. include:: <isonum.txt>


tedana: TE Dependent ANAlysis
=============================

The ``tedana`` package is part of the ME-ICA pipeline, performing TE-dependent
analysis of multi-echo functional magnetic resonance imaging (fMRI) data.

.. image:: https://circleci.com/gh/ME-ICA/tedana.svg?style=svg
    :target: https://circleci.com/gh/ME-ICA/tedana

.. image:: http://img.shields.io/badge/License-LGPL%202.0-blue.svg
   :target: https://opensource.org/licenses/LGPL-2.1

Citations
---------

When using tedana, please include the following citations:

  Kundu, P., Inati, S. J., Evans, J. W., Luh, W. M. & Bandettini, P. A. (2011).
  `Differentiating BOLD and non-BOLD signals in fMRI time series using multi-echo EPI.`_
  *NeuroImage*, *60*, 1759-1770.

  Kundu, P., Brenowitz, N. D., Voon, V., Worbe, Y., Vértes, P. E., Inati, S. J.,
  Saad, Z. S., Bandettini, P. A., & Bullmore, E. T. (2013).
  `Integrated strategy for improving functional connectivity mapping using multiecho fMRI.`_
  *Proceedings of the National Academy of Sciences*, 110(40), 16187-16192.

  DuPre, E. M., Salo, T., Markello, R. D., Kundu, P., Whitaker K. J. (2018).
  `ME-ICA/tedana: Initial tedana release (Version 0.0.1).`_
  *Zenodo*. doi:10.5281/zenodo.1250562.

Alternatively, you can automatically compile relevant citations by running your
tedana code with `duecredit`_. For example, if you plan to run a script using
tedana (in this case, ``tedana_script.py``):

.. code-block:: bash

  python -m duecredit tedana_script.py

.. _Differentiating BOLD and non-BOLD signals in fMRI time series using multi-echo EPI.: http://dx.doi.org/10.1016/j.neuroimage.2011.12.028
.. _Integrated strategy for improving functional connectivity mapping using multiecho fMRI.: http://dx.doi.org/10.1073/pnas.1301725110
.. _ME-ICA/tedana\: Initial tedana release (Version 0.0.1).: http://dx.doi.org/10.5281/zenodo.1250561
.. _duecredit: https://github.com/duecredit/duecredit

License Information
-------------------

tedana is licensed under GNU Lesser General Public License version 2.1.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   usage
   api
   contributing


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
