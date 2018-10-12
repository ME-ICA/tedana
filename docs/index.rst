.. include:: <isonum.txt>

tedana: TE Dependent ANAlysis
=============================

The ``tedana`` package is part of the ME-ICA pipeline, performing TE-dependent
analysis of multi-echo functional magnetic resonance imaging (fMRI) data.
``TE``-``de``\pendent ``ana``\lysis (``tedana``) is a Python module for denoising
multi-echo functional magnetic resonance imaging (fMRI) data.

.. image:: https://img.shields.io/pypi/v/tedana.svg
   :target: https://pypi.python.org/pypi/tedana/
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/tedana.svg
   :target: https://pypi.python.org/pypi/tedana/
   :alt: PyPI - Python Version

.. image:: https://zenodo.org/badge/110845855.svg
   :target: https://zenodo.org/badge/latestdoi/110845855
   :alt: DOI

.. image:: https://circleci.com/gh/ME-ICA/tedana.svg?style=shield
   :target: https://circleci.com/gh/ME-ICA/tedana
   :alt: CircleCI

.. image:: http://img.shields.io/badge/License-LGPL%202.0-blue.svg
   :target: https://opensource.org/licenses/LGPL-2.1
   :alt: License

.. image:: https://readthedocs.org/projects/tedana/badge/?version=latest
   :target: http://tedana.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://codecov.io/gh/me-ica/tedana/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/me-ica/tedana
   :alt: Codecov

.. image:: https://badges.gitter.im/ME-ICA/tedana.svg
   :target: https://gitter.im/ME-ICA/tedana
   :alt: Join the chat

About
-----

``tedana`` originally came about as a part of the `ME-ICA`_ pipeline.
The ME-ICA pipeline originally performed both pre-processing and TE-dependent
analysis of multi-echo fMRI data; however, ``tedana`` now assumes that you're
working with data which has been previously preprocessed.
If you're in need of a preprocessing pipeline, we recommend
`fmriprep`_, which has been tested
for compatibility with multi-echo fMRI data and ``tedana``.

.. image:: https://user-images.githubusercontent.com/7406227/40031156-57b7cbb8-57bc-11e8-8c51-5b29f2e86a48.png
  :target: http://tedana.readthedocs.io/

.. image:: /_static/tedana-poster.png

.. _ME-ICA: https://github.com/me-ica/me-ica
.. _fmriprep: https://github.com/poldracklab/fmriprep/

Citations
---------

When using tedana, please include the following citations:

   .. raw:: html

      <script language="javascript">
      var version = 'latest';
      function fillCitation(){
         $('#tedana_version').text(version);

         function cb(err, zenodoID) {
            getCitation(zenodoID, 'vancouver-brackets-no-et-al', function(err, citation) {
               $('#tedana_citation').text(citation);
            });
            getDOI(zenodoID, function(err, DOI) {
               $('#tedana_doi_url').text('https://doi.org/' + DOI);
               $('#tedana_doi_url').attr('href', 'https://doi.org/' + DOI);
            });
         }

         if(version == 'latest') {
            getLatestIDFromconceptID("1250561", cb);
         } else {
            getZenodoIDFromTag("1250561", version, cb);
         }
      }
      </script>
      <p>
      <span id="tedana_citation">tedana</span> Available from: <a id="tedana_doi_url" href="https://doi.org/10.5281/zenodo.1250561">https://doi.org/10.5281/zenodo.1250561</a>
      <img src onerror='fillCitation()' alt=""/>

      <p>
      2. Kundu, P., Inati, S. J., Evans, J. W., Luh, W. M. & Bandettini, P. A. (2011).
      <a href=http://dx.doi.org/10.1016/j.neuroimage.2011.12.028>Differentiating BOLD and non-BOLD signals in fMRI time series using multi-echo EPI.</a>
      <i>NeuroImage</i>, <i>60</i>, 1759-1770.
      </p>
      <p>
      3. Kundu, P., Brenowitz, N. D., Voon, V., Worbe, Y., Vértes, P. E., Inati, S. J.,
      Saad, Z. S., Bandettini, P. A., & Bullmore, E. T. (2013).
      <a href=http://dx.doi.org/10.1073/pnas.1301725110>Integrated strategy for improving functional connectivity mapping using multiecho fMRI.</a>
      <i>Proceedings of the National Academy of Sciences</i>, <i>110</i>, 16187-16192.
      </p>

Alternatively, you can automatically compile relevant citations by running your
tedana code with `duecredit`_. For example, if you plan to run a script using
tedana (in this case, ``tedana_script.py``):

.. code-block:: bash

 python -m duecredit tedana_script.py

You can also learn more about `why citing software is important`_.

.. _Differentiating BOLD and non-BOLD signals in fMRI time series using multi-echo EPI.: http://dx.doi.org/10.1016/j.neuroimage.2011.12.028
.. _Integrated strategy for improving functional connectivity mapping using multiecho fMRI.: http://dx.doi.org/10.1073/pnas.1301725110
.. _duecredit: https://github.com/duecredit/duecredit
.. _`why citing software is important`: https://www.software.ac.uk/how-cite-software

Installation
------------

You'll need to set up a working development environment to use ``tedana``.
To set up a local environment, you will need Python >=3.6 and the following
packages will need to be installed:

- mdp
- nilearn
- nibabel>=2.1.0
- numpy
- scikit-learn
- scipy

You can then install ``tedana`` with:

.. code-block:: bash

  pip install tedana

Getting involved
----------------

We 💛 new contributors!
To get started, check out `our contributing guidelines`_.

Want to learn more about our plans for developing ``tedana``?
Have a question, comment, or suggestion?
Open or comment on one of `our issues`_!

We ask that all contributions to ``tedana`` respect our `code of conduct`_.

.. _our contributing guidelines: https://github.com/ME-ICA/tedana/blob/master/CONTRIBUTING.md
.. _our issues: https://github.com/ME-ICA/tedana/issues
.. _code of conduct: https://github.com/ME-ICA/tedana/blob/master/CODE_OF_CONDUCT.md

License Information
-------------------

tedana is licensed under GNU Lesser General Public License version 2.1.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   multi-echo
   usage
   approach
   outputs
   contributing
   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
