.. include:: <isonum.txt>

#############################
tedana: TE Dependent ANAlysis
#############################

.. image:: https://img.shields.io/pypi/v/tedana.svg
   :target: https://pypi.python.org/pypi/tedana/
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/tedana.svg
   :target: https://pypi.python.org/pypi/tedana/
   :alt: PyPI - Python Version

.. image:: https://joss.theoj.org/papers/10.21105/joss.03669/status.svg
   :target: https://doi.org/10.21105/joss.03669
   :alt: JOSS DOI

.. image:: https://zenodo.org/badge/110845855.svg
   :target: https://zenodo.org/badge/latestdoi/110845855
   :alt: Zenodo DOI

.. image:: https://circleci.com/gh/ME-ICA/tedana.svg?style=shield
   :target: https://circleci.com/gh/ME-ICA/tedana
   :alt: CircleCI

.. image:: http://img.shields.io/badge/License-LGPL%202.0-blue.svg
   :target: https://opensource.org/licenses/LGPL-2.1
   :alt: License

.. image:: https://readthedocs.org/projects/tedana/badge/?version=latest
   :target: http://tedana.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://codecov.io/gh/me-ica/tedana/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/me-ica/tedana
   :alt: Codecov

.. image:: http://isitmaintained.com/badge/resolution/ME-ICA/tedana.svg
   :target: http://isitmaintained.com/project/ME-ICA/tedana
   :alt: Average time to resolve an issue

.. image:: http://isitmaintained.com/badge/open/ME-ICA/tedana.svg
   :target: http://isitmaintained.com/project/ME-ICA/tedana
   :alt: Percentage of issues still open

.. image:: https://img.shields.io/badge/Chat%20on-Mattermost-purple.svg
   :target: https://mattermost.brainhack.org/brainhack/channels/tedana
   :alt: Join the chat

.. image:: https://img.shields.io/badge/receive-our%20newsletter%20❤%EF%B8%8F-blueviolet.svg
   :target: https://groups.google.com/g/tedana-newsletter
   :alt: Join our Google Group mailing list

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black


*****
About
*****

``TE``\ -``de``\ pendent ``ana``\ lysis (``tedana``\ )is a Python library for denoising multi-echo functional magnetic resonance imaging (fMRI) data.
``tedana`` originally came about as a part of the `ME-ICA`_ pipeline, although it has since diverged.
An important distinction is that while the ME-ICA pipeline originally performed both pre-processing and TE-dependent analysis of multi-echo fMRI data,
``tedana`` now assumes that you're working with data which has been previously preprocessed.


.. image:: https://user-images.githubusercontent.com/7406227/40031156-57b7cbb8-57bc-11e8-8c51-5b29f2e86a48.png
  :target: http://tedana.readthedocs.io/

For a summary of multi-echo fMRI, which is the imaging technique ``tedana`` builds on,
visit `Multi-echo fMRI`_.

For a detailed procedure of how ``tedana`` analyzes the data from multi-echo fMRI,
visit `Processing pipeline details`_.

.. _ME-ICA: https://github.com/me-ica/me-ica
.. _Multi-echo fMRI: https://tedana.readthedocs.io/en/latest/multi-echo.html
.. _Processing pipeline details: https://tedana.readthedocs.io/en/latest/approach.html#


*********
Citations
*********

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
      <span id="tedana_citation">tedana</span>
      This link is for the most recent version of the code and that page has links to DOIs
      for older versions. To support reproducibility, please cite the version you used:
      <a id="tedana_doi_url" href="https://doi.org/10.5281/zenodo.1250561">https://doi.org/10.5281/zenodo.1250561</a>
      <img src onerror='fillCitation()' alt=""/>
      </p>

      <p>
      2. DuPre, E. M., Salo, T., Ahmed, Z., Bandettini, P. A., Bottenhorn, K. L.,
      Caballero-Gaudes, C., Dowdle, L. T., Gonzalez-Castillo, J., Heunis, S.,
      Kundu, P., Laird, A. R., Markello, R., Markiewicz, C. J., Moia, S.,
      Staden, I., Teves, J. B., Uruñuela, E., Vaziri-Pashkam, M.,
      Whitaker, K., & Handwerker, D. A. (2021).
      <a href=https://doi.org/10.21105/joss.03669>TE-dependent analysis of multi-echo fMRI with tedana.</a>
      <i>Journal of Open Source Software</i>, <i>6(66)</i>, 3669.
      doi:10.21105/joss.03669.
      </p>

      <p>
      3. Kundu, P., Inati, S. J., Evans, J. W., Luh, W. M., & Bandettini, P. A. (2011).
      <a href=https://doi.org/10.1016/j.neuroimage.2011.12.028>Differentiating BOLD and non-BOLD signals in fMRI time series using multi-echo EPI.</a>
      <i>NeuroImage</i>, <i>60</i>, 1759-1770.
      </p>

      <p>
      4. Kundu, P., Brenowitz, N. D., Voon, V., Worbe, Y., Vértes, P. E., Inati, S. J.,
      Saad, Z. S., Bandettini, P. A., & Bullmore, E. T. (2013).
      <a href=https://doi.org/10.1073/pnas.1301725110>Integrated strategy for improving functional connectivity mapping using multiecho fMRI.</a>
      <i>Proceedings of the National Academy of Sciences</i>, <i>110</i>, 16187-16192.
      </p>

Alternatively, you can use the text and citations produced by the tedana workflow.

You can also learn more about `why citing software is important`_.

.. _Differentiating BOLD and non-BOLD signals in fMRI time series using multi-echo EPI.: https://doi.org/10.1016/j.neuroimage.2011.12.028
.. _Integrated strategy for improving functional connectivity mapping using multiecho fMRI.: https://doi.org/10.1073/pnas.1301725110
.. _`why citing software is important`: https://www.software.ac.uk/how-cite-software


*******
Posters
*******

.. image:: /_static/tedana-ohbm2019-poster.png

.. image:: /_static/tedana-ohbm2018-poster.png


*******************
License Information
*******************

tedana is licensed under GNU Lesser General Public License version 2.1.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   multi-echo
   usage
   approach
   outputs
   faq
   building_decision_trees
   support
   contributing
   roadmap
   api
   denoising

.. toctree::
   :hidden:
   :name: hiddentoc

   dependence_metrics
   included_decision_trees


******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
