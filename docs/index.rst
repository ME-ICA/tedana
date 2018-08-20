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


License Information
-------------------

tedana is licensed under GNU Lesser General Public License version 2.1.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   multi-echo
   approach
   usage
   api
   contributing


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
