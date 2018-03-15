Installation
============

We recommend running `containerized versions`_ of ``meica`` to avoid dependency issues.
The `Docker Engine`_ appropriate for your system (i.e., linux, Mac OSX, or Windows) is required to access and run the container.

.. _Docker Engine: https://docs.docker.com/engine/installation/

It is also possible to run a local or "bare-metal" ``meica`` if your system has `AFNI`_ and python 2.7 or 3.4+ installed.
With a local python installation, we recommend using the `Anaconda`_ or `Miniconda`_ package manager, as these allow for the creation of `virtual environments`_.

.. _AFNI: https://afni.nimh.nih.gov/
.. _Anaconda: https://docs.continuum.io/anaconda/install/
.. _Miniconda: https://conda.io/miniconda.html
.. _virtual environments: https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

Containerized versions
----------------------

To access a containerized version of ``meica`` simply pull the latest Docker image.
This can be accomplished with the following command:

``docker pull emdupre/meica:0.0.1``

Local installation
------------------

Local installation requires the following dependencies:

* Python 2.7 or 3.4+
* AFNI

You can download ``meica`` to your local environment with the command ``pip install meica``.
Alternatively, for "bleeding-edge" features you can clone the latest version of ``meica`` directly from GitHub.
To do so, ``git clone https://github.com/me-ica/me-ica.git``.
