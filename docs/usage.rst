Usage
=====

tedana minimally requires:

#. acquired echo times (in milliseconds), and
#. functional datasets equal to the number of acquired echoes.

But you can supply many other options, viewable with ``tedana -h``.

Run tedana
----------
.. argparse::
   :ref: tedana.cli.run_tedana.get_parser
   :prog: tedana
   :func: get_parser

Run t2smap
----------
.. argparse::
   :ref: tedana.cli.run_t2smap.get_parser
   :prog: t2smap
   :func: get_parser
