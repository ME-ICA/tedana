Usage
=====

tedana minimally requires:

#. acquired echo times (in milliseconds), and
#. functional datasets equal to the number of acquired echoes.

But you can supply many other options, viewable with ``tedana -h``.

Command line options
--------------------
.. argparse::
   :ref: tedana.get_parser
   :prog: tedana
   :nodefault:
   :nodefaultconst:
