.. _spreadsheet of publications:

Multi-echo fMRI Publications
============================

The following spreadsheet is an on-going effort to track all publications that
use multi-echo fMRI. This is a volunteer led effort so, if you know of an
excluded publication, whether or not it is yours, please add it.

The figure below highlights the average TEs used in currently published papers using multiecho 
at 3T. 


.. plot::

   import matplotlib.pyplot as plt
   import pandas as pd
    metable= = pd.read_csv('https://docs.google.com/spreadsheets/d/1WERojJyxFoqcg_tndUm5Kj0H1UfUc9Ban0jFGGfPaBk/export?gid=0&format=csv',
                   header=0
                  )
   TEs = [metable.TE1.mean(), metable.TE2.mean(), metable.TE3.mean(), metable.TE4.mean(), metable.TE5.mean()]
   plt.bar([1, 2, 3, 4, 5], TEs)
   plt.show()


You can view and suggest additions to this spreadsheet `here`_

.. raw:: html

    <iframe style="position: absolute; height: 60%; width: 60%; border: none" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vS0nEVp27NpwdzPunvMLflyKzcZbCo4k2qPk5zxEiaoJTD_IY1OGbWICizogAEZlTyL7d_7aDA92uwf/pubhtml?widget=true&amp;headers=false"></iframe>

.. _here: https://docs.google.com/spreadsheets/d/1WERojJyxFoqcg_tndUm5Kj0H1UfUc9Ban0jFGGfPaBk/edit#gid=0