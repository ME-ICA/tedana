.. _spreadsheet of publications:

ME-fMRI Parameters & Publications
=================================

The following page highlights a selection of parameters collected from published papers that have
used multi-echo fMRI. The subsequent spreadsheet is an on-going effort to track all of these publication. 
This is a volunteer led effort so, if you know of a excluded publication, whether or not it is yours, 
please add it.

.. plot::

    import matplotlib.pyplot as plt
    import pandas as pd
    metable = pd.read_csv('https://docs.google.com/spreadsheets/d/1WERojJyxFoqcg_tndUm5Kj0H1UfUc9Ban0jFGGfPaBk/export?gid=0&format=csv',
                           header=0 )
    TEs = [metable.TE1.mean(), metable.TE2.mean(), metable.TE3.mean(), metable.TE4.mean(), metable.TE5.mean()]
    TE_labels = ['TE1', 'TE2', 'TE3', 'TE4', 'TE5']
    plt.bar([1, 2, 3, 4, 5], TEs)
    plt.title('Echo Times at 3T', fontsize=16)
    pub_count = metable.TE1.count()
    plt.text(0.5,60, 'Average from {} studies'.format(pub_count))
    plt.xlabel('Echo Number')
    plt.ylabel('Echo Time (ms)')
    plt.show()

You can view and suggest additions to this spreadsheet `here`_

.. raw:: html

    <iframe style="position: absolute; height: 60%; width: 60%; border: none" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vS0nEVp27NpwdzPunvMLflyKzcZbCo4k2qPk5zxEiaoJTD_IY1OGbWICizogAEZlTyL7d_7aDA92uwf/pubhtml?widget=true&amp;headers=false"></iframe>

.. _here: https://docs.google.com/spreadsheets/d/1WERojJyxFoqcg_tndUm5Kj0H1UfUc9Ban0jFGGfPaBk/edit#gid=0