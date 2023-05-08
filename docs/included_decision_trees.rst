#######################
Included Decision Trees
#######################

Two decision trees are currently distributed with ``tedana``.

``kundu`` is the decision tree that is based on MEICA version 2.5
and has been included with ``tedana`` since the start of this project.
While multiple publications have used and benefits from this decision,
tree, but it includes many steps with arbitrary thresholds and, when
components seem misclassified, it's often hard to understand why.

``minimal`` is a simplified version of that decision tree with fewer
steps and arbitrary thresholds. Minimal is designed to be more stable
and comprehensible, but it has not yet be extensively validated and
parts of the tree may change in response to additional tests on a
wider range of data sets.

Flowcharts describing the steps in both trees are below.
As documented more in :doc:`building_decision_trees`, the input to each tree
is a table with metrics, like :math:`\kappa` or :math:`\rho`, for each
component. Each step or node in the decision tree either calculates
new values or changes component classifications based on these metrics.
When a component classification changes to ``accept`` or ``reject``, a
``classification_tag`` is also assigned which may help understand why
a component was given a specific classification.

Each step in the flow chart is labeled with a ``node`` number.
If ``tedana`` is run using one of these trees, those node
numbers will match the numbers in the ``ICA status table`` and the
``ICA decision tree`` that are :doc:`output_file_descriptions`.
These node numbers can be used to see when in the process a component's
classifiation changed.

.. image:: _static/decision_tree_legend.png
    :width: 300
    :alt: Legend for Decision Tree Flow Charts

..
    Reminder on how to load svg if I can figure out how to correctly generate them
    .. raw:: html

        <img src = "_static/decision_tree_legend.svg" alt="Legend for Decision Tree Flow Charts"/>


*******************
Kundu decision tree
*******************

Nodes 1-5 reject components that are very unlikely to be BOLD.
In nodes 9-10 components where :math:`\kappa` >
:math:`\kappa` elbow and :math:`\rho` < :math:`\rho`
elbow are classified as `provisional accept`. A non-obvious aspect
of this decision tree is that no decision node below this point distinguishes
components that are `provisional accept` from components that are still
`unclassified` and nothing that does not cross the :math:`\kappa` and
:math:`\rho` elbow thresholds is inherantly rejected. The number of
`provisional accept` components is used to see if the process should
be restarted (node 11) and calculate other thresholds (nodes 12-16 & 20),
but nothing is directly accepted or rejected based on the elbow thresholds.
Several additional criteria are used to reject components (nodes 17, 21, & 22).
In older versions of ``tedana`` components were classified as `ignored`.
This meant too small/minor to lose a degree of freedom by rejecting so treat
like the `accepted` components. This was widely confusing to many users so they
are now classified as `accepted` but with classification tags `low variance`
(node 18) or `accept borderline` (nodes 24 & 25).

.. image:: _static/decision_tree_kundu.png
    :width: 400
    :alt: Kundu Decision Tree Flow Chart

`LaTeX file to generate the kundu decision tree flow chart`_

.. _LaTeX file to generate the kundu decision tree flow chart: _static/decision_tree_kundu.tex

*********************
Minimal decision tree
*********************

The minimal tree starts similarly to the kundu tree by rejecting components
that are very unlikely to be BOLD (nodes 1-5). Then all components where
:math:`\kappa` > :math:`\kappa` elbow and :math:`\rho` < :math:`\rho` elbow
are `provisional accept` and otherwise are `provisional reject` (nodes 8 & 10).
The only expection to this is if :math:`\kappa` > :math:`\kappa` elbow and
:math:`\kappa` > 2* :math:`\rho` than it is `provisional accept` regardless of the
:math:`\rho` elbow under the assumption that there is enough T2* weighted signal
the component should not be rejected even if it also contains noise (node 9).
If `provisional reject` components have very low variance they are accepted rather
than losing degrees of freedom, but no more than 1% of the total variance can be
accepted this way (node 11). After that point, everything that is
`provisional accept` is accepted (node 12) and everything that is `provisional reject`
is rejected (node 13)

.. image:: _static/decision_tree_minimal.png
    :width: 400
    :alt: Minimal Decision Tree Flow Chart

`LaTeX file to generate the minimal decision tree flow chart`_

.. _LaTeX file to generate the minimal decision tree flow chart: _static/decision_tree_minimal.tex
